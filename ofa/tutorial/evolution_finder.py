import copy
import random
from random import sample
from threading import Thread
from tqdm import tqdm
import numpy as np
import time
import torch
from ofa.elastic_nn.utils import set_running_statistics
import torch.backends.cudnn as cudnn
import torch.nn as nn
from threading import Thread
from instu_nas_latency_client import instu_nas_latency_client_V2 as instu_nas_latency_client
from instu_nas_latency_client import instu_nas_accuracy_client
from multiprocessing import Manager
import math
import pickle

__all__ = ['EvolutionFinder']


class ArchManager:
	def __init__(self):
		self.num_blocks = 21
		self.num_stages = 5
		# VPU search space
		self.resolutions = [160, 176, 192, 208, 224]
		self.kernel_sizes = [3,5,7]
		# middle layer channel expand ratio
		self.expand_ratios = [3,4,6]
		# stage blocks number
		self.depths = [2,3,4]

		# TPU search space
		# self.resolutions = [208, 224, 240, 256, 272]
		# self.kernel_sizes = [3, 5, 7]
		# self.expand_ratios = [3, 4, 6]
		# self.depths = [2, 3, 4]

	def random_sample(self):
		sample = {}
		d = []
		e = []
		ks = []
		for i in range(self.num_stages):
			d.append(random.choice(self.depths))

		for i in range(self.num_blocks):
			e.append(random.choice(self.expand_ratios))
			ks.append(random.choice(self.kernel_sizes))

		sample = {
			'wid': None,
			'ks': ks,
			'e': e,
			'd': d,
			'r': [random.choice(self.resolutions)]
		}

		return sample

	def random_resample(self, sample, i):
		assert i >= 0 and i < self.num_blocks
		sample['ks'][i] = random.choice(self.kernel_sizes)
		sample['e'][i] = random.choice(self.expand_ratios)

	def random_resample_depth(self, sample, i):
		assert i >= 0 and i < self.num_stages
		sample['d'][i] = random.choice(self.depths)

	def random_resample_resolution(self, sample):
		sample['r'][0] = random.choice(self.resolutions)


class EvolutionFinder:
	valid_constraint_range = {
		'flops': [150, 600],
		'note10': [15, 60],
	}

	def __init__(self, constraint_type, efficiency_constraint,
	             efficiency_predictor, accuracy_predictor, **kwargs):
		self.constraint_type = constraint_type
		if not constraint_type in self.valid_constraint_range.keys():
			self.invite_reset_constraint_type()
		self.efficiency_constraint = efficiency_constraint
		if not (efficiency_constraint <= self.valid_constraint_range[constraint_type][1] and
		        efficiency_constraint >= self.valid_constraint_range[constraint_type][0]):
			self.invite_reset_constraint()

		self.efficiency_predictor = efficiency_predictor
		self.accuracy_predictor = accuracy_predictor
		self.arch_manager = ArchManager()
		self.num_blocks = self.arch_manager.num_blocks
		self.num_stages = self.arch_manager.num_stages

		self.mutate_prob = kwargs.get('mutate_prob', 0.1)
		self.population_size = kwargs.get('population_size', 100)
		self.max_time_budget = kwargs.get('max_time_budget', 500)
		self.parent_ratio = kwargs.get('parent_ratio', 0.25)
		self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)
		self.target_platform = kwargs.get('target_platform', 'CPU')
		# added evaluate trace
		self.lat_count = 0
		self.acc_count = 0
		self.ofa_net = kwargs.get('super_net', None)
		self.input_sz = [160,176,192,208,224]
		self.inp_dic = {}
		self.lbl_dic = {}
		self.LUT = {}
		self.acc_gpu = torch.empty(self.population_size)
		sample_num = 1000
		self.parents_size = int(round(self.parent_ratio * self.population_size))
		sample_split = 1
		load_start = time.time()
		# no GPU currently
		# accuracy_server_ip = "159.226.41.50"
		# accuracy_server_port = 20026
		# self.acc_client = instu_nas_accuracy_client(accuracy_server_ip, accuracy_server_port)
		estimator_server_ip = "127.0.0.1"
		estimator_server_port = 20027
		self.epoch = 0
		self.exploration_table_name = 'exploration_table_{}_{}ms.pkl'.format(self.target_platform, efficiency_constraint)
		self.exploration_table = {}
		self.LUT_name = 'LUT_{}_{}ms.pkl'.format(self.target_platform, efficiency_constraint)
		self.latency_client = instu_nas_latency_client(estimator_server_ip, estimator_server_port)
		self.acc_actual_run = kwargs.get('acc_actual_run', False)
		if self.acc_actual_run:
			self.log_file = open(f'search_log_{self.target_platform}_{self.efficiency_constraint}ms.txt', 'w')
		else:
			self.log_file = open(f'search_log_no_run_{self.target_platform}_{self.efficiency_constraint}ms.txt', 'w')
		load_end = time.time()
		print("Loading done in {}".format(load_end-load_start))

	def clean(self):
		print("Latency eval count {}\n ACC eval count {}\n".format(self.lat_count, self.acc_count))
		if self.acc_actual_run:
			self.log_file.close()


	def arch_key(self, arch):
		ks_key = "-".join(str(x) for x in arch['ks'][:4])
		e_key = '-'.join(str(x) for x in arch['e'][:4])
		stage_depth = arch['d'][:3]
		stage_depth.append(sum(arch['d'][-2:]))
		stage_key = '-'.join(str(x) for x in stage_depth)
		resolution_key = str(arch['r'][0])
		return ks_key, e_key, stage_key, resolution_key

	def arch_large_than(self, arch1, arch2):
		# return true if arch1 > arch
		is_valid = 0
		ks_key, e_key, stage_key, resolution_key = self.arch_key(arch1)
		# invalid_arch_list = self.exploration_table['-'.join([ks_key, e_key])]
		_, _, invalid_stage_key, invalid_resolution_key = self.arch_key(arch2)
		stage_depth = stage_key.split('-')
		invalid_stage_depth = invalid_stage_key.split('-')
		# first check same input size with more layers
		if invalid_resolution_key == resolution_key:
			is_valid = all(int(stage_depth[i]) >= int(invalid_stage_depth[i]) for i in range(len(stage_depth))) and sum([int(x) for x in stage_depth]) > sum([int(x) for x in invalid_stage_depth])
		# check same depth with larger input size
		elif invalid_stage_key == stage_key:
			is_valid = int(resolution_key) > int(invalid_resolution_key)
		# check larger input size and deeper structure
		else :
			check1 = int(resolution_key) > int(invalid_resolution_key)
			check2 = ( all(int(stage_depth[i]) >= int(invalid_stage_depth[i]) for i in range(len(stage_depth))) and sum([int(x) for x in stage_depth]) > sum([int(x) for x in invalid_stage_depth]) )
			is_valid = check1 and check2
		return is_valid

	def check_arch_invalid(self, arch, stage:str=None):
		# check whether the architecture is larger than invalid architectures
		is_invalid = False
		ks_key, e_key, stage_key, resolution_key = self.arch_key(arch)
		invalid_arch_list = self.exploration_table.get('-'.join([ks_key, e_key]), [])
		stage_depth = stage_key.split('-')
		for invalid_arch_info in invalid_arch_list:
			invalid_arch, invalid_stage_key, invalid_resolution_key, add_epoch, efficiency = invalid_arch_info
			invalid_stage_depth = invalid_stage_key.split('-')
			# first check same input size with more layers
			if invalid_resolution_key == resolution_key:
				is_invalid = all(int(stage_depth[i]) >= int(invalid_stage_depth[i]) for i in range(len(stage_depth))) and sum([int(x) for x in stage_depth]) > sum([int(x) for x in invalid_stage_depth])
			# check same depth with larger input size
			elif invalid_stage_key == stage_key:
				is_invalid = int(resolution_key) > int(invalid_resolution_key)
			# check larger input size and deeper structure
			else :
				check1 = int(resolution_key) > int(invalid_resolution_key)
				check2 = ( all(int(stage_depth[i]) >= int(invalid_stage_depth[i]) for i in range(len(stage_depth))) and sum([int(x) for x in stage_depth]) > sum([int(x) for x in invalid_stage_depth]) )
				check3 = str(arch) == str(invalid_arch)		# same arch
				is_invalid = (check1 and check2) or check3
			if is_invalid: # early stop
				# print("Epoch : {} Stage: {} Detecting invalid arch : {}, baseline arch : {}".format(self.epoch, stage, arch, invalid_arch_info))
				return is_invalid
		return is_invalid

	def add_to_exploration_history(self, arch, efficiency):
		# add the arch to the invalid arch exploration list
		# print("Epoch : {}, Adding arch : {} ".format(self.epoch, arch))
		ks_key, e_key, stage_key, resolution_key = self.arch_key(arch)
		table_key = '-'.join([ks_key, e_key])
		if table_key not in self.exploration_table:
			self.exploration_table[table_key] = []
		invalid_arch_list = self.exploration_table[table_key]
		arch_info = [arch, stage_key, resolution_key, self.epoch, efficiency]
		# test arch is good to add

		# check arch in the list can be removed
		invalid_arch_remove_idx = []
		for idx, invalid_arch_info in enumerate(invalid_arch_list):
			invalid_arch, invalid_stage_key, invalid_resolution_key, add_epoch, efficiency = invalid_arch_info
			if self.arch_large_than(invalid_arch, arch):
				invalid_arch_remove_idx.append(idx)
		for idx in invalid_arch_remove_idx[::-1]:
			# print("Epoch : {} Deleting idx : {}, arch : {}".format(self.epoch, idx, invalid_arch_list[idx]))
			del invalid_arch_list[idx]
		invalid_arch_list.append(arch_info)
		# print("After deleting list le

	def validate(self, net, input_data, label_data):
		cudnn.benchmark = True
		net.eval()
		topk = 1
		res = []

		with torch.no_grad():
			for i, images in enumerate(input_data):
				images = images
				labels = label_data[i]
				# compute output
				output = net(images)
				# measure accuracy 
				_, pred = output.topk(topk, 1, True, True)
				pred = pred.t()
				correct = pred.eq(labels.view(1, -1).expand_as(pred))

				correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
				res.append(correct_k)

		return torch.stack(res).sum().div_(10.)

	def random_samples_pool(self, sample_num, base_population):
		samples_pool = [self.arch_manager.random_sample() for i in range(sample_num)]
		return samples_pool

	def crossover_samples_pool(self, sample_num, base_population):
		samples_pool = []
		par_sample1 = base_population[np.random.randint(self.parents_size)][1]
		par_sample2 = base_population[np.random.randint(self.parents_size)][1]
		for i in range(sample_num):
			new_sample = copy.deepcopy(par_sample1)
			for key in new_sample.keys():
				if not isinstance(new_sample[key], list):
					continue
				for i in range(len(new_sample[key])):
					new_sample[key][i] = random.choice([par_sample1[key][i], par_sample2[key][i]])
			samples_pool.append(new_sample)
		return samples_pool
	
	def mutate_samples_pool(self, sample_num, base_population):
		samples_pool = []
		par_sample = base_population[np.random.randint(self.parents_size)][1]
		for i in range(sample_num):
			new_sample = copy.deepcopy(par_sample)

			if random.random() < self.mutate_prob:
				self.arch_manager.random_resample_resolution(new_sample)

			for i in range(self.num_blocks):
				if random.random() < self.mutate_prob:
					self.arch_manager.random_resample(new_sample, i)

			for i in range(self.num_stages):
				if random.random() < self.mutate_prob:
					self.arch_manager.random_resample_depth(new_sample, i)
			samples_pool.append(new_sample)
		return samples_pool

	def acc_eval_request(self, samples):
		self.acc_client.send_samples(samples)

	def latency_eval_request(self, samples):
		self.latency_client.eval_subnet_list(samples)

	def samples_eval(self, samples):
		if len(samples) == 0:
			return []
		populations = []
		latencys = []
		# arch_latency_dict = {}
		arch_acc_dict = {}
		
		try:
			start = time.time()
			print("sending eval request")
			# self.acc_eval_request(samples)
			self.latency_eval_request(samples)
			# self.latency_client.eval_subnet_list_local(samples)
			send_done = time.time()
			print("receving ", send_done - start)
			# accs = self.acc_client.recv_results()
			# no GPU, TODO: replace by on-device accuracy testing
			accs = [80 for i in range(len(samples))]
			print("wait ret")
			latencys = self.latency_client.get_latencys()
			# print('receive done')
			self.acc_count += len(samples)
			self.lat_count += len(samples)

			for arch, acc in accs:		# (arch:str, acc:float)
				arch_acc_dict[arch] = acc
			for arch, latency, std in latencys: # (arch:dict, latency:float, std:float)
				# arch_latency_dict[str(arch)] = (latency, std)
				populations.append((arch_acc_dict[str(arch)], arch, latency, std))
				
			end = time.time()
			print("Epoch : {} sample eval time : {:.3f}, evaled {} samples, average time : {:.3f}".format(self.epoch, end-start, len(samples), (end-start)/len(samples)))
		except Exception as e:
			print("samples_eval Error!", e)
		return populations	# (acc:float, arch:dict, latency:float, std:float)
		
	def get_valid_population(self, population_size, sample_pool_method, base_population):
		# generate population list with at least population_size
		population = []
		constraint = self.efficiency_constraint
		valid_rate = 1
		try: 
			while len(population) < population_size:
				print('population size ', len(population))
				print(population_size)
				eval_num = population_size-len(population)
				pool_size = math.ceil(eval_num / valid_rate)
				# pool_size = population_size
				# print("eval_num ", eval_num, 'pool size', pool_size)
				samples_pool = sample_pool_method(pool_size, base_population)
				eval_samples = []

				# lookup for LUT
				lut_cnt = 0
				for sample in samples_pool:
					if str(sample) in self.LUT:
						acc, latency, std = self.LUT[str(sample)]
						if latency <= constraint:
							population.append((acc, sample, latency, std))
						lut_cnt += 1
					else:
						eval_samples.append(sample)

				# eval unseen samples 
				assert(lut_cnt + len(eval_samples) == len(samples_pool))
				# print("eval sample num : ", len(eval_samples))
				eval_result = self.samples_eval(eval_samples)

				# add valid population 
				unseen_samples = []
				for acc, sample, latency, std in eval_result:
					if str(sample) not in self.LUT:
						self.LUT[str(sample)] = (acc, latency, std)
						unseen_samples.append(sample)
					if latency <= constraint:
						population.append((acc, sample, latency, std))
					else :
						self.add_to_exploration_history(sample, latency)
					# is_invalid = self.check_arch_invalid(sample) 
					# self.log_file.write(str((self.epoch, sample, latency, acc, is_invalid))+'\n')
				
				# statistic the exploration history
				for sample in samples_pool:
					is_invalid = self.check_arch_invalid(sample) 
					acc, latency, std = self.LUT[str(sample)]
					unseen = sample in unseen_samples
					self.log_file.write(str((self.epoch, sample, latency, acc, is_invalid, unseen))+'\n')
				valid_cnt = len(population) - population_size + eval_num  # valid arch cnt in thie eval_samples
				valid_rate = (valid_cnt / pool_size) * 0.3 + valid_rate * 0.7 # smooth 
				# print('valid cnt : ', valid_cnt,' valid rate : ', valid_rate, 'eval_num : ', population_size-len(population))
				valid_rate = max(0.8, valid_rate)  # maximum 2
		except Exception as e:
			print("get_valid_population Error!", e)
		return population
	
	def run_evolution_search(self, verbose=True):
		"""Run a single roll-out of regularized evolution to a fixed time budget."""
		max_time_budget = self.max_time_budget
		population_size = self.population_size
		mutation_numbers = int(round(self.mutation_ratio * population_size))

		best_valids = [-100]
		population = []  # (validation, sample, latency) tuples
		best_info = None
		if verbose:
			print('Generate random population...')

		population = self.get_valid_population(population_size, self.random_samples_pool, population)
		if verbose:
			print('Start Evolution...')
		best_arch_each_gen_recorder = open('best_arch_each_gen_{}_{}ms.txt'.format(self.target_platform, self.efficiency_constraint), 'w')
		# After the population is seeded, proceed with evolving the population.
		for iter in tqdm(range(max_time_budget), desc='Searching with %s constraint (%s)' % (self.constraint_type, self.efficiency_constraint)):
			parents = sorted(population, key=lambda x: x[0])[::-1][:self.parents_size]
			self.epoch = iter
			acc = parents[0][0]
			if verbose:
				print("Parents : ")
				acc_lat_pair = []
				for p in parents:
					acc_lat_pair.append((p[0], p[2]))
				print(acc_lat_pair)
				print('Iter: {} Acc: {}'.format(iter - 1, parents[0][0]))

			assert isinstance(acc, float)
			if acc > best_valids[-1]:
				best_valids.append(acc)
				best_info = parents[0]
				best_arch_each_gen_recorder.write(str(parents[0])+'\n')
			else:
				best_valids.append(best_valids[-1])

			population = parents
			# print("mutate next generation")
			mutatation_population = self.get_valid_population(mutation_numbers, self.mutate_samples_pool, population)
			# print("crossover next generation")
			crossover_population = self.get_valid_population(population_size - mutation_numbers, self.crossover_samples_pool, population)
			population = mutatation_population + crossover_population
			for p in population:
				assert p[2]<self.efficiency_constraint
			# print("Populations : ", population)
			with open(self.exploration_table_name, 'wb') as f:
				pickle.dump(self.exploration_table, f)
			with open(self.LUT_name, 'wb') as f:
				pickle.dump(self.LUT, f)
		best_arch_each_gen_recorder.close()
		return best_valids, best_info
