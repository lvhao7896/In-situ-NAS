# This script runs the overall search process
# Target platform : 
import os
import torch
import numpy as np
import time
import random

from ofa.elastic_nn.utils import set_running_statistics

from ofa.model_zoo import ofa_net
from ofa.utils import download_url
# print("fuck")
from ofa.tutorial import EvolutionFinder


# os.environ['CUDA_VISIBLE_DEVICES'] = '12'
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print('Using GPU.')
else:
    print('Using CPU.')

# ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
print('The OFA Network is ready.')
if cuda_available:
    # path to the ImageNet dataset
    print("Please input the path to the ImageNet dataset.\n")
    imagenet_data_path = '/home/lvhao/github/once-for-all/eval_datasets'
    # if 'imagenet_data_path' is empty, download a subset of ImageNet containing 2000 images (~250M) for test
    if not os.path.isdir(imagenet_data_path):
        os.makedirs(imagenet_data_path, exist_ok=True)
        download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_cvpr_tutorial/imagenet_1k.zip', model_dir='data')
        os.system(' cd data && unzip imagenet_1k 1>/dev/null && cd ..')
        os.system(' cp -r data/imagenet_1k/* {}'.format(imagenet_data_path))
        os.system(' rm -rf data')
        print('%s is empty. Download a subset of ImageNet for test.' % imagenet_data_path)

    print('The ImageNet dataset files are ready.')
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

# accuracy predictor
# accuracy_predictor = AccuracyPredictor(
#     pretrained=True,
#     device='cuda:0' if cuda_available else 'cpu'
# )
accuracy_predictor = None
target_hardware = 'note10'
# latency_table = LatencyTable(device=target_hardware)
# print('The Latency lookup table on %s is ready!' % target_hardware)
latency_table = None

""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
latency_constraint = 15  # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
# N = 500  # How many generations of population to be searched
N = 100
r = 0.25  # The ratio of networks that are used as parents for next generation
# latency_constraints = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
latency_constraints = [30]
# test use
Ps = [10]
Ns = [4]
result_lis = []
target_platform = 'VPU'
for latency_constraint in latency_constraints:
    # set random seed
    random_seed = 2
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print('Successfully imported all packages and configured random seed to %d!'%random_seed)
    if cuda_available:
        torch.cuda.manual_seed(random_seed)
    for idx in range(len(Ps)):
        P = Ps[idx]
        N = Ns[idx]
        params = {
            'constraint_type': target_hardware, # Let's do FLOPs-constrained search
            'efficiency_constraint': latency_constraint,
            'mutate_prob': 0.1, # The probability of mutation in evolutionary search
            'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
            'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
            'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
            'population_size': P,
            'max_time_budget': N,
            'parent_ratio': r,
            # 'super_net': ofa_network,
            'acc_actual_run': True,
            'target_platform' : target_platform
        }

        # build the evolution finder
        finder = EvolutionFinder(**params)

        # start searching
        st = time.time()
        subnet_list = []
        
        # subnet_list.append({'wid': None, 'ks': [5, 3, 7, 3, 7, 5, 7, 3, 7, 7, 3, 3, 5, 5, 3, 3, 3, 3, 5, 3, 5], 'e': [4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 6, 6, 3, 3, 6], 'd': [2, 2, 3, 3, 4], 'r': [192]})
        # subnet_list.append({'wid': None, 'ks': [5, 5, 5, 5, 5, 3, 5, 5, 3, 3, 5, 3, 5, 3, 5, 3, 5, 3, 3, 3, 3], 'e': [4, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3], 'd': [3, 3, 3, 4, 4], 'r': [192]})
        # print(finder.samples_eval(subnet_list))
        # exit()
        best_valids, best_info = finder.run_evolution_search()
        # finder.print_statistic()
        result_lis.append(best_info)
        ed = time.time()
        print('Found best architecture %s  with latency <= %.2f ms in %.2f seconds! '
            'It achieves %.2f%s accuracy with %.2f ms latency .' %
            (target_hardware, latency_constraint, ed-st, best_info[0], '%', best_info[-2]))

        # visualize the architecture of the searched sub-net
        _, net_config, latency, std = best_info
        # ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
        print("latency constraint : ", latency_constraint)
        print("accuracy eval count ", finder.acc_count)
        print("latency eval count ", finder.lat_count)
        print(best_info)
        finder.clean()


print("Validating result list : ", len(result_lis))
ofa_network = ofa_net('ofa_proxyless_d234_e346_k357_w1.3', pretrained=True).eval()
sample_num=1000
sample_split=2
for result in result_lis:
    _, net_config, latency, std = result
    ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
    subnet = ofa_network.get_active_subnet().eval()
    isz = net_config['r'][0]
    data_loader = torch.Tensor(np.load('inp_sample_{}.npz'.format(isz))['arr_0']).view((sample_split,int(sample_num/sample_split),3,isz,isz))
    set_running_statistics(subnet, data_loader)
    lbl_loader = torch.LongTensor(np.load('lbl_sample_{}.npz'.format(isz))['arr_0']).view((sample_split,int(sample_num/sample_split)))
    top1 = finder.validate(subnet, data_loader, lbl_loader)
    print("top 1 : ", top1)
