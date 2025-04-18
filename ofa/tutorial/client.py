GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'

try:
    from openvino.inference_engine import IENetwork, ExecutableNetwork, IECore
    import openvino.inference_engine.ie_api
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)

import time
import os
from sys import argv
import numpy as np
import time
import socket
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
from  proxyless_nets_tf2 import ProxylessNASNets_tf2
from queue import *
from ofa.model_zoo import ofa_net
import ast
from proxyless_nets_tf2 import Conv_BN_act_tf2, FCLayer_tf2, MobileInvertedResidualBlock
from multiprocessing.managers import BaseManager
from multiprocessing import Manager
import multiprocessing
from multiprocessing import Queue
import psutil
import tensorflow as tf
tf.keras.backend.set_learning_phase(0)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import psutil

class Model_Converter:
    def __init__(self, output_dir="./model_pool/"):
        self.output_dir = output_dir

    def convert(self, model):
        raise NotImplementedError

class Torch2Keras_Converter(Model_Converter):
    def __init__(self):
        super(Torch2Keras_Converter, self).__init__()
        self.load_supernet()

    def load_supernet(self):
        self.super_net = ofa_net('ofa_proxyless_d234_e346_k357_w1.3', pretrained=True).eval().cpu()

    def torch2keras(self, subnet, input_shape=[160,160,3]):
        subnet_tf = ProxylessNASNets_tf2(subnet.first_conv, subnet.blocks, subnet.feature_mix_layer, subnet.classifier, input_shape)
        return subnet_tf
    
    def subnet_build(self, arch_desc):
        self.super_net.set_active_subnet(ks=arch_desc['ks'], d=arch_desc['d'], e=arch_desc['e'])
        subnet = self.super_net.get_active_subnet().eval().cpu()
        return subnet

    def keras_freeze(self, subnet_tf, save_name='eval_model', input_shape=[None,160,160,3]):
        full_model = tf.function(lambda x : subnet_tf(x))
        # full_model = full_model.get_concrete_function(tf.TensorSpec(shape=[None, input_shape, input_shape, 3], dtype=tf.float32))
        full_model = full_model.get_concrete_function(tf.TensorSpec(shape=input_shape, dtype=tf.float32))
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                            logdir=self.output_dir,
                            name=save_name + '.pb',
                            as_text=False)       

    def convert(self, arch_desc, save_name='eval_model', input_sz=160):
        build_start = time.time()
        subnet = self.subnet_build(arch_desc)
        torch2keras_start = time.time()
        subnet_tf = self.torch2keras(subnet, (input_sz, input_sz, 3))
        freeze_start = time.time()
        self.keras_freeze(subnet_tf, save_name, input_shape=(None,input_sz, input_sz, 3))
        end = time.time()
        # print("Build time : {:.3f}, t2k time : {:.3f}, freeze time : {:.3f}".format(torch2keras_start-build_start,freeze_start-torch2keras_start,end-freeze_start))
 
class Compiler:
    def __init__(self, output_dir='./model_pool/'):
        self.output_dir = output_dir
    def compile(self):
        raise NotImplementedError

class VPU_compiler(Compiler):
    def __init__(self):
        super(VPU_compiler, self).__init__()

    def compile(self, name, inp_shape=[None,160,160,3]):
        # cmd = "mo.py --data_type=FP16 --input_model={}.pb --input_shape=[1,{},{},3] --silent".format(self.subnet_name, inp_sz, inp_sz)
        cmd = "mo_tf.py --data_type=FP16 --input_model=./model_pool/{}.pb --input_shape={} --silent --output_dir {} 1>/dev/null 2>&1".format(name, str(inp_shape).replace(' ', ''), self.output_dir)
        # cmd = "mo_tf.py --data_type=FP16 --input_model=./model_pool/{}.pb --input_shape={} --silent --output_dir {} ".format(name, str(inp_shape).replace(' ', ''), self.output_dir)
        os.system(cmd)

class Estimator:
    def latency_eval(self):
        raise NotImplementedError

class VPU_Estimator(Estimator):
    def __init__(self, model_dir='./model_pool/'):
        super(VPU_Estimator, self).__init__()
        self.ie = IECore()
        self.inp_sample_sz = {}
        self.model_pool_dir = model_dir
        inp_sz = (160, 176, 192, 208, 224)
        for sz in inp_sz:
            self.inp_sample_sz[str(sz)] = np.random.rand(1, 3, sz, sz) # NCHW

    def latency_eval(self, subnet_name):
        xml_path = self.model_pool_dir + subnet_name + '.xml'
        bin_path = self.model_pool_dir + subnet_name+ '.bin'
        net = IENetwork(model=xml_path, weights=bin_path)
        run_times = 100
        input_blob = next(iter(net.inputs))
        # output_blob = next(iter(net.outputs))
        n, c, h, w = net.inputs[input_blob].shape
        assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(net.outputs) == 1, "Sample supports only single output topologies"
        self.exec_net = self.ie.load_network(network=net, device_name = "MYRIAD", num_requests=run_times)
        # infer_start = time.time()
        res = None
        latencys = []
        # warm up
        self.exec_net.requests[0].infer(inputs={input_blob: np.random.rand(n, c, h, w)})
        for request in self.exec_net.requests:
            request.infer(inputs={input_blob: np.random.rand(n, c, h, w)})
            latencys.append(request.latency)
        # infer_end = time.time()
        print(latencys)
        latencys = np.array(latencys)
        lat_mean = np.mean(latencys)
        lat_std =  np.std(latencys)
        print("infer latency {:.4f}ms(mean), {:.4f}ms(std)".format(lat_mean, lat_std))
        latency = lat_mean # ms
        del self.exec_net
        return latency, lat_std

    def clean(self):
        # super(VPU_Estimator, self).clean()
        del self.ie
        
all_start = time.time()

def prepare_eval(arch_desc, name:str, idx:int, estimator_queue:Queue):
    name = name + str(idx)
    p = psutil.Process()
    cpu_list = [0,1,2,3,4,5,6]
    p.cpu_affinity([cpu_list[idx%len(cpu_list)]])
    # print("Run on cpu : ", p.cpu_affinity())
    convert_start = time.time()
    converter = Torch2Keras_Converter()
    converter.convert(arch_desc, save_name=name, input_sz=arch_desc['r'][0])
    end = time.time()
    # print("[Convert] idx : {}, convert_start : {:.3f}, end : {:.3f}, run time : {:.3f}".format(idx, convert_start-all_start, end-all_start, end-convert_start))
    inp_shape = [1, arch_desc['r'][0], arch_desc['r'][0], 3]
    compile_start = time.time()
    compiler = VPU_compiler()
    compiler.compile(name, inp_shape=inp_shape)
    end = time.time()
    estimator_queue.put_nowait((arch_desc, name, idx))
    # print("[Compile] idx : {}, compile_start : {:.3f} end : {:.3f}, run time : {:.3f}".format(idx, compile_start-all_start, end-all_start, end-compile_start))


def eval(estimator_queue:Queue, socket):
    start = time.time()
    p = psutil.Process()
    cpu_list = [7]
    arch_desc, name, idx = estimator_queue.get()
    p.cpu_affinity([cpu_list[idx%len(cpu_list)]])
    eval_start = time.time()
    estimator = VPU_Estimator()
    latency, std = estimator.latency_eval(name)
    end = time.time()
    # resp = str(arch_desc) + ';' + '{:.3f}'.format(latency) + ';' + '{:.3f}'.format(std)
    # resp = resp.encode(encoding='utf-8')
    # assert len(resp) < 1024, 'Long msg not suppoted.'
    # socket.sendall(resp)
    # print("[Eval]  idx : {}, start : {:.3f}, estimate_start : {:.3f}, end : {:.3f}, run time : {:.3f}".format(idx, start-all_start, eval_start-all_start, end-all_start, end-eval_start))
    return (arch_desc, latency, std)

class instu_nas_accuracy_client:
    def __init__(self, server_ip:str, port:int):
        self.server_ip = server_ip
        self.port = port
        self.encoding = 'utf-8'
        self.max_msg_len = 1024
        self.tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def conn_server(self): 
        self.tcp_client_socket.connect((self.server_ip, self.port))  

    def send_samples(self, samples):
        cmd = 'gpu_eval' + ';' +str(len(samples))
        cmd = cmd.encode(encoding='utf-8')
        self.tcp_client_socket.sendall(cmd)
        for sample in samples:
            sample = str(sample).encode(encoding=self.encoding)
            assert(len(sample) < self.max_msg_len), 'Long msg not supportted.'
            self.tcp_client_socket.sendall(sample)

    def recv_results(self, sample_num):
        accs = []
        for i in range(sample_num):
            recv_data = self.tcp_client_socket.recv(self.max_msg_len).decode(self.encoding)
            arch, acc = recv_data.split(';')
            accs.append((arch, acc))

    def clean(self):
        self.tcp_client_socket.close()

class instu_nas_client:
    def __init__(self, server_ip:str, port:int):
        num_processors_to_use = multiprocessing.cpu_count()
        print("Avaiable parallal processors ", num_processors_to_use)
        num_estimator_processing = 1
        num_converter_processing = num_processors_to_use - num_estimator_processing
        self.converter_pool = multiprocessing.Pool(int(num_converter_processing))   
        self.estimator_pool = multiprocessing.Pool(num_estimator_processing,)
        self.compiler_estimator_queue = Manager().Queue()
        self.estimator_res = []
        self.converter_res = []

    def eval_subnet_list(self, net_lists:list):
        converted_num = 0
        while converted_num < len(net_lists):
            self.estimator_res.append(self.apply_async(eval, args=(self.compiler_estimator_queue,)))
            self.converter_res.append(self.converter_pool.apply_async(prepare_eval,  args=(net_lists[converted_num], 'subnet', converted_num, self.compiler_estimator_queue)))
            converted_num += 1
        # Eval left subnets
        print("estimated_num ", converted_num)

    def get_latencys(self):
        latencys = []
        for r in self.estimator_res:
            latencys.append(r.get())
        self.converter_pool.close()
        self.estimator_pool.close()
        self.converter_pool.join()
        self.estimator_pool.join()
        # print(latencys)
        return latencys

class instu_nas_client:
    def __init__(self, server_ip:str, port:int):
        self.server_ip = server_ip
        self.port = port
        self.tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.parallalism = self.estimator.parallalism()

    def conn_server(self): 
        self.tcp_client_socket.connect((self.server_ip, self.port))

    def setup(self):
        self.conn_server()
        num_processors_to_use = multiprocessing.cpu_count()
        print("Avaiable parallal processors ", num_processors_to_use)
        num_estimator_processing = 1
        num_converter_processing = num_processors_to_use - num_estimator_processing
        self.converter_pool = multiprocessing.Pool(int(num_converter_processing))   
        self.estimator_pool = multiprocessing.Pool(num_estimator_processing,)
        self.compiler_estimator_queue = Manager().Queue()
        self.converted_num = 0

    def run(self):
        self.setup()
        while True:
            cmd = self.tcp_client_socket.recv(1024)
            cmd = cmd.decode('utf-8')
            if cmd == 'eval':
                arch = self.tcp_client_socket.recv(1024)
                arch = arch.decode('utf-8')
                arch = ast.literal_eval(arch)
                self.eval_subnet(arch)
            elif cmd == 'end':
                self.clean()
                return 
            else:
                raise NotImplementedError

    def eval_subnet(self, arch):
        self.estimator_pool.apply_async(eval, args=(self.compiler_estimator_queue, self.tcp_client_socket))
        self.converter_pool.apply_async(prepare_eval,  args=(arch, 'subnet', self.converted_num % 100, self.compiler_estimator_queue))
        self.converter_num += 1

    def eval_subnet_list_local(self, net_lists:list):
        prepare_start = time.time()
        num_processors_to_use = multiprocessing.cpu_count()
        print("Avaiable parallal processors ", num_processors_to_use)
        # num_converter_processing = int(num_processors_to_use/4)
        num_estimator_processing = 1
        num_converter_processing = num_processors_to_use - num_estimator_processing
        converter_pool = multiprocessing.Pool(int(num_converter_processing))   
        estimator_pool = multiprocessing.Pool(num_estimator_processing,)

        converter_res = []
        estimator_res = []
        
        prepare_end = time.time()
        estimate_start = time.time()
        converted_num = 0
        estimated_num = 0

        latencys = []
        compiler_estimator_queue = Manager().Queue()
        run_start = time.time()
        while estimated_num < len(net_lists):
            if estimated_num < len(net_lists):
                estimator_res.append(estimator_pool.apply_async(eval, args=(compiler_estimator_queue,)))
                estimated_num += 1

            if converted_num < len(net_lists):
                converter_res.append(converter_pool.apply_async(prepare_eval,  args=(net_lists[converted_num], 'subnet', converted_num, compiler_estimator_queue)))
                converted_num += 1
                print("converted task ", converted_num)
        run_end =time.time()
        final_compile_start = time.time()   
        final_compile_end = time.time() 
        final_eval_start = time.time()
        # Eval left subnets
        print("estimated_num ", estimated_num)
        for r in estimator_res:
            latencys.append(r.get())
        final_eval_end =time.time()
        converter_pool.close()
        estimator_pool.close()
        converter_pool.join()
        estimator_pool.join()

        estimate_end = time.time()
        print("Prepare : {:.3f}, Estimate : {:.3f}".format(prepare_end-prepare_start, estimate_end-estimate_start))
        # print(latencys)
        return latencys

    def eval_subnet_block_local(self, img_size):
        # used to setup the latency LUT for subnet, evaluating the latency of each basic block of the subnet, TEST USE
        net_config = {'wid': None, 'ks': [3, 3, 3, 3, 5, 3, 5, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 3,3], 'e': [3, 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 3, 3, 3, 4, 3, 3, 4, 3, 3,4], 'd': [4, 2, 3, 4, 3], 'r': [160]}
        subnet = self.estimator.super_net.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
        subnet = self.estimator.super_net.get_active_subnet().eval().cpu()

        inp_shape = [1,160,160,3]
        first_conv_tf = Conv_BN_act_tf2(subnet.first_conv.conv, subnet.first_conv.bn, 'relu6')
        self.estimator.converter.keras_freeze(first_conv_tf, "eval_model", inp_shape)
        self.estimator.compile(inp_shape)

        # inp_shape = [1, 80, 80, 40]   # block0
        # # inp_shape = [1, 80, 80, 24]     # block1
        # # inp_shape = [1, 40, 40, 32]     # block2, 3, 4, 5
        # # inp_shape = [1, 20, 20, 56]     # block6, 7
        # # inp_shape = [1, 10, 10, 104]    # blcok8, 9, 10
        # # inp_shape = [1, 10, 10, 128]    # block11, 12, 13, 14
        # # inp_shape = [1, 5, 5, 248]      # block15, 16, 17
        # block_idx = 0
        # # print(len(subnet.blocks))
        # block = MobileInvertedResidualBlock(subnet.blocks[block_idx].mobile_inverted_conv, subnet.blocks[block_idx].shortcut) 
        # self.estimator.converter.keras_freeze(block, "eval_model", inp_shape)
        # self.estimator.compile(inp_shape)

        # classifier infer time 
        # inp_shape = [1,5,5,416] # feature_mix layer
        # classifier = tf.keras.Sequential()
        # classifier.add(Conv_BN_act_tf2(subnet.feature_mix_layer.conv, subnet.feature_mix_layer.bn, 'relu6'))
        # classifier.add(tf.keras.layers.GlobalAveragePooling2D())
        # classifier.add(FCLayer_tf2(subnet.classifier.linear))
        # x = np.random.rand(*inp_shape)
        # y = classifier(x)
        # self.estimator.converter.keras_freeze(classifier, "eval_model", inp_shape)
        # self.estimator.compile(inp_shape)

        # baseline infer time 
        # inp_shape = [1,3,3,1]
        # basic_block = tf.keras.layers.Conv2D(1, (3,3),1,padding='valid', use_bias=False, trainable=False)
        # basic_inp = tf.constant(np.random.rand(*inp_shape).reshape(inp_shape))
        # print(basic_block.get_weights())
        # y = basic_block(basic_inp)
        # print(basic_block.get_weights())
        # self.estimator.converter.keras_freeze(basic_block, "eval_model", inp_shape)
        # self.estimator.compile(inp_shape)

        # inp_shape = [1,160,160,3]
        # subnet_tf = ProxylessNASNets_tf2(subnet.first_conv, subnet.blocks, subnet.feature_mix_layer, subnet.classifier, (img_size, img_size, 3))
        # self.estimator.converter.keras_freeze(subnet_tf,  "eval_model", inp_shape)
        # self.estimator.compile(inp_shape)

        # first_conv + block0 + block1 + block2
        # inp_shape = [1,160,160,3]
        # first_conv_tf = Conv_BN_act_tf2(subnet.first_conv.conv, subnet.first_conv.bn, 'relu6')
        # block0 = MobileInvertedResidualBlock(subnet.blocks[0].mobile_inverted_conv, subnet.blocks[0].shortcut)
        # block1 = MobileInvertedResidualBlock(subnet.blocks[1].mobile_inverted_conv, subnet.blocks[1].shortcut)
        # block2 = MobileInvertedResidualBlock(subnet.blocks[2].mobile_inverted_conv, subnet.blocks[2].shortcut)
        # block3 = MobileInvertedResidualBlock(subnet.blocks[2].mobile_inverted_conv, subnet.blocks[2].shortcut)
        # blocks = tf.keras.Sequential()
        # blocks.add(first_conv_tf)
        # blocks.add(block0)
        # blocks.add(block1)
        # blocks.add(block2)
        # blocks.add(block3)
        # x = np.random.rand(*inp_shape)
        # y = blocks(x)
        # self.estimator.converter.keras_freeze(blocks,  "eval_model", inp_shape)
        # self.estimator.compile(inp_shape)

        latency = self.estimator.latency_eval()
        print(latency)
        # feature_mix_layer = self.super_net.feature_mix_layer
        # avg_layer = None
        # linear_layer = self.super_net.classifier
        

    def clean(self):
        self.converter_pool.close()
        self.estimator_pool.close()
        self.converter_pool.join()
        self.estimator_pool.join()
        self.tcp_client_socket.close()

def main():
    """Main function for the program.  Everything starts here.
    :return: None
    """
    images = np.load("inp_sample.npy").transpose((0,2,3,1))
    print("Loading input data shape :",images.shape)
    y_data = np.load('lbl_sample.npy')
    convert_start = time.time()
    os.system("mo.py --data_type=FP16 --input_model=test_mod.pb --input_shape=[1,160,160,3]")
    model_xml_fullpath='test_mod.xml'
    model_bin_fullpath='test_mod.bin'
    convert_end = time.time()

    # load a single plugin for the application
    load_start = time.time()
    ie = IECore()
    net = IENetwork(model=model_xml_fullpath, weights=model_bin_fullpath)
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    n, c, h, w = net.inputs[input_blob].shape

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    exec_net = ie.load_network(network=net, device_name = "MYRIAD")
    load_end = time.time()
    net_start = time.time()
    res = exec_net.infer(inputs={input_blob: images[0].reshape((1,3,160,160))})
    res = res[output_blob]
    net_end = time.time()
    print("convert time : {} \n \
            load time : {}\n \
            net inference time : {}\n ".format(convert_end-convert_start, load_end-load_start, net_end-net_start))
    # print("result : ", res)
    del net
    del exec_net

if __name__ == '__main__':
    server_ip = "159.226.41.50"
    server_port = 20026
    nas_client = instu_nas_latency_client(server_ip, server_port)
    # nas_client.conn_server()
    # nas_client.eval_subnet()
    # nas_client.eval_subnet_block_local(160)
    eval_start = time.time()
    net_list = []
    net_config = {'wid': None, 'ks': [3, 3, 5, 3, 5, 3, 5, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 3,3], 'e': [3, 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 3, 3, 3, 4, 3, 3, 4, 3, 3,4], 'd': [4, 2, 3, 4, 3], 'r': [160]}
    for i in range(1):
        net_list.append(net_config)
    latencys = nas_client.eval_subnet_list_local(net_list)
    print(latencys)
    eval_end = time.time()
    print("Total time : ", eval_end-eval_start)
    # nas_client.clean()
