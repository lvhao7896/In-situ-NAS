from lib2to3.pgen2.literals import evalString
from platform import node
from re import sub
import time
import os
from sys import argv
import sys
import numpy as np
import time
import socket
import hashlib
import dill
import copy
from functools import partial
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
from  proxyless_nets_tf2 import ProxylessNASNets_tf2, convert_block_list
from ofa.model_zoo import ofa_net
import ast
import struct
import multiprocessing
from multiprocessing.managers import BaseManager
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor
# from viztracer import log_sparse
import psutil
import tensorflow as tf
assert tf.__version__.startswith('2'), 'Only supported for TF2'
tf.keras.backend.set_learning_phase(0)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
target_platform = 'VPU'
multiprocessing.set_start_method('fork')
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if target_platform.upper() == 'VPU':
    from estimators.VPU_Estimator import VPU_Estimator as Estimator
    from compiler import VPU_compiler as Compiler
# elif target_platform.upper() == 'TPU':
#     from estimators.TPU_Estimator import TPU_Estimator as Estimator
#     from compiler import TPU_compiler as Compiler
# elif target_platform.upper() == 'DPU':
#     from estimators.DPU_Estimator import DPU_Estimator as Estimator
#     from compiler import DPU_compiler as Compiler
# elif target_platform.upper() == 'ARM':
#     from estimators.ARM_Estimator import ARM_Estimator as Estimator
#     from compiler import ARM_compiler as Compiler
# elif target_platform.upper() == 'CUSTOM':
#     from estimators.Custom_Estimator import Custom_Estimator as Estimator
#     from compiler import custom_compiler as Compiler
else:
    raise NotImplemented

def delete_graph_from_node(graph, node_name):
    import copy
    out_node_edges = copy.deepcopy(graph.out_edges(node_name))
    for e in out_node_edges:
        out_node = e[1]
        delete_graph_from_node(graph, out_node)
    graph.remove_node(node_name)

def delete_graph_to_node(graph, node_name):
    # print("removing ", node_name)
    import copy
    in_node_edges = copy.deepcopy(graph.in_edges(node_name))
    for e in in_node_edges:
        in_node = e[0]
        delete_graph_to_node(graph, in_node)
    graph.remove_node(node_name)
    # print("remove done ", node_name)

def post_compile(ir, args):
    # compile the IR to the executable format for the target device
    from mo.main import emit_ir
    ret_res = emit_ir(ir, args)
    return ret_res

def resolve_name_conflict(graph1_node_name_list:list, graph2):
    import random
    import networkx as nx
    rename_dict = {}
    graph2_nodes_list = copy.deepcopy(list(graph2.nodes()))
    # rename the nodes/edges name/attr of the graph2 which has conflict the node from graph1
    for node_name in graph2_nodes_list:
        if node_name in graph1_node_name_list:
            # in_edges_list = graph2.in_edges(node_name)
            # out_edges_list = graph2.out_edges(node_name)
            new_node_name = str(random.randint(1,1e5))+'_'+node_name
            while new_node_name in graph1_node_name_list:
                new_node_name = str(random.randint(1,1e5))+'_'+node_name
            rename_dict[node_name] = new_node_name
            # for e in in_edges_list:
            #     e[1] = new_node_name
            # for e in out_edges_list:
            #     e[0] = new_node_name
            mapping = {node_name:new_node_name}
            nx.relabel_nodes(graph2, mapping, copy=False)
            node = graph2.node[new_node_name]
            # modify attrs
            node['name'] = new_node_name
            if 'fw_tensor_debug_info' in node and new_node_name.find('/Output_0') > 0:
                node['fw_tensor_debug_info'][0] = new_node_name[:new_node_name.find('/Output_0')]
        else:
            rename_dict[node_name] = node_name
    # print("sink source rename : ", rename_dict['Identity/sink_port_0'])
    return rename_dict

def compile_from_IRs(ir_list:list, verbose:bool=False):
    import networkx as nx
    import copy
    # assemble the block ir_list to the complete network topology.
    assert(len(ir_list)>1), 'The input ir_list for compile_from_IRs should have element more than 1.'
    complete_graph = copy.deepcopy(ir_list[0])
    sink_node_name = 'Identity/sink_port_0'
    source_node_name = 'x/Output_0/Data_'
    for idx, block_ir in enumerate(ir_list[1:]):
        ir = copy.deepcopy(block_ir)
        # remove sink node from the complte graph
        sink_node_from_edge_list = complete_graph.in_edges(sink_node_name)
        # print("sink node in edges : ", sink_node_from_edge_list)
        assert(len(sink_node_from_edge_list)==1), "Sink node should have only single input node." + str(sink_node_from_edge_list)
        # sink_node_from_node_name = sink_node_from_edge_list[0]
        for e in sink_node_from_edge_list:
            sink_node_from_node_name = e[0]
        # print("sink source node name : ", sink_node_from_node_name)
        assert(complete_graph.has_node(sink_node_name)), 'Detele node error, no node : '.format(sink_node_name)
        delete_graph_from_node(complete_graph, sink_node_name)

        complete_graph_nodes = complete_graph.nodes()
        rename_dict = resolve_name_conflict(complete_graph_nodes, ir)
        if verbose:
            print("concating ir idx : {} \n ".format(idx))
            print(rename_dict)
            print("nodes : ", list(nx.topological_sort(complete_graph)))
            print('-'*60)
            print("edges : ", complete_graph.edges())
        # remove the source node from the half compiled IR block
        source_node_out_edge_list = copy.deepcopy(ir.out_edges(rename_dict[source_node_name], data=True))
        assert(ir.has_node(rename_dict[source_node_name])), 'Detele node error, no node : '.format(source_node_name)
        delete_graph_to_node(ir, rename_dict[source_node_name])

        # graph_assemble
        complete_graph.add_nodes_from(ir.nodes(data=True))
        complete_graph.add_edges_from(ir.edges(data=True))
        # print("stage : ", complete_graph.stage)
        # complete_graph.strict_mode = False
        edge_list = []
        for e in source_node_out_edge_list:
            edge_list.append((sink_node_from_node_name, e[1], e[2]))
        # print("adding edges : ", edge_list)
        # print("#"*60)
        complete_graph.add_edges_from(edge_list)
        # complete_graph.add_edges(source_node_out_edge_list)
        # complete_graph.strict_mode = True
    return complete_graph

def compile_arch_from_lut(supernet, block_ir_dict:dict, arch_desc:dict, save_name:str='subnet'):
    # build the executable arch for target platform according the arch description
    import time
    from mo.utils.cli_parser import get_tf_cli_parser, append_exp_keys_to_namespace
    inp_sz = arch_desc['r'][0]
    inp_c = 3
    blocks_ir = []
    
    # supernet = ofa_net('ofa_proxyless_d234_e346_k357_w1.3', pretrained=True)   

    inp_c = 3
    supernet.set_active_subnet(ks=arch_desc['ks'], d=arch_desc['d'], e=arch_desc['e'])
    subnet = supernet.get_active_subnet().eval().cpu()
    
    start = time.perf_counter()
    # first conv and first dconv block
    subnet_blocks = [supernet.first_conv, supernet.blocks[0]]
    first_conv = supernet.first_conv
    layer_id = 0
    layer_type = 'first_conv'
    isz = [inp_sz, inp_sz, first_conv.in_channels]
    # print('input_sz ', inp_sz)
    oup_sz = (inp_sz + 1) // 2
    osz = [oup_sz, oup_sz, subnet_blocks[1].mobile_inverted_conv.out_channels]
    expand_channel = subnet_blocks[0].out_channels  # first block depthwise_conv out_channel
    stride=2
    id_skip = 0
    feature_list = [str(layer_id), layer_type, 'input:'+'x'.join([str(_) for _ in isz]), 
                    'output:'+'x'.join([str(_) for _ in osz]), 'expand:'+str(expand_channel),
                    'kernel:'+str(first_conv.kernel_size), 'stride:'+str(stride), 'idskip:'+str(id_skip) ]
    key = '-'.join(feature_list)
    blocks_ir.append(block_ir_dict[key])
    inp_sz = oup_sz
    layer_id += 1
    
    # mutable blocks
    block_num = len(subnet.blocks) - 1 # exclude the first block
    # print("block_num : ", block_num)
    stage_idx = 0
    stage_layers = copy.deepcopy(arch_desc['d'])
    stage_layers.append(1)
    stage_max_layers = [4,4,4,4,4,1]
    cur_stage_layer_cnt = 0
    for i in range(block_num):
        block = subnet.blocks[i+1]
        layer_type = 'expand_conv'
        inp_c = block.mobile_inverted_conv.in_channels
        isz = [inp_sz, inp_sz, inp_c]
        # print("input_sz ", inp_sz)
        stride = block.mobile_inverted_conv.stride
        oup_sz = int((inp_sz - 1)/stride + 1)
        osz = [oup_sz, oup_sz, block.mobile_inverted_conv.out_channels]
        id_skip = 1 if block.shortcut else 0
        kernel_sz = block.mobile_inverted_conv.kernel_size
        expand_channel = block.mobile_inverted_conv.depth_conv.conv.in_channels 
        feature_list = [str(layer_id), layer_type, 'input:'+'x'.join([str(_) for _ in isz]), 
                    'output:'+'x'.join([str(_) for _ in osz]), 'expand:'+str(expand_channel),
                    'kernel:'+str(kernel_sz), 'stride:'+str(stride), 'idskip:'+str(id_skip) ]
        key = '-'.join(feature_list)
        assert(key in block_ir_dict), "Feature key not found : {}".format(key)
        blocks_ir.append(block_ir_dict[key])
        inp_sz = oup_sz
        cur_stage_layer_cnt += 1
        layer_id+=1
        if stage_layers[stage_idx] == cur_stage_layer_cnt:
            layer_id += (stage_max_layers[stage_idx]-cur_stage_layer_cnt)
            cur_stage_layer_cnt = 0
            stage_idx+=1

    # final feature mix layer, pooling and fc.
    subnet_blocks = [subnet.feature_mix_layer, subnet.classifier]
    block = subnet.feature_mix_layer
    layer_type = 'conv_pooling_fc'
    inp_c = block.in_channels
    isz = [inp_sz, inp_sz, inp_c]
    # print("input_sz ", inp_sz)
    stride = block.stride
    oup_sz = int((inp_sz - 1)/stride + 1)
    osz = [oup_sz, oup_sz, block.out_channels]
    id_skip = 0
    expand_channel = block.out_channels
    feature_list = [str(layer_id), layer_type, 'input:'+'x'.join([str(_) for _ in isz]), 
                'output:'+'x'.join([str(_) for _ in osz]), 'expand:'+str(expand_channel),
                'kernel:'+str(block.kernel_size), 'stride:'+str(stride), 'idskip:'+str(id_skip) ]
    key = '-'.join(feature_list)
    blocks_ir.append(block_ir_dict[key])  
    complete_ir = compile_from_IRs(blocks_ir)
    # print("nodes : ", complete_ir.nodes(data=True))
    # print("edges : ", complete_ir.edges(data=True))
    # exit()
    argv = get_tf_cli_parser().parse_args()
    append_exp_keys_to_namespace(argv)
    argv.framework = 'tf'
    argv.data_type='FP16'
    argv.input_model=os.path.abspath('./model_pool/')+save_name+'.pb'
    resolution = arch_desc['r'][0]
    argv.input_shape= '[1,{},{},{}]'.format(*[resolution, resolution, 3])
    argv.output_dir=os.path.abspath('./model_pool/')
    argv.silent=True
    argv.model_name = save_name
    argv.log_level = 'ERROR'
    end = time.perf_counter()
    ret = post_compile(complete_ir, argv)
    post_compile_end = time.perf_counter()
    # print("LUT time : ", end-start)
    # print('post compile time : ', post_compile_end-end)
    output_dir = './model_pool/'
    return [output_dir + save_name + '.xml', output_dir + save_name + '.bin']




def prepare_eval_deamon(args_queue, deamon_idx, estimator_queue, output_dir, subnet_name='subnet'):
    super_nas_net = ofa_net('ofa_proxyless_d234_e346_k357_w1.3', pretrained=True)
    block_ir_dict = {}
    resolution_list = [160, 176, 192, 208, 224]
    for r in resolution_list:
        with open('block_ir_dict_{}_{}.pkl'.format(r, 'VPU'), 'rb') as f:
            block_ir_dict[r] = dill.load(f)

    while(True):
        try:
            start = time.perf_counter()
            idx, arch_desc = args_queue.get()
            # print("Compiling arch : ", idx, " ", arch_desc)
            name = subnet_name + str(idx)
            file_list = []

            file_list = compile_arch_from_lut(super_nas_net, block_ir_dict[arch_desc['r'][0]], arch_desc, name)
            
            estimator_queue.put_nowait((arch_desc, file_list, name, idx))
            end = time.perf_counter()
            # print(f"[Convert] subnet {name} start: {start} end: {end} total: in {convert_end-start}s.\n[Compile] in {end-convert_end}.")
        except Exception as e:
            print("prepare_eval_deamon Error! ", e)

# v2 added support for FPGA and network connection 
class instu_nas_latency_client_V2:
    def __init__(self, server_ip:str, port:int):
        self.server_ip = server_ip
        self.port = port
        num_processors_to_use = multiprocessing.cpu_count()
        # print("Avaiable parallal processors ", num_processors_to_use)
        self.num_estimator_processing = 1
        # self.num_compiler_processing = num_processors_to_use - self.num_estimator_processing - 1
        self.num_compiler_processing = 2
        self.compiler_pool = multiprocessing.Pool(int(self.num_compiler_processing))
        self.estimator_res = []
        self.compiler_res = []
        class Nas_Manager(BaseManager):
            pass
        Nas_Manager.register('queue', Queue)

        manager = Nas_Manager()
        manager.start()
        # self.super_nas_nets = [manager.ofa_net('ofa_proxyless_d234_e346_k357_w1.3', pretrained=True) for i in range(self.num_compiler_processing)]
        self.prepare_args_queue = manager.queue()
        self.compiler_estimator_queue = manager.queue()
        # self.compiler_queue = manager.queue()
        # for i in range(self.num_compiler_processing):
        #     self.compiler_queue.put_nowait(i)
        self.encoding = 'utf-8'
        self.max_msg_len = 1024
        self.cmd_format = '8sQ'
        self.file_info_format = '128sIQ32s'
        self.output_dir = './model_pool/'
        for i in range(self.num_compiler_processing):
            try:
                self.compiler_pool.apply_async(prepare_eval_deamon, args=(self.prepare_args_queue, i, self.compiler_estimator_queue, self.output_dir))
            except Exception as e:
                print("converter deamon start error ! ", e)
    def conn_server(self):
        try:
            tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_client_socket.connect((self.server_ip, self.port)) 
            return tcp_client_socket
        except Exception as e:
            print("Connect to estimator server error! ", e)

    def cal_md5(self, file_path):
        with open(file_path, 'rb') as fr:
            md5 = hashlib.md5()
            md5.update(fr.read())
            md5 = md5.hexdigest()
            return md5

    def get_file_info(self, file_path):
        file_name = os.path.basename(file_path)
        file_name_len = len(file_name)
        file_size = os.path.getsize(file_path)
        md5 = self.cal_md5(file_path)
        return file_name, file_name_len, file_size, md5

    def recv_sample(self, tcp_client_socket):
        try:
            file_info = tcp_client_socket.recv(struct.calcsize(self.file_info_format))
            file_name, file_name_len, file_size, md5 = struct.unpack(self.file_info_format, file_info)
            recv_sz = 0
            save_path = self.recv_dir + file_name
            with open(save_path, 'wb') as f:
                while recv_sz < file_size:
                    if file_size - recv_sz > self.max_msg_len:
                        recv_data = tcp_client_socket.recv(self.max_msg_len)
                        recv_sz += self.max_msg_len
                    else:
                        recv_data = tcp_client_socket.recv(file_size - recv_sz)
                        recv_sz += file_size - recv_sz
                    f.write(recv_data)
            recv_md5 = self.cal_md5(save_path)
            if md5 != recv_md5:
                raise Exception("MD5 check failed")
        except Exception as e:
            print("Recv Error ! ", e)

    def send_file(self, file_path, tcp_client_socket):
        assert(os.path.exists(file_path))
        start = time.perf_counter()
        buf = bytes()
        file_name, file_name_len, file_size, md5 = self.get_file_info(file_path)
        assert(len(file_name) < 128), 'Support maximux file name length 128, but {} provided'.format(len(file_name))
        file_head = struct.pack(self.file_info_format, file_name.encode(self.encoding), file_name_len, file_size, md5.encode(self.encoding))
        try: 
            tcp_client_socket.sendall(file_head)
            with open(file_path, 'rb',) as f:
                send_file = f.read()
                tcp_client_socket.sendall(send_file)
        except socket.error as e:
            print("Send sample Error : ", e)

    def get_eval_result(self):
        latency = 1e10
        arch_desc, send_file_list, subnet_name, idx = self.compiler_estimator_queue.get()
        # send file and request
        # print("sending subnet : {}, file list {}".format(subnet_name, send_file_list) )
        tcp_client_socket = self.conn_server()
        eval_cmd_format = '128sII'
        subnet_name_info  = struct.pack(eval_cmd_format, subnet_name.encode(self.encoding), len(subnet_name), len(send_file_list))
        tcp_client_socket.sendall(subnet_name_info)
        for file in send_file_list:
            self.send_file(file, tcp_client_socket)
        # recv file and request
        result_format = 'ff'
        result_size = struct.calcsize(result_format)
        recv_data = tcp_client_socket.recv(result_size)
        latency, std = struct.unpack(result_format, recv_data)
        return (arch_desc, latency, std)
    
    def eval_subnet_list(self, net_lists:list):
        class Temporary_resource_manager(BaseManager):
            pass
        compiled_num = 0
        try:
            start = time.time()
            # compile tasks distribute
            while compiled_num < len(net_lists):
                self.prepare_args_queue.put_nowait((compiled_num, net_lists[compiled_num]))
                compiled_num += 1

            # estimator tasks distribute
            with ThreadPoolExecutor() as executor:
                for _ in range(len(net_lists)):
                    self.estimator_res.append(executor.submit(self.get_eval_result))
            end = time.time()
            print('eval_subnet list time : ', end -start)
        except Exception as e:
            print("eval subnet list error ! ", e)
        print("estimating_num : ", compiled_num)

    def eval_subnet_list_local(self, net_list:list):
        pass

    def get_latencys(self):
        latencys = []
        # estimator results collect
        for future in self.estimator_res:
            latencys.append(future.result())
        print("lantency num : ", len(latencys))
        print(latencys)
        # clean up 
        self.compiler_res = []
        self.estimator_res = []
        # self.compiler_pool.join()
        return latencys



class instu_nas_accuracy_client:
    def __init__(self, server_ip:str, port:int):
        self.server_ip = server_ip
        self.port = port
        self.encoding = 'utf-8'
        self.max_msg_len = 1024
        self.cmd_format = '8sQ'

    def conn_server(self): 
        try :
            self.tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_client_socket.connect((self.server_ip, self.port))  
        except Exception as e:
            print("Connnect to GPU server error! ", e)

    def send_samples(self, samples):
        try:
            print("start sending")
            send_start = time.time()
            self.conn_server()
            print("connected")
            buf = bytes()
            for sample in samples:
                sample = str(sample) + ";"
                buf += sample.encode(encoding=self.encoding)
            cmd = struct.pack(self.cmd_format, 'gpu_eval'.encode('utf-8'), len(buf))
            self.tcp_client_socket.sendall(cmd)
            self.tcp_client_socket.sendall(buf)
            send_end = time.time()
            print("send {} samples total time : {:.3f}".format(len(samples), send_end-send_start))
        except Exception as e:
            print("acc client send samples error! ", e)
        # print("Send {} samples ".format(len(samples)))

    def recv_results(self):
        recv_start = time.time()
        accs = []
        cmd_size = struct.calcsize(self.cmd_format)
        cmd_data = self.tcp_client_socket.recv(cmd_size)
        cmd, msg_length = struct.unpack(self.cmd_format, cmd_data)
        assert(cmd == 'gpu_rets'.encode(self.encoding))
        buf = bytes()
        print("Recving acc result ....")
        while len(buf) < msg_length:
            if msg_length - len(buf) > 1024:
                recv_data = self.tcp_client_socket.recv(1024)
            else:
                recv_data = self.tcp_client_socket.recv(msg_length - len(buf))
            buf += recv_data
        buf = buf.decode(self.encoding)
        arch_desc_accs = buf.split(';')[:-1]     # remove empty arch_desc
        # print("Recv samples : ", len(arch_desc_accs))
        for arch_acc in arch_desc_accs:
            arch, acc = arch_acc.split('-')
            acc = float(acc)
            accs.append((arch, acc))
        self.clean()
        recv_end = time.time()
        print("Recv {} samples total time : {:.3f}".format(len(arch_desc_accs), recv_end-recv_start))
        return accs         # list(arch:str, acc:float)

    def clean(self):
        self.tcp_client_socket.close()