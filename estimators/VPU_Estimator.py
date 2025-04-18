from .base_Estimator import Estimator
import numpy as np
import time
import os
# VPU libs
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
    
class VPU_Estimator(Estimator):
    def __init__(self, model_dir='./model_pool/'):
        super(VPU_Estimator, self).__init__(model_dir)
        self.inp_sample_sz = {}
        self.ie = IECore()
        INFERENCE_DEV = "MYRIAD"
        # self.ie = iecore
        # inp_sz = (160, 176, 192, 208, 224)
        # for sz in inp_sz:
        #     self.inp_sample_sz[str(sz)] = np.random.rand(1, 3, sz, sz) # NCHW

    def latency_eval(self, subnet_name, get_output_shape=False):
        # self.setup()
        latency = 1e5
        lat_std = 0
        net = None
        try :
            net = None
            exec_net = None
            xml_path = self.model_pool_dir + subnet_name + '.xml'
            bin_path = self.model_pool_dir + subnet_name + '.bin'
            net = IENetwork(model=xml_path, weights=bin_path)
            run_times = 50
            input_blob = next(iter(net.inputs))

            n, c, h, w = net.inputs[input_blob].shape
            assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
            assert len(net.outputs) == 1, "Sample supports only single output topologies"
            load_start = time.time()
            exec_net = self.ie.load_network(network=net, device_name = "MYRIAD", num_requests=run_times)
            infer_start = time.time()
            res = None
            latencys = []

            # warm up
            # print("input shape ", n, c, h, w)
            # input_sample = np.random.rand(n, c, h, w)
            input_sample = np.ones(shape=(n,c,h,w))
            exec_net.requests[0].infer(inputs={input_blob: input_sample})
            if get_output_shape:
                # print(net.outputs)
                # print(iter(net.outputs))
                # print(next(iter(net.outputs)))
                # print( exec_net.requests[0].outputs)
                # print("output_blob data ", exec_net.requests[0].outputs[output_blob][:1])
                output_blob = next(iter(net.outputs))
                exec_net.requests[0].infer(inputs={input_blob: input_sample})
                output_data = exec_net.requests[0].outputs[output_blob]
                self.output_shape = output_data.shape
            for request in exec_net.requests:
                res = request.infer(inputs={input_blob: input_sample})
                # request.infer()
                latencys.append(request.latency)
            infer_end = time.time()
            # print(latencys)
            latencys = np.array(latencys)
            lat_mean = np.mean(latencys)
            lat_std =  np.std(latencys)
            all_end = time.time()
            # print("infer time : ", infer_start-load_start, infer_end-infer_start, all_end-infer_end, lat_mean)
            # print("infer latency {:.4f}ms(mean), {:.4f}ms(std)".format(lat_mean, lat_std))
            latency = lat_mean # ms
        except Exception as e:
            print("VPU estimator latency eval Error !", e)
        finally:
            if net is not None:
                del net
            if exec_net is not None:
                del exec_net
        if latency == 1e5:  # error occured in profiling, retest
            latency, lat_std = self.latency_eval(subnet_name)
        # self.clean()
        return latency, lat_std
        
    def clean(self):
        # super(VPU_Estimator, self).clean()
        del self.ie
