from .base_Estimator import Estimator
import numpy as np
import time
import os
import json
# TVM libs
from tvm import rpc
from tvm.contrib import graph_runtime as runtime
import tvm
ndk_cc_path = '/opt/android-toolchain-arm64/bin/aarch64-linux-android-g++'
os.environ.setdefault("TVM_NDK_CC", ndk_cc_path)
class ARM_Estimator(Estimator):
    def __init__(self, model_dir='./model_pool/', host='0.0.0.0', port='9190', key='android'):
        super(ARM_Estimator, self).__init__(model_dir)
        
        self.tracker_host = os.environ.get('TVM_TRACKER_HOST', host)
        self.tracker_port = int(os.environ.get('TVM_TRACKER_PORT', port))
        self.key = key

    def latency_eval(self, subnet_name, get_output_shape=False):
        print("connecting remote device ...")
        repeat_times = 50
        conn_start = time.perf_counter()
        tracker = rpc.connect_tracker(self.tracker_host, self.tracker_port)
        latency = (1e5, 0)
        try:
            self.remote = tracker.request(self.key, priority=0,
                                    session_timeout=7)
            print("conncection setup!")
            conn_end = time.perf_counter()
            module_load_start = time.perf_counter()
            self.ctx = self.remote.cpu(0)
            # upload the library to remote device and load it
            lib_path = f'{self.model_pool_dir}/{subnet_name}.so'
            graph_path = f'{self.model_pool_dir}/{subnet_name}.json'
            with open(graph_path) as graph_file:
                graph = json.load(graph_file)
            # input_name = 'input_1'
            self.remote.upload(lib_path)
            rlib = self.remote.load_module(f'{subnet_name}.so')
            # create the remote runtime module
            module = runtime.create(graph, rlib, self.ctx)
            # assert (module.get_num_inputs() == 1), "The subnet should have only one input."
            # module = runtime.GraphModule(rlib['default'](self.ctx))
            module_load_end = time.perf_counter()
            # set input data
            input_shape = module.get_input(0).shape
            # print("input shape ", input_shape)
            x = np.random.rand(*input_shape)

            module.set_input(0, tvm.nd.array(x.astype(np.float32)))
            module.run()
            if get_output_shape:
                self.output_shape = module.get_output(0).shape

            eval_start = time.perf_counter()
            ftimer = module.module.time_evaluator('run', self.ctx, number=1, repeat=repeat_times)
            prof_res = np.array(ftimer().results)  # convert to millisecond
            eval_end = time.perf_counter()
            latency = (np.mean(prof_res)* 1000, np.std(prof_res)* 1000)
            
            print(f"connection setup time {conn_end-conn_start}")
            print(f"module load time : {module_load_end-module_load_start}")
            print(f"eval done in {eval_end-eval_start}s, average {(eval_end-eval_start)/100}.")
            print(f"Total time : {eval_end-conn_start}")
        except Exception as e:
            print("Eval ARM Error ! ", e)
        print('Mean inference time (std dev): %.2f ms (%.2f ms)' % latency)
        return latency