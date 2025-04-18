from .base_Estimator import Estimator
# TPU libs
import platform
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import os

class TPU_Estimator(Estimator):
    def __init__(self, model_dir='./model_pool/'):
        super(TPU_Estimator, self).__init__(model_dir)

        self.EDGETPU_SHARED_LIB = {
        'Linux': 'libedgetpu.so.1',
        'Darwin': 'libedgetpu.1.dylib',
        'Windows': 'edgetpu.dll'
        }[platform.system()]
        name = '{}_edgetpu.tflite'
 
    def make_interpreter(self, model_file):
        model_file, *device = model_file.split('@')
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(self.EDGETPU_SHARED_LIB,
                                    {'device': device[0]} if device else {})
            ])
 
    def latency_eval(self, subnet_name, get_output_shape=False):
        iter_count = 100
        latencys = []
        # setup_start = time.perf_counter()
        tflite_name = '{}/{}_edgetpu.tflite'.format(self.model_pool_dir, subnet_name)
        interpreter = self.make_interpreter(tflite_name)
        interpreter.allocate_tensors()
        if interpreter.get_input_details()[0]['dtype'] != np.uint8:
            raise ValueError('Only support uint8 input type.')
        params = interpreter.get_input_details()[0]['quantization_parameters']
        scale = params['scales']
        zero_point = params['zero_points']

        """Returns input image size as (width, height) tuple."""
        out_channel, height, width, in_channel = interpreter.get_input_details()[0]['shape']
        input_sz = (height, width)
        # print("input size : {}\n".format(input_sz))
        input_sample = np.ones((out_channel, height, width, in_channel))
        # start_idx = 9
        # input_sample = np.transpose(np.load('inp_sample_224.npz')['arr_0'].reshape((1000,3,224,224))[start_idx:start_idx+1], (0,2,3,1))
        # print(input_sample)
        # lbl_sample = np.load('lbl_sample_{}.npz'.format(224))['arr_0'].reshape((1000,))[start_idx:start_idx+1]
        # print("lbl_sample : " , lbl_sample)

        # input data range [0,255]
        # normalized_input = (np.asarray(input_sample)) / (scale) + zero_point
        # np.clip(normalized_input, 0, 255, out=normalized_input)

        # setup_end = time.perf_counter()
        # print("setup time : ", setup_end-setup_start)
        # print('----INFERENCE TIME----')
        # print('Note: The first inference on Edge TPU is slow because it includes',
        #     'loading the model into Edge TPU memory.')
        # warm up
        tensor_index = interpreter.get_input_details()[0]['index']
        interpreter.tensor(tensor_index)()[0][:, :] = input_sample.astype(np.uint8)
        interpreter.invoke()
        if get_output_shape:
            output_details = interpreter.get_output_details()[0]
            output = np.array(interpreter.tensor(output_details['index'])())
            self.output_shape = np.squeeze(output).shape
            print("output : ", output.reshape(-1)[:10])
        for _ in range(iter_count):
            start = time.perf_counter()
            # set input sample 
            """Returns input tensor view as numpy array of shape (height, width, 3)."""
            # tensor_index = interpreter.get_input_details()[0]['index']
            # interpreter.tensor(tensor_index)()[0][:, :] = input_sample
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            latencys.append(inference_time*1000)
            # print('%.1fms' % (inference_time * 1000))
 
        # print(latencys)
        latencys = np.array(latencys)
        lat_mean = np.mean(latencys)
        lat_std =  np.std(latencys)
        # print("infer latency {:.4f}ms(mean), {:.4f}ms(std)".format(lat_mean, lat_std))
        latency = lat_mean # ms
        os.system('rm ' + tflite_name)
        return latency, lat_std
 
    def clean(self):
        pass 