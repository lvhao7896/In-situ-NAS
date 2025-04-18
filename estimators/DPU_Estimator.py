from base_Estimator import Estimator
# DPU libs
from pynq_dpu import DpuOverlay, dputils
from dnndk import n2cube
class DPU_Estimator(Estimator):
    def __init__(self, model_dir='./model_pool/'):
        super(DPU_Estimator, self).__init__(model_dir)
        self.overlay = DpuOverlay("dpu.bit")
    
    def latency_eval(self, subnet_name):
        self.overlay.load_model("{}/{}.elf".format(self.model_pool_dir, subnet_name))
        KERNEL_CONV_INPUT = '316_1_Conv2D'
        KERNEL_FC_OUTPUT = 'output_1_MatMu'
        iter_count = 50
        n2cube.dpuOpen()
        kernel = n2cube.dpuLoadKernel(subnet_name)
        task = n2cube.dpuCreateTask(kernel, 0)
        # pred
        latencys = []
        for _ in range(iter_count):
            start = time.perf_counter()
            dputils.dpuSetInputImage2(task, KERNEL_CONV_INPUT, img)
            n2cube.dpuGetInputTensor(task, KERNEL_CONV_INPUT)
            n2cube.dpuRunTask(task)
            size = n2cube.dpuGetOutputTensorSize(task, KERNEL_FC_OUTPUT)
            channel = n2cube.dpuGetOutputTensorChannel(task, KERNEL_FC_OUTPUT)
            conf = n2cube.dpuGetOutputTensorAddress(task, KERNEL_FC_OUTPUT)
            outputScale = n2cube.dpuGetOutputTensorScale(task, KERNEL_FC_OUTPUT)
            softmax = n2cube.dpuRunSoftmax(conf, channel, size//channel, outputScale)
            inference_time = time.perf_counter() - start
            latencys.append(inference_time*1000)
        # clean up
        n2cube.dpuDestroyTask(task)
        n2cube.dpuDestroyKernel(kernel)
        return latencys
    
    def clean(self):
        pass