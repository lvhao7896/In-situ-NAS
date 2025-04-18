import os 

class Compiler:
    def __init__(self, output_dir='./model_pool/'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def prepare_compile(self, **params):
        raise NotImplementedError

    def compile(self, in_h, in_w, in_c, name='subnet'):
        raise NotImplementedError
    
    def finish_compile(self, **parmas):
        raise NotImplementedError