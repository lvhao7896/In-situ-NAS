
class Estimator:
    def __init__(self, model_dir):
        self.model_pool_dir = model_dir
        self.output_shape = None

    def get_output_shape(self):
        return self.output_shape

    def latency_eval(self):
        raise NotImplementedError