class KernelContainer:
    def __init__(
        self,
        kernel_name,
        kernel_class,
        kernel_parameters, 
        hyperparameters = None
    ):
        self.kernel_name = kernel_name
        self.kernel_class = kernel_class
        self.kernel_parameters = kernel_parameters
        self.hyperparameters = hyperparameters

    def update_parameters(self, **update_params):
        self.kernel_parameters.update(update_params)
    
    def build_kernel(self):
        return self.kernel_class(**self.kernel_parameters)
    