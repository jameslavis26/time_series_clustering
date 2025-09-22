class ModelContainer:
    def __init__(
        self,
        model_name,
        model_class,
        model_parameters,
        model_kernels,
        hyperparameters = None
    ):
        self.model_name = model_name
        self.model = model_class
        self.parameters = model_parameters
        self.model_kernels = model_kernels
        self.hyperparameters = hyperparameters

    def update_parameters(self, dct):
        self.parameters.update(dct)

    def __repr__(self):
        rep = f"{str(self.model)}: {str(self.parameters)}"

        return rep