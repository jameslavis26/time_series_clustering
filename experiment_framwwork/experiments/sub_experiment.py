import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

class SubExperiment:
    def __init__(
        self, 
        experiment_name,
        model_container,
        dataset,
        evaluators,
        kernels,
        hyperparameter_evaluator=None
    ):
        self.experiment_name = experiment_name
        self.model_container = model_container
        self.dataset = dataset
        self.evaluators = evaluators
        self.kernels = kernels
        self.hyperparameter_evaluator = hyperparameter_evaluator

        name = f"{self.dataset.dataset_name} - {self.model_container.model_name}"

        self.results = dict(exp_name=name)        

    def build_model(self):
        model_class = self.model_container.model
        model_parameters = self.model_container.parameters
        model_kernels = self.model_container.model_kernels

        # build_model
        model = model_class(
            kernels=[k.build_kernel() for k in model_kernels], 
            **model_parameters
        )

        return model
    
    def tune_parameters(self):
        model_mapping = {self.model_container.model_name:self.model_container}
        kernel_mapping = {k_name:kernel for k_name, kernel in self.kernels.items()}

        object_mapping = dict()
        object_mapping.update(model_mapping)
        object_mapping.update(kernel_mapping)

        model_hparams = self.model_container.hyperparameters if self.model_container.hyperparameters else []
        X_train, y_train = self.dataset.train_data()
        X_val, y_val = self.dataset.val_data()

        evaluator = self.hyperparameter_evaluator()

        def objective(trial):
            # Update model parameters
            if self.model_container.hyperparameters:
                model_parameters = {}
                for hparam, hparam_conf in model_hparams.items():
                    if hparam_conf["type"] == "float":
                        model_parameters[hparam] = trial.suggest_float(
                            self.model_container.model_name + " " + hparam, 
                            hparam_conf["min"], 
                            hparam_conf["max"]
                        )
                    elif hparam_conf["type"] == "int":
                        model_parameters[hparam] = trial.suggest_int(
                            self.model_container.model_name + " " + hparam, 
                            hparam_conf["min"], 
                            hparam_conf["max"]
                        )
                    else:
                        raise ValueError("Expecting float or int")

                self.model_container.update_parameters(model_parameters)

            # Update kernel parameters
            for k_name, kernel in self.kernels.items():
                if kernel.hyperparameters:
                    params = {}
                    for hparam, hparam_conf in kernel.hyperparameters.items():
                        if hparam_conf["type"] == "float":
                            params[hparam] = trial.suggest_float(
                                k_name + " " + hparam, 
                                hparam_conf["min"], 
                                hparam_conf["max"]
                            )
                        elif hparam_conf["type"] == "int":
                            params[hparam] = trial.suggest_int(
                                k_name + " " + hparam, 
                                hparam_conf["min"], 
                                hparam_conf["max"]
                            )
                        else:
                            raise ValueError("Expecting float or int")
                        
                    kernel.update_parameters(params)

            model = self.build_model()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            return evaluator(y_val, y_pred)

        study = optuna.create_study()
        study.optimize(
            objective, 
            n_trials=2, 
            timeout=60*20, # Optimise for n_trails or timeout seconds
        )

        best_params = study.best_params
        # Need to map best_params back to model and objects
        ## Also, different kernels with the same hparram name (ie bandwidth) will raise an error. Need to differentiate.

        for param, value in best_params.items():
            obj_name, param_name = param.split()
            object_mapping[obj_name].update_parameters({param_name:value})

            # if obj_name in kernel_mapping:
            #     self.results.update({param:value})
            # else:
            #     self.results.update({param_name:value})
                            
    def run_experiment(self):
        X_train, y_train = self.dataset.train_data()
        X_test, y_test = self.dataset.test_data()

        # Tune hyperparameters
        tune_hparams = False
        if self.model_container.hyperparameters:
            tune_hparams = True
        for kernel in self.model_container.model_kernels:
            if kernel.hyperparameters:
                tune_hparams = True
        
        if tune_hparams:
            self.tune_parameters()

        model = self.build_model()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        for evaluator_name, evaluator in self.evaluators.items():
            self.results[evaluator_name] = evaluator(y_test, y_pred)

    def get_results(self):
        results = dict()
        results.update(self.results)
        results.update(self.model_container.parameters)
        for k_name, kernel in self.kernels.items():
            params = kernel.kernel_parameters
            for i,j in params.items():
                results.update({k_name + "_" + i:j})
        return results

    def __repr__(self):
        rep = f"Experiment:\n {str(self.model_container)} \n {str(self.dataset)}"

        return rep