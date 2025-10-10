import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml

from tqdm import tqdm

from time_series import time_series_models
from time_series import kernels
from time_series import evaluators
from time_series import data_generators

from .time_series_data import TimeSeriesData
from .kernel_container import KernelContainer 
from .model_container import ModelContainer
from .sub_experiment import SubExperiment

model_library = {name:model for name, model in time_series_models.__dict__.items() if "_" not in name}
kernel_library = {name:kernel for name, kernel in kernels.__dict__.items() if "_" not in name}
evaluator_library = {name:evaluator for name, evaluator in evaluators.__dict__.items() if "_" not in name}
data_generator_library = {name:generator for name, generator in data_generators.__dict__.items() if "_" not in name} 

def iterate_datasets(datasets_conf):
    for dataset, conf in datasets_conf.items():
        generator = conf["generator"]
        parameters = conf["parameters"] if "parameters" in conf else {}
        tsp = conf["train_test_split"]

        generator_class = data_generator_library[generator]

        if "sweeps" in conf:
            sweep_val_names = []
            sweep_values = []
            for param, sweep_vals in conf["sweeps"].items():
                sweep_val_names.append(param)
                if type(sweep_vals) == dict:
                    sweep_values.append(np.linspace(
                        float(sweep_vals["min"]), 
                        float(sweep_vals["max"]), 
                        int(sweep_vals["N_steps"])
                    ))
                else:
                    sweep_values.append(sweep_vals)

            # print(sweep_val_names, sweep_values)
            
            # Combine the sweep values
            all_combinations = itertools.product(*sweep_values)     
            for combined_vals in all_combinations:
                sweep_result = dict(parameters)           
                for i, param in enumerate(sweep_val_names):
                    sweep_result[param] = combined_vals[i]

                generator = generator_class(**sweep_result)

                _, data = generator()
                X, y = data[:-1], data[1:]

                yield TimeSeriesData(
                    X, 
                    y, 
                    train_val_test_split=tsp,
                    dataset_name=dataset,
                    parameters=sweep_result
                )
        else: 
            generator = generator_class(**parameters)

            _, data = generator()
            X, y = data[:-1], data[1:]

            yield TimeSeriesData(
                    X, 
                    y, 
                    train_val_test_split=tsp,
                    dataset_name=dataset,
                    parameters=parameters
                )

class Experiment:
    def __init__(self, filepath):
        self.filepath = filepath

        self.completed_experiments = []

        self.plots = []
        self.report_cols = []

        self.savefolder = ""

        self.sub_experiments = self.parse_config(filepath)

        if self.savefolder and not os.path.exists(self.savefolder):
            os.mkdir(self.savefolder)

        
    def load_dataset(self, filepath):
        data = pd.read_csv(filepath, index_col=0).values
        return data
    
    def parse_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.config = config
        
        self.experiment_config = config["experiment_config"]
        experiments = config["experiments"]
        for exp in config["experiments"].values():
            self.plots.append(exp["plots"])
            self.report_cols.append(exp["reports"])

        self.savefolder = self.experiment_config["savefolder"]

        return self.parse_experiments(experiments)

    def parse_experiments(self, experiments):
        for experiment, experiment_confs in experiments.items():
            models = experiment_confs["models"]
            datasets = experiment_confs["datasets"]
            metrics = experiment_confs["metrics"]
            # self.plots.append(experiment_confs["plots"])

            # Load evaluators
            hparam_evaluator = evaluator_library[metrics["hyperparameter_tuning"]]
            evaluators = {i:evaluator_library[i]() for i in metrics["evaluation"]}

            # Process kernels
            kernels = {
                k_name:KernelContainer(
                    kernel_name=k_name,
                    kernel_class=kernel_library[k_conf["kernel"]],
                    kernel_parameters=k_conf["parameters"]
                )
                for k_name, k_conf in experiment_confs["kernels"].items()
            }

            datasets = iterate_datasets(datasets)

            product = itertools.product(datasets, models)
            
            for dataset, model_name in product:
                # Load dataset
                # data = self.load_dataset(datasets[dataset_name]["filepath"])
                # X, y = data[:-1], data[1:]
                # tsp = datasets[dataset_name]["train_test_split"]


                # Load model
                model_class = model_library[models[model_name]["model"]]
                model_params = models[model_name]["parameters"] if "parameters" in models[model_name] else {}
                kernel_names = models[model_name]["kernels"] if "kernels" in models[model_name] else []
                model_hparams = models[model_name]["hyperparameters"] if "hyperparameters" in models[model_name] else None

                yield SubExperiment(
                    experiment_name=experiment,
                    dataset=dataset,
                    model_container= ModelContainer(
                        model_name=model_name,
                        model_class=model_class,
                        model_parameters=model_params,
                        model_kernels=[kernels[k] for k in kernel_names],
                        hyperparameters=model_hparams
                    ),
                    evaluators=evaluators,
                    hyperparameter_evaluator=hparam_evaluator,
                    kernels={k:v for k, v in kernels.items() if k in kernel_names}
                )
    
    def run_experiments(self):
        for sub_exp in tqdm(self.sub_experiments):
            sub_exp.run_experiment()
            sub_exp.drop_data()
            self.completed_experiments.append(sub_exp)

    def get_results(self):
        return pd.DataFrame(map(lambda x: x.get_results(), self.completed_experiments)).set_index("exp_name")
    
    def generate_plots(self):
        results = self.get_results()
        count = 0
        for r in [i for i in self.plots if i]:
            count += 1
            for report, report_conf in r.items():
                
                plt.figure()
                plottype = report_conf["type"] if "type" in report_conf else "line"

                if plottype in ["line", "scatter"]:
                    if plottype == "line":
                        plotter = plt.plot
                    elif plottype == "scatter":
                        plotter = plt.scatter

                    x_data = results[report_conf["x"]]

                    if type(report_conf["y"]) == list:
                        for yi in report_conf["y"]:
                            y_data = results[yi]
                            plotter(x_data, y_data)
                        plt.legend(report_conf["y"])
                    else:
                        y_data = results[report_conf["y"]]
                        plotter(x_data, y_data)
                        plt.ylabel(report_conf["y"])
                
                    plt.xlabel(report_conf["x"])
                
                elif plottype == "histogram":
                    x_data = results[report_conf["x"]]
                    plt.hist(x_data, bins=int(report_conf["bins"]))
                    plt.xlabel(report_conf["x"])
                
                savefile = report + ".png"

                plt.title(report)
                plt.savefig(os.path.join(self.savefolder, savefile))

    def generate_reports(self):
        df = self.get_results()
        for r in self.report_cols:
            print(df[r])


    def save(self):       
        self.get_results().to_csv(os.path.join(self.savefolder, "results.csv"))
        
        with open(os.path.join(self.savefolder, "experiment_config.yaml"), "w") as file:
            yaml.dump(self.config, file)
            file.close()