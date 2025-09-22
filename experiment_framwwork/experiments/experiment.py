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

from .time_series_data import TimeSeriesData
from .kernel_container import KernelContainer 
from .model_container import ModelContainer
from . sub_experiment import SubExperiment

model_library = {name:model for name, model in time_series_models.__dict__.items() if "_" not in name}
kernel_library = {name:kernel for name, kernel in kernels.__dict__.items() if "_" not in name}
evaluator_library = {name:evaluator for name, evaluator in evaluators.__dict__.items() if "_" not in name}

class Experiment:
    def __init__(self, filepath):
        self.filepath = filepath

        self.completed_experiments = []

        self.reports = []

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
        
        self.experiment_config = config["experiment_config"]
        experiments = config["experiments"]

        self.savefolder = self.experiment_config["savefolder"]

        return self.parse_experiments(experiments)

    def parse_experiments(self, experiments):
        for experiment, experiment_confs in experiments.items():
            models = experiment_confs["models"]
            datasets = experiment_confs["datasets"]
            metrics = experiment_confs["metrics"]
            self.reports.append(experiment_confs["reports"])

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

            product = itertools.product(datasets, models)
            
            for dataset_name, model_name in product:
                # Load dataset
                data = self.load_dataset(datasets[dataset_name]["filepath"])
                X, y = data[:-1], data[1:]
                tsp = datasets[dataset_name]["train_test_split"]


                # Load model
                model_class = model_library[models[model_name]["model"]]
                model_params = models[model_name]["parameters"]
                kernel_names = models[model_name]["kernels"]
                model_hparams = models[model_name]["hyperparameters"] if "hyperparameters" in models[model_name] else None

                yield SubExperiment(
                    experiment_name=experiment,
                    dataset = TimeSeriesData(
                        X, 
                        y, 
                        train_val_test_split=tsp,
                        dataset_name=dataset_name
                    ),
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
            # self.completed_experiments.append(sub_exp)

            sub_exp.run_experiment()

            self.completed_experiments.append(sub_exp)

    def get_results(self):
        return pd.DataFrame(map(lambda x: x.get_results(), self.completed_experiments)).set_index("exp_name")
    
    def generate_reports(self):
        results = self.get_results()
        i = 0
        for r in self.reports:
            i += 1
            for report, report_conf in r.items():
                if report == "plot":
                    if "type" not in report_conf:
                        plottype = "line"
                    else:
                        plottype = report_conf["type"]

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

                    if "savefile" not in report_conf:
                        savefile = os.path.join(self.savefolder, f"plot_{i}.png")
                    else:
                        savefile = os.path.join(self.savefolder, report_conf["savefile"])
                    
                    plt.savefig(savefile)

                    
    def save(self):       
        self.get_results().to_csv(os.path.join(self.savefolder, "results.csv"))
        
                