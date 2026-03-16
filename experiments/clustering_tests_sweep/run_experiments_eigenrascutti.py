import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from collections import defaultdict
from tqdm import tqdm

from sklearn.metrics import silhouette_score

from time_series.data_generators import LorenzGenerator
from time_series.models import KernelRidgeRegression, RascuttiModel, EigenRascuttiModel
from time_series.evaluators import MeanSquaredError

from experiment_logging import Experiment
from time_series_clustering import TimeSeriesClustering
from dataset_creator import create_dataset, dynamics_sincos, time_series_generator
from time_series.data_handlers import TimeSeriesData

import optuna

from pathlib import Path
import json
import os

def load_latest_results(experiment_root):
    experiment_root = Path(experiment_root)

    # list timestamped runs
    runs = [p for p in experiment_root.iterdir() if p.is_dir()]
    if not runs:
        raise ValueError(f"No runs found in {experiment_root}")

    latest_run = max(runs, key=lambda p: p.name)

    results_path = latest_run / "results" / "results.json"
    config_path = latest_run / "configs" / "config.json"

    with open(results_path) as f:
        results = json.load(f)

    with open(config_path) as f:
        config = json.load(f)

    return {
        "run_path": latest_run,
        "results": results,
        "config": config
    }



# ============================================================
# CONFIG
# ============================================================
SEED = 0

# default
N_POINTS = 1000
THETA_REF = np.pi/2
NOISE = 0.2
N_CORRELATED_DIMS = 3
N_UNCORRELATED_DIMS = 0

# Tuning
MIN_BANDWIDTH = 0.1
MAX_BANDWIDTH = 10

# sweeps
N_THETAS = 20
N_REPEAT = 100
MAX_CORRELATED_DIMS = 10
MAX_UNCORRELATED_DIMS = 10

# data
NOISE_SWEEP = np.linspace(1e-3, 2, 20)
N_DIM_SWEEP = np.arange(3, MAX_CORRELATED_DIMS)
N_CORR_DIM_SWEEP = np.arange(0, MAX_UNCORRELATED_DIMS)

# optuna
N_TRIALS = 30

# model
MODEL_NAME = "EigenRascutti"
KERNEL = "Gaus"

EXPERIMENTS_TO_RUN = [1,2,3,4]

# ============================================================
# SETUP
# ============================================================

np.random.seed(SEED)
theta_values = np.linspace(0, np.pi, N_THETAS)


# ============================================================
# EXPERIMENT 1: similarity vs theta (averaged, noise sweep)
# ============================================================
if 1 in EXPERIMENTS_TO_RUN:
    EXP_NAME = f"Similarity vs Noise - {MODEL_NAME} - TransitionData - Tune All"
    print("Running Experiment 1")
    experiment = Experiment(
        EXP_NAME,
        "experiments"
    )

    experiment.add_config(
        seed=SEED,
        n_points=N_POINTS,
        n_thetas=N_THETAS,
        n_repeat=N_REPEAT,
        sweep="theta",
        model=MODEL_NAME,
        theta_ref=float(THETA_REF),
    )

    for noise in tqdm(NOISE_SWEEP, desc="Noise sweep"):
        similarities_all = []

        for r in range(N_REPEAT):
            # reference
            ref_data = create_dataset(
                THETA_REF,
                n_points=N_POINTS,
                n_correlated_dimensions=3,
                n_uncorrelated_dimensions=0,
                noise=noise,
            )

            ref_dataset = TimeSeriesData(
                X=ref_data[:-1],
                y=ref_data[1:],
                lag=1,
                train_val_test_split=[0.5, 0.3, 0.2],
            )

            # theta datasets
            datasets = []
            for theta in theta_values:
                data = create_dataset(
                    theta,
                    n_points=N_POINTS,
                    n_correlated_dimensions=N_CORRELATED_DIMS,
                    n_uncorrelated_dimensions=N_UNCORRELATED_DIMS,
                    noise=noise,
                )

                datasets.append(
                    TimeSeriesData(
                        X=data[:-1],
                        y=data[1:],
                        lag=1,
                        train_val_test_split=[0.5, 0.3, 0.2],
                        theta=theta,
                    )
                )

            # similarities
            sims = np.zeros(len(theta_values))
            for i, ds in enumerate(datasets):
                # hyperparams
                def objective(trial):
                    bandwidth = trial.suggest_float("bandwidth", MIN_BANDWIDTH, MAX_BANDWIDTH)
                    lam = trial.suggest_float("reg", 1e-12, 1e-4)
                    rho = trial.suggest_float("reg", 1e-12, 1e-4)

                    mse = 0.0

                    X_tr, y_tr = ref_dataset.train_data()
                    X_va, y_va = ref_dataset.val_data()

                    model = EigenRascuttiModel(
                        bandwidth=bandwidth,
                        lam=lam,
                        rho=rho
                    )
                    model.fit(X_tr, y_tr)
                    mse += np.mean((model.predict(X_va) - y_va) ** 2)

                    X_tr, y_tr = ds.train_data()
                    X_va, y_va = ds.val_data()

                    model = EigenRascuttiModel(
                        bandwidth=bandwidth,
                        lam=lam,
                        rho=rho
                    )
                    model.fit(X_tr, y_tr)
                    mse += np.mean((model.predict(X_va) - y_va) ** 2)

                    return mse

                study = optuna.create_study()
                study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
                best_params = study.best_params

                X_ref, y_ref = ref_dataset.full_data()
                model_ref = KernelRidgeRegression(kernel=KERNEL, **best_params)
                model_ref.fit(X_ref, y_ref)
                ip11 = model_ref.inner_product(model_ref)

                X, y = ds.full_data()
                model = KernelRidgeRegression(kernel=KERNEL, **best_params)
                model.fit(X, y)

                ip12 = model_ref.inner_product(model)
                ip22 = model.inner_product(model)

                sims[i] = ip12/np.sqrt(ip11 * ip22)

            similarities_all.append(sims)

        sims = np.stack(similarities_all)

        experiment.add_result(
            **{f"theta_sweep_noise_{noise:.3f}": dict(
                noise=float(noise),
                theta_ref=float(THETA_REF),
                theta_values=theta_values.tolist(),
                mean_similarities=sims.mean(axis=0).tolist(),
                std_similarities=sims.std(axis=0).tolist(),
                n_repeat=N_REPEAT,
                bandwidth = best_params["bandwidth"],
                reg = best_params["reg"]
            )}
        )

    ###################
    ### Make figures
    ###################

    # Load latest
    noise_data_json = load_latest_results("experiments/" + EXP_NAME)

    ### Plot similarity for different noises
    noise_data_json["run_path"].joinpath("")

    if "figures" not in os.listdir(noise_data_json["run_path"]):
        os.mkdir(noise_data_json["run_path"].joinpath("figures"))

    for k, r in noise_data_json["results"].items():
        plt.figure()
        std = np.array(r["std_similarities"])
        ci = 1.96*std/np.sqrt(len(std))
        plt.errorbar(r["theta_values"], r["mean_similarities"], ci, marker="*")
        plt.vlines(r["theta_ref"], 0, max(r["mean_similarities"]), "r")
        plt.title(k)
        plt.xlabel("Theta")
        plt.ylabel("Similarity +- 95% ci")

        savepath = noise_data_json["run_path"].joinpath("figures").joinpath(k + ".png")

        plt.savefig(savepath)    

    ### Plot Similarity drop off
    plt.figure()
    x = []
    y = []
    std = []

    for k, r in noise_data_json["results"].items():
        idx = np.argmin(np.abs(np.array(r["theta_values"]) - r["theta_ref"]))
        x.append(r["noise"])
        y.append(r["mean_similarities"][idx])
        std.append(r["std_similarities"][idx])

    ci = [1.96*s/np.sqrt(len(std)) for s in std]

    plt.errorbar(x, y, ci, marker="*")
    plt.xlabel("noise")
    plt.ylabel("Similarity at $\\theta_{ref}$")

    savepath = noise_data_json["run_path"].joinpath("figures").joinpath("similarity_at_ref" + ".png")

    plt.savefig(savepath)  

# ============================================================
# EXPERIMENT 2: similarity vs n_dimensions
# ============================================================
if 2 in EXPERIMENTS_TO_RUN:
    print("Running Experiment 2")
    EXP_NAME = "Similarity vs CorrelatedDims - KRR - TransitionData - Tune All"
    experiment = Experiment(
        EXP_NAME,
        "experiments"
    )

    experiment.add_config(
        seed=SEED,
        n_points=N_POINTS,
        n_repeat=N_REPEAT,
        sweep="n_dimensions",
        model=MODEL_NAME,
        theta_ref=float(THETA_REF),
        noise=NOISE,
    )

    for n_dim in tqdm(N_DIM_SWEEP, desc="Dim sweep"):
        similarities_all = []

        for r in range(N_REPEAT):
            # reference
            ref_data = create_dataset(
                THETA_REF,
                n_points=N_POINTS,
                n_correlated_dimensions=n_dim,
                n_uncorrelated_dimensions=N_UNCORRELATED_DIMS,
                noise=NOISE,
            )

            ref_dataset = TimeSeriesData(
                X=ref_data[:-1],
                y=ref_data[1:],
                lag=1,
                train_val_test_split=[0.5, 0.3, 0.2],
            )

            # theta datasets
            datasets = []
            for theta in theta_values:
                data = create_dataset(
                    theta,
                    n_points=N_POINTS,
                    n_correlated_dimensions=n_dim,
                    n_uncorrelated_dimensions=N_UNCORRELATED_DIMS,
                    noise=NOISE,
                )

                datasets.append(
                    TimeSeriesData(
                        X=data[:-1],
                        y=data[1:],
                        lag=1,
                        train_val_test_split=[0.5, 0.3, 0.2],
                        theta=theta,
                    )
                )

            # similarities
            sims = np.zeros(len(theta_values))
            for i, ds in enumerate(datasets):
                # hyperparams
                def objective(trial):
                    bandwidth = trial.suggest_float("bandwidth", MIN_BANDWIDTH, MAX_BANDWIDTH)
                    reg = trial.suggest_float("reg", 1e-12, 1e-4)

                    mse = 0.0

                    X_tr, y_tr = ref_dataset.train_data()
                    X_va, y_va = ref_dataset.val_data()

                    model = KernelRidgeRegression(
                        kernel=KERNEL,
                        bandwidth=bandwidth,
                        reg=reg,
                    )
                    model.fit(X_tr, y_tr)
                    mse += np.mean((model.predict(X_va) - y_va) ** 2)

                    X_tr, y_tr = ds.train_data()
                    X_va, y_va = ds.val_data()

                    model = KernelRidgeRegression(
                        kernel=KERNEL,
                        bandwidth=bandwidth,
                        reg=reg,
                    )
                    model.fit(X_tr, y_tr)
                    mse += np.mean((model.predict(X_va) - y_va) ** 2)

                    return mse

                study = optuna.create_study()
                study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
                best_params = study.best_params

                X_ref, y_ref = ref_dataset.full_data()
                model_ref = KernelRidgeRegression(kernel=KERNEL, **best_params)
                model_ref.fit(X_ref, y_ref)
                ip11 = model_ref.inner_product(model_ref)

                X, y = ds.full_data()
                model = KernelRidgeRegression(kernel=KERNEL, **best_params)
                model.fit(X, y)

                ip12 = model_ref.inner_product(model)
                ip22 = model.inner_product(model)

                sims[i] = ip12/np.sqrt(ip11 * ip22)

            similarities_all.append(sims)

        sims = np.stack(similarities_all)

        experiment.add_result(
            **{f"theta_sweep_ndim_{n_dim:.3f}": dict(
                noise=float(NOISE),
                n_corr_dim = n_dim,
                n_uncorr_dim = N_UNCORRELATED_DIMS,
                theta_ref=float(THETA_REF),
                theta_values=theta_values.tolist(),
                mean_similarities=sims.mean(axis=0).tolist(),
                std_similarities=sims.std(axis=0).tolist(),
                n_repeat=N_REPEAT,
                bandwidth = best_params["bandwidth"],
                reg = best_params["reg"]
            )}
        )

    ###################
    ### Make figures
    ###################

    # Load latest
    corr_data_json = load_latest_results("experiments/"+ EXP_NAME)

    ### Plot similarity for different noises
    corr_data_json["run_path"].joinpath("")

    if "figures" not in os.listdir(corr_data_json["run_path"]):
        os.mkdir(corr_data_json["run_path"].joinpath("figures"))

    for k, r in corr_data_json["results"].items():
        plt.figure()
        std = np.array(r["std_similarities"])
        ci = 1.96*std/np.sqrt(len(std))
        plt.errorbar(r["theta_values"], r["mean_similarities"], ci, marker="*")
        plt.vlines(r["theta_ref"], 0, max(r["mean_similarities"]), "r")
        plt.title(k)
        plt.xlabel("Theta")
        plt.ylabel("Similarity +- 95% ci")

        savepath = corr_data_json["run_path"].joinpath("figures").joinpath(k + ".png")

        plt.savefig(savepath)    

    ### Plot Similarity drop off
    plt.figure()
    x = []
    y = []
    std = []

    for k, r in corr_data_json["results"].items():
        idx = np.argmin(np.abs(np.array(r["theta_values"]) - r["theta_ref"]))
        x.append(r["n_corr_dim"])
        y.append(r["mean_similarities"][idx])
        std.append(r["std_similarities"][idx])

    ci = [1.96*s/np.sqrt(len(std)) for s in std]

    plt.errorbar(x, y, ci, marker="*")
    plt.xlabel("Number of dimensions")
    plt.ylabel("Similarity at $\\theta_{ref}$")

    savepath = corr_data_json["run_path"].joinpath("figures").joinpath("similarity_at_ref" + ".png")

    plt.savefig(savepath)  

# ============================================================
# EXPERIMENT 3: similarity vs n_uncorrelated_dimensions
# ============================================================
if 3 in EXPERIMENTS_TO_RUN:
    print("Running Experiment 3")
    EXP_NAME = "Similarity vs UncorrelatedDims - KRR - TransitionData - Tune All"
    experiment = Experiment(
        EXP_NAME,
        "experiments"
    )

    experiment.add_config(
        seed=SEED,
        n_points=N_POINTS,
        n_repeat=N_REPEAT,
        sweep="n_uncorrelated_dimensions",
        model=MODEL_NAME,
        theta_ref=float(THETA_REF),
        noise=NOISE,
        n_correlated_dimensions=N_CORRELATED_DIMS,
    )

    for n_udim in tqdm(N_CORR_DIM_SWEEP, desc="Uncorrelated Dim sweep"):
        similarities_all = []

        for r in range(N_REPEAT):
            # reference
            ref_data = create_dataset(
                THETA_REF,
                n_points=N_POINTS,
                n_correlated_dimensions=N_CORRELATED_DIMS,
                n_uncorrelated_dimensions=n_udim,
                noise=NOISE,
            )

            ref_dataset = TimeSeriesData(
                X=ref_data[:-1],
                y=ref_data[1:],
                lag=1,
                train_val_test_split=[0.5, 0.3, 0.2],
            )

            # theta datasets
            datasets = []
            for theta in theta_values:
                data = create_dataset(
                    theta,
                    n_points=N_POINTS,
                    n_correlated_dimensions=N_CORRELATED_DIMS,
                    n_uncorrelated_dimensions=n_udim,
                    noise=NOISE,
                )

                datasets.append(
                    TimeSeriesData(
                        X=data[:-1],
                        y=data[1:],
                        lag=1,
                        train_val_test_split=[0.5, 0.3, 0.2],
                        theta=theta,
                    )
                )

            # similarities
            sims = np.zeros(len(theta_values))
            for i, ds in enumerate(datasets):
                # hyperparams
                def objective(trial):
                    bandwidth = trial.suggest_float("bandwidth", MIN_BANDWIDTH, MAX_BANDWIDTH)
                    reg = trial.suggest_float("reg", 1e-12, 1e-4)

                    mse = 0.0

                    X_tr, y_tr = ref_dataset.train_data()
                    X_va, y_va = ref_dataset.val_data()

                    model = KernelRidgeRegression(
                        kernel=KERNEL,
                        bandwidth=bandwidth,
                        reg=reg,
                    )
                    model.fit(X_tr, y_tr)
                    mse += np.mean((model.predict(X_va) - y_va) ** 2)

                    X_tr, y_tr = ds.train_data()
                    X_va, y_va = ds.val_data()

                    model = KernelRidgeRegression(
                        kernel=KERNEL,
                        bandwidth=bandwidth,
                        reg=reg,
                    )
                    model.fit(X_tr, y_tr)
                    mse += np.mean((model.predict(X_va) - y_va) ** 2)

                    return mse

                study = optuna.create_study()
                study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
                best_params = study.best_params

                X_ref, y_ref = ref_dataset.full_data()
                model_ref = KernelRidgeRegression(kernel=KERNEL, **best_params)
                model_ref.fit(X_ref, y_ref)
                ip11 = model_ref.inner_product(model_ref)

                X, y = ds.full_data()
                model = KernelRidgeRegression(kernel=KERNEL, **best_params)
                model.fit(X, y)

                ip12 = model_ref.inner_product(model)
                ip22 = model.inner_product(model)

                sims[i] = ip12/np.sqrt(ip11 * ip22)

            similarities_all.append(sims)

        sims = np.stack(similarities_all)

        experiment.add_result(
            **{f"theta_sweep_ndim_{n_udim:.3f}": dict(
                noise=float(NOISE),
                n_corr_dim = N_CORRELATED_DIMS,
                n_uncorr_dim = n_udim,
                theta_ref=float(THETA_REF),
                theta_values=theta_values.tolist(),
                mean_similarities=sims.mean(axis=0).tolist(),
                std_similarities=sims.std(axis=0).tolist(),
                n_repeat=N_REPEAT,
                bandwidth = best_params["bandwidth"],
                reg = best_params["reg"]
            )}
        )

    ###################
    ### Make figures
    ###################

    # Load latest
    uncorr_data_json = load_latest_results("experiments/" + EXP_NAME)

    ### Plot similarity for different noises
    uncorr_data_json["run_path"].joinpath("")

    if "figures" not in os.listdir(uncorr_data_json["run_path"]):
        os.mkdir(uncorr_data_json["run_path"].joinpath("figures"))

    for k, r in uncorr_data_json["results"].items():
        plt.figure()
        std = np.array(r["std_similarities"])
        ci = 1.96*std/np.sqrt(len(std))
        plt.errorbar(r["theta_values"], r["mean_similarities"], ci, marker="*")
        plt.vlines(r["theta_ref"], 0, max(r["mean_similarities"]), "r")
        plt.title(k)
        plt.xlabel("Theta")
        plt.ylabel("Similarity +- 95% ci")

        savepath = uncorr_data_json["run_path"].joinpath("figures").joinpath(k + ".png")

        plt.savefig(savepath)    

    ### Plot Similarity drop off
    plt.figure()
    x = []
    y = []
    std = []

    for k, r in uncorr_data_json["results"].items():
        idx = np.argmin(np.abs(np.array(r["theta_values"]) - r["theta_ref"]))
        x.append(r["n_uncorr_dim"])
        y.append(r["mean_similarities"][idx])
        std.append(r["std_similarities"][idx])

    ci = [1.96*s/np.sqrt(len(std)) for s in std]

    plt.errorbar(x, y, ci, marker="*")
    plt.xlabel("Number of uncorrelated dimensions")
    plt.ylabel("Similarity at $\\theta_{ref}$")

    savepath = uncorr_data_json["run_path"].joinpath("figures").joinpath("similarity_at_ref" + ".png")

    plt.savefig(savepath)  

# ============================================================
# EXPERIMENT 4: mse vs n_dimensions
# ============================================================
if 4 in EXPERIMENTS_TO_RUN:
    print("Running Experiment 4")
    EXP_NAME = "MSE vs CorrelatedDims - KRR - Tune All"
    experiment = Experiment(
        EXP_NAME,
        "experiments"
    )

    experiment.add_config(
        seed=SEED,
        n_points=N_POINTS,
        n_repeat=N_REPEAT,
        sweep="n_dimensions",
        model=MODEL_NAME,
        theta_ref=float(THETA_REF),
        noise=NOISE,
    )

    for n_dim in tqdm(N_DIM_SWEEP, desc="Dim sweep"):
        train_mse_all = []
        test_mse_all = []

        train_error_all = []
        test_error_all = []

        for r in range(N_REPEAT):
            # reference
            dynamics_func = dynamics_sincos(
                THETA_REF,
                n_correlated_dimensions=n_dim,
                n_uncorrelated_dimensions=N_UNCORRELATED_DIMS,
            )

            ref_data = time_series_generator(
                dynamics_func,
                x0 = np.random.uniform(-1.0, 1.0, size=n_dim + N_UNCORRELATED_DIMS),
                n_points=N_POINTS,
                noise=NOISE
            )

            ref_dataset = TimeSeriesData(
                X=ref_data[:-1],
                y=ref_data[1:],
                lag=1,
                train_val_test_split=[0.5, 0.3, 0.2],
            )

            X_tr, y_tr = ref_dataset.train_data()
            X_va, y_va = ref_dataset.val_data()

            def objective(trial):
                bandwidth = trial.suggest_float("bandwidth", MIN_BANDWIDTH, MAX_BANDWIDTH)
                reg = trial.suggest_float("reg", 1e-12, 1e-4)

                model = KernelRidgeRegression(
                    kernel=KERNEL,
                    bandwidth=bandwidth,
                    reg=reg,
                )
                model.fit(X_tr, y_tr)
                mse = np.mean((model.predict(X_va) - y_va) ** 2)

                return mse
            
            X_train = np.concatenate([X_tr, X_va])
            y_train = np.concatenate([y_tr, y_va])

            X_test, y_test = ref_dataset.test_data()

            study = optuna.create_study()
            study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
            best_params = study.best_params

            model_ref = KernelRidgeRegression(kernel=KERNEL, **best_params)
            model_ref.fit(X_train, y_train)

            training_mse = np.mean((model_ref.predict(X_train) - y_train) ** 2)
            test_mse = np.mean((model_ref.predict(X_test) - y_test) ** 2)

            train_mse_all.append(training_mse)
            test_mse_all.append(test_mse)

            # print(X_train.shape, model_ref.predict(X_train).shape, np.apply_along_axis(dynamics_func, 1, X_train.squeeze()).shape)

            training_error = np.mean((model_ref.predict(X_train) - np.apply_along_axis(dynamics_func, 1, X_train.squeeze())) ** 2)
            test_error = np.mean((model_ref.predict(X_test) - np.apply_along_axis(dynamics_func, 1, X_test.squeeze())) ** 2)

            train_error_all.append(training_error)
            test_error_all.append(test_error)

        experiment.add_result(
            **{f"theta_sweep_ndim_{n_dim:.3f}": dict(
                noise=float(NOISE),
                n_corr_dim = n_dim,
                n_uncorr_dim = N_UNCORRELATED_DIMS,
                theta_ref=float(THETA_REF),
                train_mse = float(np.mean(train_mse_all)),
                train_mse_std = float(np.std(train_mse_all)),
                test_mse = float(np.mean(test_mse_all)),
                test_mse_std = float(np.std(test_mse_all)),
                train_error = float(np.mean(train_error_all)),
                train_error_std = float(np.std(train_error_all)),
                test_error = float(np.mean(test_error_all)),
                test_error_std = float(np.std(test_error_all)),
                n_repeat=N_REPEAT,
                bandwidth = best_params["bandwidth"],
                reg = best_params["reg"]
            )}
        )

    ###################
    ### Make figures
    ###################

    # Load latest
    data_json = load_latest_results("experiments/"+ EXP_NAME)

    ### Plot similarity for different noises
    data_json["run_path"].joinpath("")

    if "figures" not in os.listdir(data_json["run_path"]):
        os.mkdir(data_json["run_path"].joinpath("figures"))   

    ### Plot MSE
    dims = []
    train_error = []
    train_error_std = []
    test_error = []
    test_error_std = []


    for k, r in data_json["results"].items():
        dims.append(r["n_corr_dim"])
        train_error.append(r["train_mse"])
        train_error_std.append(r["train_mse_std"])
        test_error.append(r["test_mse"])
        test_error_std.append(r["test_mse_std"])

    ci_train = [1.96*s/np.sqrt(N_REPEAT) for s in train_error_std]
    ci_test = [1.96*s/np.sqrt(N_REPEAT) for s in test_error_std]

    plt.figure()
    plt.errorbar(dims, train_error, ci_train, marker="*")
    plt.errorbar(dims, test_error, ci_test, marker="*")
    plt.xlabel("Number of dimensions")
    plt.ylabel("MSE at $\\theta_{ref}$")
    plt.legend(["Train MSE", "Test MSE"])
    plt.title("MSE vs dim")

    savepath = data_json["run_path"].joinpath("figures").joinpath("mse_at_ref" + ".png")

    plt.savefig(savepath)  

    ### Plot Error
    dims = []
    train_error = []
    train_error_std = []
    test_error = []
    test_error_std = []


    for k, r in data_json["results"].items():
        dims.append(r["n_corr_dim"])
        train_error.append(r["train_error"])
        train_error_std.append(r["train_error_std"])
        test_error.append(r["test_error"])
        test_error_std.append(r["test_error_std"])

    ci_train = [1.96*s/np.sqrt(N_REPEAT) for s in train_error_std]
    ci_test = [1.96*s/np.sqrt(N_REPEAT) for s in test_error_std]

    plt.figure()
    plt.errorbar(dims, train_error, ci_train, marker="*")
    plt.errorbar(dims, test_error, ci_test, marker="*")
    plt.xlabel("Number of dimensions")
    plt.ylabel("MSE at $\\theta_{ref}$")
    plt.legend(["Train Error", "Test Error"])
    plt.title("Error vs dimension")

    savepath = data_json["run_path"].joinpath("figures").joinpath("error_at_ref" + ".png")

    plt.savefig(savepath)  


# # ============================================================
# # EXPERIMENT 2: similarity vs n_correlated_dimensions         
# # ============================================================

# experiment = Experiment(
#     "Similarity vs CorrelatedDims - KRR - TransitionData",
#     "experiments"
# )

# experiment.add_config(
#     seed=SEED,
#     n_points=N_POINTS,
#     n_repeat=N_REPEAT,
#     sweep="n_correlated_dimensions",
#     model=MODEL_NAME,
#     theta_ref=float(THETA_REF),
#     noise=NOISE_FIXED,
# )

# for n_corr in range(1, MAX_CORRELATED_DIMS + 1):
#     sims = []

#     for r in range(N_REPEAT):
#         data = create_dataset(
#             THETA_REF,
#             n_points=N_POINTS,
#             n_correlated_dimensions=n_corr,
#             n_uncorrelated_dimensions=0,
#             noise=NOISE_FIXED,
#         )

#         ds = TimeSeriesData(
#             X=data[:-1],
#             y=data[1:],
#             lag=1,
#             train_val_test_split=[0.5, 0.3, 0.2],
#         )

#         def objective(trial):
#             bandwidth = trial.suggest_float("bandwidth", 0.1, 4.0)
#             reg = trial.suggest_float("reg", 1e-12, 1e-4)

#             X_tr, y_tr = ds.train_data()
#             X_va, y_va = ds.val_data()

#             model = KernelRidgeRegression(
#                 kernel=KERNEL,
#                 bandwidth=bandwidth,
#                 reg=reg,
#             )
#             model.fit(X_tr, y_tr)
#             return np.mean((model.predict(X_va) - y_va) ** 2)

#         study = optuna.create_study()
#         study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
#         best_params = study.best_params

#         model_ref = KernelRidgeRegression(kernel=KERNEL, **best_params)
#         X, y = ds.full_data()
#         model_ref.fit(X, y)

#         model = KernelRidgeRegression(kernel=KERNEL, **best_params)
#         model.fit(X, y)

#         sims.append(model_ref.inner_product(model))

#     sims = np.array(sims)

#     experiment.add_result(
#         **{f"corr_dims_{n_corr}": dict(
#             n_correlated_dimensions=n_corr,
#             theta_values=theta_values.tolist(),
#             mean_similarities=sims.mean(axis=0).tolist(),
#             std_similarities=sims.std(axis=0).tolist(),
#             n_repeat=N_REPEAT,
#             noise=NOISE_FIXED,
#             theta_ref=float(THETA_REF),
#         )}
#     )


# # ============================================================
# # EXPERIMENT 3: similarity vs n_uncorrelated_dimensions
# # ============================================================

# experiment = Experiment(
#     "Similarity vs UncorrelatedDims - KRR - TransitionData",
#     "experiments"
# )

# experiment.add_config(
#     seed=SEED,
#     n_points=N_POINTS,
#     n_repeat=N_REPEAT,
#     sweep="n_uncorrelated_dimensions",
#     model=MODEL_NAME,
#     theta_ref=float(THETA_REF),
#     noise=NOISE_FIXED,
#     n_correlated_dimensions=N_CORRELATED_FIXED,
# )

# for n_uncorr in range(0, MAX_UNCORRELATED_DIMS + 1):
#     sims = []

#     for r in range(N_REPEAT):
#         data = create_dataset(
#             THETA_REF,
#             n_points=N_POINTS,
#             n_correlated_dimensions=N_CORRELATED_FIXED,
#             n_uncorrelated_dimensions=n_uncorr,
#             noise=NOISE_FIXED,
#         )

#         ds = TimeSeriesData(
#             X=data[:-1],
#             y=data[1:],
#             lag=1,
#             train_val_test_split=[0.5, 0.3, 0.2],
#         )

#         def objective(trial):
#             bandwidth = trial.suggest_float("bandwidth", 0.1, 4.0)
#             reg = trial.suggest_float("reg", 1e-12, 1e-4)

#             X_tr, y_tr = ds.train_data()
#             X_va, y_va = ds.val_data()

#             model = KernelRidgeRegression(
#                 kernel=KERNEL,
#                 bandwidth=bandwidth,
#                 reg=reg,
#             )
#             model.fit(X_tr, y_tr)
#             return np.mean((model.predict(X_va) - y_va) ** 2)

#         study = optuna.create_study()
#         study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
#         best_params = study.best_params

#         model_ref = KernelRidgeRegression(kernel=KERNEL, **best_params)
#         X, y = ds.full_data()
#         model_ref.fit(X, y)

#         model = KernelRidgeRegression(kernel=KERNEL, **best_params)
#         model.fit(X, y)

#         sims.append(model_ref.inner_product(model))

#     sims = np.array(sims)

#     experiment.add_result(
#         **{f"uncorr_dims_{n_uncorr}": dict(
#             n_uncorrelated_dimensions=n_uncorr,
#             n_correlated_dimensions=N_CORRELATED_FIXED,
#             theta_values=theta_values.tolist(),
#             mean_similarities=sims.mean(axis=0).tolist(),
#             std_similarities=sims.std(axis=0).tolist(),
#             n_repeat=N_REPEAT,
#             noise=NOISE_FIXED,
#             theta_ref=float(THETA_REF),
#         )}
#     )
