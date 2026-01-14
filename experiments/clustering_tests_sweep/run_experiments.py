import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from collections import defaultdict
from tqdm import tqdm

from sklearn.metrics import silhouette_score

from time_series.data_generators import LorenzGenerator
from time_series.models import KernelRidgeRegression, RascuttiModel
from time_series.evaluators import MeanSquaredError

from experiment_logging import Experiment
from time_series_clustering import TimeSeriesClustering
from dataset_creator import create_dataset
from time_series.data_handlers import TimeSeriesData

import optuna


# ============================================================
# CONFIG
# ============================================================

SEED = 0

# data
N_POINTS = 400
THETA_REF = np.pi / 4
NOISE_SWEEP = np.linspace(1e-3, 10, 20)

# sweeps
N_THETAS = 15
N_REPEAT = 20
MAX_CORRELATED_DIMS = 10
MAX_UNCORRELATED_DIMS = 10

# fixed settings
NOISE_FIXED = 1.0
N_CORRELATED_FIXED = 5

# optuna
N_TRIALS = 30

# model
MODEL_NAME = "KRR"
KERNEL = "rbf"


# ============================================================
# SETUP
# ============================================================

np.random.seed(SEED)

theta_values = np.linspace(0, np.pi / 2, N_THETAS)


# ============================================================
# EXPERIMENT 1: similarity vs theta (averaged, noise sweep)
# ============================================================

experiment = Experiment(
    "Similarity vs Theta - KRR - TransitionData (Averaged)",
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
                n_correlated_dimensions=3,
                n_uncorrelated_dimensions=0,
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

        # hyperparams
        def objective(trial):
            bandwidth = trial.suggest_float("bandwidth", 0.1, 4.0)
            reg = trial.suggest_float("reg", 1e-12, 1e-4)

            mse = 0.0
            for ds in datasets:
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

        # fit ref
        model_ref = KernelRidgeRegression(kernel=KERNEL, **best_params)
        X_ref, y_ref = ref_dataset.full_data()
        model_ref.fit(X_ref, y_ref)

        # similarities
        sims = np.zeros(len(theta_values))
        for i, ds in enumerate(datasets):
            X, y = ds.full_data()
            model = KernelRidgeRegression(kernel=KERNEL, **best_params)
            model.fit(X, y)
            sims[i] = model_ref.inner_product(model)

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
        )}
    )


# ============================================================
# EXPERIMENT 2: similarity vs n_correlated_dimensions
# ============================================================

experiment = Experiment(
    "Similarity vs CorrelatedDims - KRR - TransitionData",
    "experiments"
)

experiment.add_config(
    seed=SEED,
    n_points=N_POINTS,
    n_repeat=N_REPEAT,
    sweep="n_correlated_dimensions",
    model=MODEL_NAME,
    theta_ref=float(THETA_REF),
    noise=NOISE_FIXED,
)

for n_corr in range(1, MAX_CORRELATED_DIMS + 1):
    sims = []

    for r in range(N_REPEAT):
        data = create_dataset(
            THETA_REF,
            n_points=N_POINTS,
            n_correlated_dimensions=n_corr,
            n_uncorrelated_dimensions=0,
            noise=NOISE_FIXED,
        )

        ds = TimeSeriesData(
            X=data[:-1],
            y=data[1:],
            lag=1,
            train_val_test_split=[0.5, 0.3, 0.2],
        )

        def objective(trial):
            bandwidth = trial.suggest_float("bandwidth", 0.1, 4.0)
            reg = trial.suggest_float("reg", 1e-12, 1e-4)

            X_tr, y_tr = ds.train_data()
            X_va, y_va = ds.val_data()

            model = KernelRidgeRegression(
                kernel=KERNEL,
                bandwidth=bandwidth,
                reg=reg,
            )
            model.fit(X_tr, y_tr)
            return np.mean((model.predict(X_va) - y_va) ** 2)

        study = optuna.create_study()
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
        best_params = study.best_params

        model_ref = KernelRidgeRegression(kernel=KERNEL, **best_params)
        X, y = ds.full_data()
        model_ref.fit(X, y)

        model = KernelRidgeRegression(kernel=KERNEL, **best_params)
        model.fit(X, y)

        sims.append(model_ref.inner_product(model))

    sims = np.array(sims)

    experiment.add_result(
        **{f"corr_dims_{n_corr}": dict(
            n_correlated_dimensions=n_corr,
            theta_values=theta_values.tolist(),
            mean_similarities=sims.mean(axis=0).tolist(),
            std_similarities=sims.std(axis=0).tolist(),
            n_repeat=N_REPEAT,
            noise=NOISE_FIXED,
            theta_ref=float(THETA_REF),
        )}
    )


# ============================================================
# EXPERIMENT 3: similarity vs n_uncorrelated_dimensions
# ============================================================

experiment = Experiment(
    "Similarity vs UncorrelatedDims - KRR - TransitionData",
    "experiments"
)

experiment.add_config(
    seed=SEED,
    n_points=N_POINTS,
    n_repeat=N_REPEAT,
    sweep="n_uncorrelated_dimensions",
    model=MODEL_NAME,
    theta_ref=float(THETA_REF),
    noise=NOISE_FIXED,
    n_correlated_dimensions=N_CORRELATED_FIXED,
)

for n_uncorr in range(0, MAX_UNCORRELATED_DIMS + 1):
    sims = []

    for r in range(N_REPEAT):
        data = create_dataset(
            THETA_REF,
            n_points=N_POINTS,
            n_correlated_dimensions=N_CORRELATED_FIXED,
            n_uncorrelated_dimensions=n_uncorr,
            noise=NOISE_FIXED,
        )

        ds = TimeSeriesData(
            X=data[:-1],
            y=data[1:],
            lag=1,
            train_val_test_split=[0.5, 0.3, 0.2],
        )

        def objective(trial):
            bandwidth = trial.suggest_float("bandwidth", 0.1, 4.0)
            reg = trial.suggest_float("reg", 1e-12, 1e-4)

            X_tr, y_tr = ds.train_data()
            X_va, y_va = ds.val_data()

            model = KernelRidgeRegression(
                kernel=KERNEL,
                bandwidth=bandwidth,
                reg=reg,
            )
            model.fit(X_tr, y_tr)
            return np.mean((model.predict(X_va) - y_va) ** 2)

        study = optuna.create_study()
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
        best_params = study.best_params

        model_ref = KernelRidgeRegression(kernel=KERNEL, **best_params)
        X, y = ds.full_data()
        model_ref.fit(X, y)

        model = KernelRidgeRegression(kernel=KERNEL, **best_params)
        model.fit(X, y)

        sims.append(model_ref.inner_product(model))

    sims = np.array(sims)

    experiment.add_result(
        **{f"uncorr_dims_{n_uncorr}": dict(
            n_uncorrelated_dimensions=n_uncorr,
            n_correlated_dimensions=N_CORRELATED_FIXED,
            theta_values=theta_values.tolist(),
            mean_similarities=sims.mean(axis=0).tolist(),
            std_similarities=sims.std(axis=0).tolist(),
            n_repeat=N_REPEAT,
            noise=NOISE_FIXED,
            theta_ref=float(THETA_REF),
        )}
    )
