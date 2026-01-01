import numpy as np
import scipy as sp
import optuna

from time_series.data_handlers import TimeSeriesData
from time_series.models import KernelRidgeRegression
from time_series.evaluators import MeanSquaredError

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TimeSeriesClustering:
    def __init__(
        self,
        model_cls=KernelRidgeRegression,
        kernel="rbf",
        lag=1,
        split=(0.5, 0.2, 0.3),
        n_trials=50,
        n_jobs=-1,
        random_state=None,
    ):
        self.model_cls = model_cls
        self.kernel = kernel
        self.lag = lag
        self.split = split
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.datasets = None
        self.similarity_matrix = None

        self.error_matrix = None

    # -------------------------
    # Public API
    # -------------------------

    def fit(self, datasets: list[np.ndarray]):
        """
        Computes the pairwise similarity matrix between time series.
        """
        self.datasets = datasets
        n = len(datasets)
        self.similarity_matrix = np.zeros((n, n))
        self.error_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                score, error = self._pairwise_similarity(datasets[i], datasets[j])
                self.similarity_matrix[i, j] = score
                self.similarity_matrix[j, i] = score

                self.error_matrix[i][j] = error
                self.error_matrix[j][i] = error

        return self

    def get_similarity_matrix(self) -> np.ndarray:
        if self.similarity_matrix is None:
            raise RuntimeError("Call fit() first.")
        return self.similarity_matrix
    
    def get_error_matrix(self) -> np.ndarray:
        if self.error_matrix is None:
            raise RuntimeError("Call fit() first.")
        return self.error_matrix 

    # -------------------------
    # Internal helpers
    # -------------------------

    def _prepare_dataset(self, data: np.ndarray) -> TimeSeriesData:
        return TimeSeriesData(
            X=data[:-1],
            y=data[1:],
            train_val_test_split=self.split,
            lag=self.lag,
        )

    def _optimise_hyperparams(self, ds1: TimeSeriesData, ds2: TimeSeriesData):
        def objective(trial):
            bandwidth = trial.suggest_float("bandwidth", 1e-9, 4)
            reg_1 = trial.suggest_float("reg_1", 1e-12, 1e-2)
            reg_2 = trial.suggest_float("reg_2", 1e-12, 1e-2)

            model1 = self.model_cls(
                kernel=self.kernel, bandwidth=bandwidth, reg=reg_1
            )
            model2 = self.model_cls(
                kernel=self.kernel, bandwidth=bandwidth, reg=reg_2
            )

            X1_tr, y1_tr = ds1.train_data()
            X2_tr, y2_tr = ds2.train_data()
            X1_te, y1_te = ds1.test_data()
            X2_te, y2_te = ds2.test_data()

            model1.fit(X1_tr, y1_tr)
            model2.fit(X2_tr, y2_tr)

            mse = MeanSquaredError()
            return (
                mse(model1.predict(X1_te), y1_te)
                + mse(model2.predict(X2_te), y2_te)
            )

        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
        )

        return study.best_params, study.best_value

    def _fit_models_full(self, ds1, ds2, params):
        model1 = self.model_cls(
            kernel=self.kernel,
            bandwidth=params["bandwidth"],
            reg=params["reg_1"],
        )
        model2 = self.model_cls(
            kernel=self.kernel,
            bandwidth=params["bandwidth"],
            reg=params["reg_2"],
        )

        X1, y1 = ds1.full_data()
        X2, y2 = ds2.full_data()

        model1.fit(X1, y1)
        model2.fit(X2, y2)

        return model1, model2, X1, X2

    def _kernel_inner_product(self, model1, model2, X1, X2):
        kernels = model1.kernels

        if isinstance(kernels, list):
            K = sp.linalg.block_diag(
                *[k(X1, X2) for k in kernels]
            )
        else:
            K = sp.linalg.block_diag(
                *[
                    kernels(X1, X2)
                    for _ in range(X1.shape[-1])
                ]
            )

        return model1.alpha.T @ K @ model2.alpha

    def _pairwise_similarity(self, data1, data2) -> float:
        ds1 = self._prepare_dataset(data1)
        ds2 = self._prepare_dataset(data2)

        params, best_score = self._optimise_hyperparams(ds1, ds2)
        model1, model2, X1, X2 = self._fit_models_full(ds1, ds2, params)

        ip11 = self._kernel_inner_product(model1, model1, X1, X1)
        ip22 = self._kernel_inner_product(model2, model2, X2, X2)

        ip12 = self._kernel_inner_product(model1, model2, X1, X2)
        

        return ip12/np.sqrt(ip11*ip22), best_score
