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
        shared_hparams:dict = None,
        hparams:dict = None
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

        if shared_hparams:
            self._verify_paramdict(shared_hparams)
            self.shared_hparams = shared_hparams
        else:
            self.shared_hparams = dict()

        if hparams:
            self._verify_paramdict(hparams)
            self.hparams = hparams
        else:
            self.hparams = dict()

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

    def _verify_paramdict(self, paramdict):
        if not isinstance(paramdict, dict):
            raise TypeError("Hyperparameters must be provided as a dictionary")

        for param, spec in paramdict.items():
            if not isinstance(spec, dict):
                raise TypeError(
                    f"Hyperparameter '{param}' must be a dict with keys: type, min, max (if applicable)"
                )

            # --- type ---
            if "type" not in spec:
                raise ValueError(f"Hyperparameter '{param}' is missing required key 'type'")

            if spec["type"] not in (int, float, list):
                raise ValueError(
                    f"Hyperparameter '{param}': type must be int, float, or list"
                )

            # --- int / float params ---
            if spec["type"] in (int, float):
                if "min" not in spec or "max" not in spec:
                    raise ValueError(
                        f"Hyperparameter '{param}' of type {spec['type'].__name__} "
                        "must specify 'min' and 'max'"
                    )

                if not isinstance(spec["min"], spec["type"]) or not isinstance(spec["max"], spec["type"]):
                    raise TypeError(
                        f"Hyperparameter '{param}': 'min' and 'max' must be of type {spec['type'].__name__}"
                    )

                if spec["min"] >= spec["max"]:
                    raise ValueError(
                        f"Hyperparameter '{param}': 'min' must be < 'max'"
                    )

            # --- list params ---
            elif spec["type"] is list:
                if "values" not in spec:
                    raise ValueError(
                        f"Hyperparameter '{param}' of type list must specify 'values'"
                    )

                if not isinstance(spec["values"], list) or len(spec["values"]) == 0:
                    raise TypeError(
                        f"Hyperparameter '{param}': 'values' must be a non-empty list"
                )


    def _prepare_dataset(self, data: np.ndarray) -> TimeSeriesData:
        return TimeSeriesData(
            X=data[:-1],
            y=data[1:],
            train_val_test_split=self.split,
            lag=self.lag,
        )

    def _optimise_hyperparams(self, ds1: TimeSeriesData, ds2: TimeSeriesData):
        def objective(trial):
            model1_params = dict()
            model2_params = dict()

            for param_name, param_dict in self.shared_hparams.items():
                if param_dict["type"] == int:
                    param_val = trial.suggest_int(param_name, param_dict["min"], param_dict["max"])
                elif param_dict["type"] == float:
                    param_val = trial.suggest_float(param_name, param_dict["min"], param_dict["max"])
                elif param_dict["type"] == list:
                    param_val = trial.suggest_categorical(param_name, param_dict["values"])

                model1_params[param_name] = param_val
                model2_params[param_name] = param_val

            for param_name, param_dict in self.hparams.items():
                if param_dict["type"] == int:
                    param_val1 = trial.suggest_int(param_name + "_1", param_dict["min"], param_dict["max"])
                    param_val2 = trial.suggest_int(param_name + "_2", param_dict["min"], param_dict["max"])
                elif param_dict["type"] == float:
                    param_val1 = trial.suggest_float(param_name + "_1", param_dict["min"], param_dict["max"])
                    param_val2 = trial.suggest_float(param_name + "_2", param_dict["min"], param_dict["max"])
                elif param_dict["type"] == list:
                    param_val1 = trial.suggest_categorical(param_name + "_1", param_dict["values"])
                    param_val2 = trial.suggest_categorical(param_name + "_2", param_dict["values"])

                model1_params[param_name] = param_val1
                model2_params[param_name] = param_val2

            model1 = self.model_cls(
                kernel=self.kernel, **model1_params
            )
            model2 = self.model_cls(
                kernel=self.kernel, **model2_params
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
        model1_params = dict()
        model2_params = dict()

        for k, v in params.items():
            if k[-2] == "_1":
                model1_params[k[:-2]] = v
            elif k[-2] == "_2":
                model2_params[k[:-2]] = v
            else:
                model1_params[k] = v
                model2_params[k] = v

        model1 = self.model_cls(
            kernel=self.kernel,
            **model1_params
        )
        model2 = self.model_cls(
            kernel=self.kernel,
            **model2_params
        )

        X1, y1 = ds1.full_data()
        X2, y2 = ds2.full_data()

        model1.fit(X1, y1)
        model2.fit(X2, y2)

        return model1, model2, X1, X2

    def _pairwise_similarity(self, data1, data2) -> float:
        ds1 = self._prepare_dataset(data1)
        ds2 = self._prepare_dataset(data2)

        params, best_score = self._optimise_hyperparams(ds1, ds2)
        model1, model2, X1, X2 = self._fit_models_full(ds1, ds2, params)

        ip11 = model1.inner_product(model1)
        ip22 = model2.inner_product(model2)

        ip12 = model1.inner_product(model2)

        return ip12/np.sqrt(ip11*ip22), best_score
