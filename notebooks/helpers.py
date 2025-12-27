from time_series.time_series_models import KernelRidgeRegression, RascuttiModel
from time_series.kernels import GaussianKernel

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator

import itertools
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np

class DataHolder:
    def __init__(self, **kwargs):
        self.update(**kwargs)
    
    def update(self, **kwargs):
        self.__dict__.update(kwargs)

class OptunaHyperparameterTuning:
    def __init__(self, estimator, param_grid, scoring=None, nfolds=5, n_trials=30):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring if scoring else mean_squared_error
        self.nfolds = nfolds
        self.n_trials = n_trials
    
    def fit(self, X, y=None):
        def objective(trial):
            parameters = dict()

            for p_name, p_params in self.param_grid.items():
                if p_params["type"] == "float":
                    step = p_params["step"] if "step" in p_params else None
                    parameters[p_name] = trial.suggest_float(p_name, p_params["min"], p_params["max"], step=step)
            
            model = self.estimator(**parameters)

            kf = KFold(n_splits=self.nfolds, random_state=None, shuffle=False)
            scores = []
            
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                X_train = X[train_index]
                X_test = X[test_index]

                y_train = y[train_index]
                y_test = y[test_index] 

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                score = self.scoring(y_test, y_pred)
                scores.append(score)
            
            cv_score = np.mean(scores)
            return cv_score


        study = optuna.create_study()
        study.optimize(
            objective,
            n_trials=self.n_trials,
        )

        self.best_params = study.best_params

class GridSearch:
    def __init__(self, estimator, param_grid, scoring=None, nfolds=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring if scoring else mean_squared_error
        self.nfolds = nfolds
    
    def fit(self, X, y=None):
        param_names = list(self.param_grid.keys())
        param_values = list(itertools.product(*self.param_grid.values()))

        result = []

        best_score = np.inf
        best_params = []
        
        for p in param_values:
            mapping = {param_names[i]:p[i] for i in range(len(param_names))}
            model = self.estimator(**mapping)

            kf = KFold(n_splits=self.nfolds, random_state=None, shuffle=False)
            scores = []
            
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                X_train = X[train_index]
                X_test = X[test_index]

                y_train = y[train_index]
                y_test = y[test_index] 

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                score = self.scoring(y_test, y_pred)
                scores.append(score)
            
            cv_score = np.mean(scores)
            result.append(
                (mapping, cv_score)
            )

            if cv_score < best_score:
                best_score = cv_score
                best_params = p
        
        self.best_score = best_score
        self.best_params = {param_names[i]:best_params[i] for i in range(len(param_names))}

        self.cv_results = result


class KRRWrapper(BaseEstimator):
    def __init__(self, bandwidth, reg):
        self.model = KernelRidgeRegression(
            kernels=[GaussianKernel(bandwidth=bandwidth)],
            reg = reg
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
def generate_time_series(x0:list, f, N=200, epsilon=0):
    x = np.zeros(shape=(N))
    for i in range(len(x0)):
        x[i] = x0[i]

    for i in range(len(x0), N):
        x[i] = f(x[:i]) + np.random.normal(0, epsilon)

    return x

def inner_product(m1, m2):
    K = GaussianKernel(bandwidth=1)
    return m1.alpha.T@K(m1.x_train/m1.kernels[0].bandwidth, m2.x_train/m2.kernels[0].bandwidth)@m2.alpha

def inner_product_b1(m1, m2):
    K = GaussianKernel(bandwidth=1)
    return m1.alpha.T@K(m1.x_train, m2.x_train)@m2.alpha