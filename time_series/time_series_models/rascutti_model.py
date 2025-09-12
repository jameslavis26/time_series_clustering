import numpy as np
import scipy as sp
from .time_series_base import TimeSeriesModel
import cvxpy as cp

import numpy as np
import scipy as sp
import cvxpy as cp
        
class RascuttiModelSingleTarget(TimeSeriesModel):
    def __init__(
        self, 
        kernels:list = None, 
        lam:float=1e-9, 
        rho:float=1e-9, 
        lag:int=1,
        kernel_perterbation = 1e-9
    ):
        super().__init__(lag=lag)
        if kernels == None:
            raise Exception("Please pass in a kernel")
        elif type(kernels) == list:
            self.kernels = kernels
        else:
            self.kernels = [kernels]

        self.lam = lam
        self.rho = rho
        self.kernel_perterbation = kernel_perterbation

    def fit(self, x:np.array, y:np.array):
        # Ensure yt is 1 dimensional
        if len(y.shape) != 1:
            raise Exception("RascuttiModelSingleTarget: y can only have one dimension")
        
        # Ensure enough kernels for each dimension
        if len(x.shape) == 1:
            if len(self.kernels) != 1:
                raise Exception("The number of kernels must match the dimension of X")
        elif x.shape[1] != len(self.kernels):    
            raise Exception("The number of kernels must match the dimension of X")

        # Record dimension of inputs
        self.d = 1 if len(x.shape) == 1 else x.shape[1]

        # Reshape data with a time series lag
        xt, yt = self.reshape_data(x, y)
        if self.d == 1:
            xt = xt[:, :, np.newaxis]

        self.N = xt.shape[0]

        self.xt_train = xt
        self.mean_y = np.mean(yt)
        
        kernels = [self.kernels[j](xt[:, :, j], xt[:, :, j]) for j in range(self.d)]

        # Cholesky decomposition where LL^T = K
        sqrt_kernels = [
            np.linalg.cholesky(
                kernel + self.kernel_perterbation*np.diag(np.random.random(kernel.shape[0]))
            ) for kernel in kernels
        ]

        alphas = [cp.Variable(self.N) for j in range(self.d)]
        t = cp.Variable()
        u = [cp.Variable() for j in range(self.d)]
        v = [cp.Variable() for j in range(self.d)]


        socp_constraints = [
            *[cp.SOC(1, cp.hstack([0.5, sqrt_kernels[j]@alphas[j]])) for j in range(self.d)],
            *[cp.SOC(v[j], kernels[j]@alphas[j]) for j in range(self.d)],
            *[cp.SOC(u[j], sqrt_kernels[j]@alphas[j]) for j in range(self.d)],
            cp.SOC(t, cp.hstack([0.5, yt - self.mean_y - np.sum([kernels[j]@alphas[j] for j in range(self.d)], axis=0)]))
        ]

        prob = cp.Problem(
            cp.Minimize(1/(2*self.N)*t + (self.lam/np.sqrt(self.N))*sum(v) + self.rho*sum(u)),
            socp_constraints
        )

        prob.solve()

        self.alphas = [alphas[j].value for j in range(self.d)]


    def predict(self, X):
        # Check input dimension is the same as training data
        dimension_x = 1 if len(X.shape) == 1 else X.shape[1]
        if dimension_x != self.d:
            raise Exception("The dimension of x should be the same dimension as the training data")
        
        x_test, _ = self.reshape_data(X)
        if dimension_x == 1:
            x_test = x_test[:, :, np.newaxis]

        kernels = [self.kernels[j](x_test[:, :, j], self.xt_train[:, :, j]) for j in range(self.d)]

        return np.sum([kernels[j]@self.alphas[j] for j in range(self.d)], axis=0) + self.mean_y   
    

class RascuttiModel:
    def __init__(self, 
        kernels:list = None, 
        lam:float=1e-9, 
        rho:float=1e-9, 
        lag:int=1,
        kernel_perterbation = 1e-9
    ):
        if kernels == None:
            raise Exception("Please pass in a kernel")
        elif type(kernels) == list:
            self.kernels = kernels
        else:
            self.kernels = [kernels]

        self.lam = lam
        self.rho = rho
        self.kernel_perterbation = kernel_perterbation
        self.lag = lag

    def fit(self, X, y=None):
        # Assign y if not assigned
        if type(y) == type(None):
            yt = X[1:]
            xt = X[:-1]
        else:
            xt = X
            yt = y

        self.d = 1 if len(xt.shape) == 1 else xt.shape[1]
        self.dy = 1 if len(yt.shape) == 1 else yt.shape[1]
        if self.dy == 1:
            yt = yt[:, np.newaxis]

        self.dimension_models = []
        for d in range(self.dy):
            model_d = RascuttiModelSingleTarget(
                kernels=self.kernels,
                lam=self.lam,
                rho=self.rho,
                lag=self.lag,
                kernel_perterbation=self.kernel_perterbation
            )
            model_d.fit(xt, yt[:, d])
            self.dimension_models.append(
                model_d
            ) 
    
    def predict(self, X):
        if self.dy == 1:
            result = [model.predict(X) for model in self.dimension_models][0]
        else:
            result = [model.predict(X) for model in self.dimension_models]
        return np.array(result).T
        