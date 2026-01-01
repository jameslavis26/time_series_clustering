import numpy as np
import scipy as sp
import cvxpy as cp
from ..kernels import GaussianKernel

class RascuttiModelSingleTarget:
    def __init__(
        self,
        kernel:str|list="gaussian",
        lam:float=1e-9,
        rho:float=1e-9,
        kernel_perterbation = 1e-9,
        **kwargs
    ):
        if kernel == "gaussian" or kernel == "rbf":
            if "bandwidth" not in kwargs:
                raise Exception("Please specify rbf bandwidths")
            
            if type(kwargs["bandwidth"]) == list:
                self.kernels = [GaussianKernel(bandwidth=b) for b in kwargs["bandwidth"]]
            else:
                self.kernels = GaussianKernel(bandwidth=kwargs["bandwidth"])
        elif type(kernel) == list:
            self.kernels = kernel
    

        self.lam = lam
        self.rho = rho
        self.kernel_perterbation = kernel_perterbation

    def fit(self, X:np.array, y:np.array):
        # Ensure yt is 1 dimensional
        if len(y.shape) != 1:
            raise Exception("RascuttiModelSingleTarget: y can only have one dimension")
        
        # Ensure enough kernels for each dimension
        if len(X.shape) == 1:
            if type(self.kernels) == list:
                raise Exception("The number of kernels must match the dimension of X")
        elif type(self.kernels) == list and x.shape[-1] != len(self.kernels):    
            raise Exception("The number of kernels must match the dimension of X")

        # Record dimension of inputs
        self.dimension_x = 1 if len(X.shape) == 1 else X.shape[-1]
        self.N = X.shape[0]

        self.x_train = X
        self.mean_y = np.mean(y)

        if type(self.kernels) == list:
            kernels = [self.kernels[j](X[..., j], X[..., j]) for j in range(self.dimension_x)]
        else:
            kernels = [self.kernels(X[..., i], X[..., i]) for i in range(self.dimension_x)]

        # Cholesky decomposition where LL^T = K
        sqrt_kernels = [
            np.linalg.cholesky(
                kernel + self.kernel_perterbation*np.diag(np.random.random(kernel.shape[0]))
            ) for kernel in kernels
        ]

        alphas = [cp.Variable(self.N) for j in range(self.dimension_x)]
        t = cp.Variable()
        u = [cp.Variable() for j in range(self.dimension_x)]
        v = [cp.Variable() for j in range(self.dimension_x)]


        socp_constraints = [
            *[cp.SOC(1, cp.hstack([0.5, sqrt_kernels[j]@alphas[j]])) for j in range(self.dimension_x)],
            *[cp.SOC(v[j], kernels[j]@alphas[j]) for j in range(self.dimension_x)],
            *[cp.SOC(u[j], sqrt_kernels[j]@alphas[j]) for j in range(self.dimension_x)],
            cp.SOC(t, cp.hstack([0.5, y - self.mean_y - np.sum([kernels[j]@alphas[j] for j in range(self.dimension_x)], axis=0)]))
        ]

        prob = cp.Problem(
            cp.Minimize(1/(2*self.N)*t + (self.lam/np.sqrt(self.N))*sum(v) + self.rho*sum(u)),
            socp_constraints
        )

        prob.solve()

        self.alphas = [alphas[j].value for j in range(self.dimension_x)]


    def predict(self, X):
        # Check input dimension is the same as training data
        dimension_x = 1 if len(X.shape) == 1 else X.shape[-1]
        if dimension_x != self.dimension_x:
            raise Exception("The dimension of x should be the same dimension as the training data")
        
        if type(self.kernels) == list:
            kernels = [self.kernels[j](X[..., j], self.x_train[..., j]) for j in range(self.dimension_x)]
        else:
            kernels = [self.kernels(X[..., j], self.x_train[..., j]) for j in range(self.dimension_x)]

        return np.sum([kernels[j]@self.alphas[j] for j in range(self.dimension_x)], axis=0) + self.mean_y   
class RascuttiModel:
    def __init__(self, 
        kernel:str="gaussian", 
        lam:float=1e-9, 
        rho:float=1e-9, 
        kernel_perterbation = 1e-9,
        **kwargs
    ):
        self.kwargs = kwargs
        if kernel == "gaussian" or kernel == "rbf":
            if "bandwidth" not in kwargs:
                raise Exception("Please specify rbf bandwidths")
            
            if type(kwargs["bandwidth"]) == list:
                self.kernels = [GaussianKernel(bandwidth=b) for b in kwargs["bandwidth"]]
            else:
                self.kernels = GaussianKernel(bandwidth=kwargs["bandwidth"])

        self.lam = lam
        self.rho = rho
        self.kernel_perterbation = kernel_perterbation

    def fit(self, X, y):
        if len(X.shape) == 1:
            N = X.shape[0]
            self.dimension_x = 1
        else:
            N = X.shape[0]
            self.dimension_x = X.shape[-1]

        if len(y.shape) == 1:
            self.dimension_y = 1
        else:
            self.dimension_y = y.shape[-1]

        self.x_train = X

        # Ensure a kernel for each dimension
        if type(self.kernels) == list:
            assert len(self.kernels) == self.dimension_x
            kernels = self.kernels
        else:
            kernels = [self.kernels]*self.dimension_x

        if self.dimension_y == 1:
            y = y[:, np.newaxis]

        self.dimension_models = []
        for d in range(self.dimension_y):
            model_d = RascuttiModelSingleTarget(
                kernel=kernels,
                lam=self.lam,
                rho=self.rho,
                kernel_perterbation=self.kernel_perterbation
            )
            model_d.fit(X, y[:, d])
            self.dimension_models.append(
                model_d
            ) 
    
    def predict(self, X):
        if self.dimension_y == 1:
            result = [model.predict(X) for model in self.dimension_models][0]
        else:
            result = [model.predict(X) for model in self.dimension_models]
        return np.array(result).T
        