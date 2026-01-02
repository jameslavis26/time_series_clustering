import numpy as np
import scipy as sp
from ..kernels import GaussianKernel

class KernelRidgeRegression:
    def __init__(
        self, 
        kernel:str,
        reg:float=1e-9, 
        **kwargs
    ):
        if kernel == "gaussian" or kernel == "rbf":
            if "bandwidth" not in kwargs:
                raise Exception("Please specify rbf bandwidths")
            
            if type(kwargs["bandwidth"]) == list:
                self.kernels = [GaussianKernel(bandwidth=b) for b in kwargs["bandwidth"]]
            else:
                self.kernels = GaussianKernel(bandwidth=kwargs["bandwidth"])

        self.reg = reg

    def fit(self, x:np.array, y:np.array=None):
        if len(x.shape) == 1:
            N = x.shape[0]
            self.dimension_x = 1
        else:
            N = x.shape[0]
            self.dimension_x = x.shape[-1]

        if len(y.shape) == 1:
            self.dimension_y = 1
        else:
            self.dimension_y = y.shape[-1]

        self.x_train = x

        # Ensure a kernel for each dimension
        if type(self.kernels) == list:
            assert len(self.kernels) == self.dimension_x
            kernels = self.kernels
        else:
            kernels = [self.kernels]*self.dimension_x

        K_train = sp.linalg.block_diag(
            *[
                kernel(self.x_train, self.x_train) for kernel in kernels
            ]
        )

        # Reshape y to be a vector
        y_window = y.T.reshape(1, -1).T
        self.y_mean = y_window.mean(axis=0)

        # Fit KRR
        LHS = K_train + N*self.reg*np.eye(self.dimension_x*N)
        RHS = y_window - self.y_mean

        self.alpha = sp.linalg.solve(LHS, RHS)
    
    def predict(self, x):
        # Check input dimension is the same as training data
        dimension_x = 1 if len(x.shape) == 1 else x.shape[-1]
        if dimension_x != self.dimension_x:
            raise Exception("The dimension of x should be the same dimension as the training data")
    
        # Ensure a kernel for each dimension
        if type(self.kernels) == list:
            assert len(self.kernels) == self.dimension_x
            kernels = self.kernels
        else:
            kernels = [self.kernels]*self.dimension_x

        # Calculate kernel matrices
        K_test = sp.linalg.block_diag(
            *[
                kernel(x, self.x_train) for kernel in kernels
            ]
        )
        
        # Calculate prediciton
        y_pred = K_test@self.alpha + self.y_mean

        # Retrun prediciton in correct shape
        return y_pred.reshape(self.dimension_y, -1).T
    
    def inner_product(self, other):
        assert isinstance(other) == KernelRidgeRegression
        kernels = self.kernels

        if isinstance(kernels, list):
            K = sp.linalg.block_diag(
                *[k(self.x_train, other.X_train) for k in kernels]
            )
        else:
            K = sp.linalg.block_diag(
                *[
                    kernels(self.x_train, other.X_train)
                    for _ in range(self.x_train.shape[-1])
                ]
            )

        return self.alpha.T @ K @ other.alpha