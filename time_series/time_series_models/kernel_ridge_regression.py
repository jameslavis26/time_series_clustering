import numpy as np
import scipy as sp
from .time_series_base import TimeSeriesModel


class KernelRidgeRegression(TimeSeriesModel):
    def __init__(
        self, 
        kernels:list = None, 
        reg:float=1e-9, 
        lag=1, 
        **kwargs
    ):
        super().__init__(lag=lag)
        if kernels == None:
            raise Exception("Please pass in a kernel")
        elif type(kernels) == list:
            self.kernels = kernels
        else:
            self.kernels = [kernels]

        self.reg = reg
        self.lag = lag




    def fit(self, x:np.array, y:np.array=None):
        # Assign y if not assigned
        if type(y) == type(None):
            yt = x[1:]
            xt = x[:-1]
        else:
            xt = x
            yt = y

        # Ensure enough kernels for each dimension
        if len(yt.shape) == 1 and len(self.kernels) != 1:
            raise Exception("The number of kernels must match the dimension of y")
        elif yt.shape[1] != len(self.kernels):    
            raise Exception("The number of kernels must match the dimension of y")

        # Record dimension of inputs
        self.dimension_x = 1 if len(xt.shape) == 1 else xt.shape[1]
        self.dimension_y = 1 if len(yt.shape) == 1 else yt.shape[1]

        # Reshape data with a rolling window
        self.x_train, y_train = self.reshape_data(xt, yt)

        # Calculate kernel matrices
        K_train = sp.linalg.block_diag(
            *[
                kernel(self.x_train, self.x_train) for kernel in self.kernels
            ]
        )
        
        # Reshape y to be a vector
        y_window = y_train.T.reshape(-1, 1)

        # Fit KRR
        N = K_train.shape[0]

        LHS = K_train + N*self.reg*np.eye(N)
        RHS = y_window

        self.alpha = sp.linalg.solve(LHS, RHS)
    
    def predict(self, x):
        # Check input dimension is the same as training data
        dimension_x = 1 if len(x.shape) == 1 else x.shape[1]
        if dimension_x != self.dimension_x:
            raise Exception("The dimension of x should be the same dimension as the training data")

        # Reshape x with a rolling window
        x_test, _ = self.reshape_data(x)
    
        # Calculate kernel matrices
        K_test = sp.linalg.block_diag(
            *[
                kernel(x_test, self.x_train) for kernel in self.kernels
            ]
        )
        
        # Calculate prediciton
        y_pred = K_test@self.alpha

        # Retrun prediciton in correct shape
        return y_pred.reshape(self.dimension_y, -1).T