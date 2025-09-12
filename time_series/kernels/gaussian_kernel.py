from .kernel_base import Kernel
import numpy as np

class GaussianKernel:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        
    def __call__(self, x1, x2):
        x1_dim = len(x1.shape)
        x2_dim = len(x2.shape)

        if x1_dim != x2_dim:
            raise Exception("GaussianKernel: x1 and x2 should have the same number of dimensions")

        if x1_dim == 2:
            ax = 2
        else:
            ax = (2, 3)        

        return np.exp(-np.linalg.norm(x2[np.newaxis, :, :]- x1[:, np.newaxis, :], axis=ax)**2/(2*self.bandwidth**2))


