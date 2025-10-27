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

        if x1_dim == 1:
            return np.exp(-(x2[np.newaxis, :]- x1[:, np.newaxis])**2/(2*self.bandwidth**2))
        elif x1_dim == 2:
            return np.exp(-np.linalg.norm(x2[np.newaxis, :, :]- x1[:, np.newaxis, :], axis=2)**2/(2*self.bandwidth**2))
        else:
            return np.exp(-np.linalg.norm(x2[np.newaxis, :, :]- x1[:, np.newaxis, :], axis=(2, 3))**2/(2*self.bandwidth**2))

        raise Exception("Number of dimensions of the data is too big!")
