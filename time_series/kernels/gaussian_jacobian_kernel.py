from .kernel_base import Kernel
import numpy as np
from scipy.special import lambertw

def sigmak(lam, n, p, lmax):
    """
    Estimate for the bandwidth based on 
    - k = 0
    - lam: regularisation parameter
    - n: dataset size
    - p: dataset dimension
    - lmax: maximum distance between any two training observations (possibly sensitive to outliers)
    """
    if lam <= 2*n*np.exp(-3/2):
        result =  (np.sqrt(2)/np.pi)*(lmax/((n-1)**(1/p) - 1))*np.sqrt(1 - lambertw(-lam*np.sqrt(np.exp(1))/(2*n), 0))   
    else: 
        new_lam =  2*n*np.exp(-3/2)
        result (np.sqrt(2)/np.pi)*(lmax/((n-1)**(1/p) - 1))*np.sqrt(1 - lambertw(-new_lam*np.sqrt(np.exp(1))/(2*n), 0))
    return np.abs(result)

class JacobianGaussianKernel:
    """
    On the first call, the kernel parameters will be evaluated. After which the kernel parameters
    are set.
    """
    def __init__(self):
        self.requires_update = True
        self.called = False

    def update_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def find_parameters(self, x):
        # Evaluate kernel parameters
        self.called = True

        if len(x.shape) == 2:
            ax = 2
            n, p = x.shape
        else:
            ax = (2, 3) 
            n, _, p = x.shape
        cross_distance = np.linalg.norm(x[np.newaxis, :, :]- x[:, np.newaxis, :], axis=ax)
        
        lmax = cross_distance.max()

        self.sigma0 = sigmak(self.lam, n, p, lmax)
        
    def __call__(self, x1, x2):
        x1_dim = len(x1.shape)
        x2_dim = len(x2.shape)

        if x1_dim != x2_dim:
            raise Exception("GaussianKernel: x1 and x2 should have the same number of dimensions")

        if x1_dim == 2:
            ax = 2
        else:
            ax = (2, 3) 

        if self.called:
            return np.exp(-np.linalg.norm(x2[np.newaxis, :, :]- x1[:, np.newaxis, :], axis=ax)**2/(2*self.sigma0**2))

        self.find_parameters(x1)
        return np.exp(-np.linalg.norm(x2[np.newaxis, :, :]- x1[:, np.newaxis, :], axis=ax)**2/(2*self.sigma0**2))

