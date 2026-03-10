import numpy as np
import cvxpy as cp
from math import factorial
from scipy.special import eval_hermitenorm

class GaussianEigenfunctions:
    """
    Eigenvalues and eigenfunctions of the 1D Gaussian kernel.

    Eigenfunctions are scaled probabilists' Hermite polynomials
    multiplied by a Gaussian envelope.
    """

    def __init__(self, bandwidth: float):
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

        self.bandwidth = float(bandwidth)

        # Stable closed-form parameter
        self.p = (
            -0.5 * self.bandwidth**2
            + 0.5 * np.sqrt(self.bandwidth**4 + 4.0)
        )

    def eigenvalue(self, k: int) -> float:
        """Return the k-th Mercer eigenvalue."""
        if k < 0:
            raise ValueError("k must be non-negative")

        return (
            self.p**k
            * np.sqrt(1.0 - self.p**2)
            * np.sqrt(2.0 * np.pi)
        )

    def eigenfunction(self, k: int):
        """
        Return the k-th eigenfunction as a callable.

        Accepts scalars or NumPy arrays.
        """
        if k < 0:
            raise ValueError("k must be non-negative")

        norm_const = np.sqrt(factorial(k)) * (2.0 * np.pi) ** 0.25
        exponent = self.p / (2.0 * (1.0 + self.p))

        def phi(x):
            x = np.asarray(x)
            return (
                eval_hermitenorm(k, x)
                * np.exp(-exponent * x**2)
                / norm_const
            )

        return phi

class EigenRascuttiSingleTarget:
    def __init__(
        self,
        bandwidth,
        lam = 1e-9,
        rho = 1e-9,
        eigen_K = 10
    ):
        self.bandwidth = float(bandwidth)
        self.eigen_K = int(eigen_K)
        self.lam = float(lam)
        self.rho = float(rho)

        self.gaussian_eigen = GaussianEigenfunctions(self.bandwidth)

    def fit(self, X, y):
        # Ensure yt is 1 dimensional
        if len(y.shape) != 1:
            raise Exception("EigenRascuttiSingleTarget: y can only have one dimension")
        
        # Record dimension of inputs
        dimension_x = 1 if len(X.shape) == 1 else X.shape[-1]
        N = X.shape[0]

        self.X_train = X.squeeze()
        self.y_shift = np.mean(y, axis=0)
        y_centered = y - self.y_shift

        # Mercer eigenpairs
        D = np.diag([
            self.gaussian_eigen.eigenvalue(k)
            for k in range(self.eigen_K)
        ]) * N

        D_inv_sqrt = np.sqrt(np.linalg.inv(D))


        Q_k = np.stack(
            [self.gaussian_eigen.eigenfunction(k)(X)
             for k in range(self.eigen_K)],
            axis=2
        ).squeeze().transpose(2, 0, 1) / np.sqrt(N) # d, N, k

        # Optimisation variables
        betas = [cp.Variable(self.eigen_K) for _ in range(dimension_x)]
        t = cp.Variable()
        u = [cp.Variable(nonneg=True) for _ in range(dimension_x)]
        v = [cp.Variable(nonneg=True) for _ in range(dimension_x)]

        constraints = []

        for j in range(dimension_x):
            constraints.append(
                cp.SOC(
                    1 / np.sqrt(2) * (1.5),
                    cp.hstack([
                        1 / np.sqrt(2) * (-0.5),
                        D_inv_sqrt @ betas[j],
                    ])
                )
            )
            constraints.append(cp.SOC(v[j], betas[j]))
            constraints.append(cp.SOC(u[j], D_inv_sqrt @ betas[j]))

        Q_sum = cp.sum(
            [Q_k[j]@ betas[j] for j in range(dimension_x)],
            axis=0
        )

        constraints += [
            cp.SOC(
                np.sqrt(2) * (t + 0.5) / 2,
                cp.hstack([
                    np.sqrt(2) * (t - 0.5) / 2,
                    y - Q_sum,
                ])
            )
        ]

        objective = cp.Minimize(
            (1 / (2 * N)) * t
            + (self.lam / np.sqrt(N)) * cp.sum(v)
            + self.rho * cp.sum(u)
        )


        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        self.betas = [b.value for b in betas]        

    def predict(self, X):
        # Record dimension of inputs
        dimension_x = 1 if len(X.shape) == 1 else X.shape[-1]
        N = X.shape[0]

        Q_k = np.stack(
            [self.gaussian_eigen.eigenfunction(k)(X)
             for k in range(self.eigen_K)],
            axis=2
        ).squeeze().transpose(2, 0, 1) / np.sqrt(N) # d, N, k

        fits = [Q_k[j]@self.betas[j] for j in range(dimension_x)]

        return np.sum(fits, axis=0) + self.y_shift
        

class EigenRascuttiModel:
    def __init__(self, 
        bandwidth,
        lam:float=1e-9, 
        rho:float=1e-9, 
        **kwargs
    ):
        self.bandwidth = bandwidth
        self.lam = lam
        self.rho = rho

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

        if self.dimension_y == 1:
            y = y[:, np.newaxis]

        self.dimension_models = []
        for d in range(self.dimension_y):
            model_d = EigenRascuttiSingleTarget(
                bandwidth=self.bandwidth,
                lam=self.lam,
                rho=self.rho,
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