import numpy as np
import cvxpy as cp
from math import factorial
from scipy.special import eval_hermitenorm


# ============================================================
# Gaussian kernel eigenpairs (1D Mercer decomposition)
# ============================================================

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


# ============================================================
# Rascutti-style regression in Gaussian eigenbasis
# ============================================================

class EigenGausRascutti:
    """
    Rascutti regression using Gaussian kernel eigen-expansion.
    """

    def __init__(self, bandwidth, lam=0.0, rho=0.0, eigen_K=10):
        self.bandwidth = float(bandwidth)
        self.eigen_K = int(eigen_K)
        self.lam = float(lam)
        self.rho = float(rho)

        self.gaussian_eigen = GaussianEigenfunctions(self.bandwidth)

    # --------------------------------------------------------
    # Internal solver
    # --------------------------------------------------------

    def _fit_rascutti(
        self,
        X,
        y,
        eigenvalues,
        eigenvectors,
        eigen_K,
        lam,
        rho,
    ):
        """
        Solve the SOCP for a single output dimension.
        """
        N, d = X.shape

        # Mercer feature blocks
        Q_k = eigenvectors[:, :, :eigen_K]   # (d, N, K)
        D = np.diag(eigenvalues[:eigen_K])

        D_inv_sqrt = np.sqrt(np.linalg.inv(D))

        # Optimisation variables
        betas = [cp.Variable(eigen_K) for _ in range(d)]
        t = cp.Variable()
        u = [cp.Variable(nonneg=True) for _ in range(d)]
        v = [cp.Variable(nonneg=True) for _ in range(d)]

        Q_sum = cp.sum(
            [Q_k[j] @ betas[j] for j in range(d)],
            axis=0
        )

        constraints = []

        # RKHS norm constraints
        for j in range(d):
            constraints += [
                cp.SOC(
                    1 / np.sqrt(2) * (1.5),
                    cp.hstack([
                        1 / np.sqrt(2) * (-0.5),
                        D_inv_sqrt @ betas[j],
                    ])
                ),
                cp.SOC(v[j], betas[j]),
                cp.SOC(u[j], D_inv_sqrt @ betas[j]),
            ]

        # Data fidelity constraint
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
            + (lam / np.sqrt(N)) * cp.sum(v)
            + rho * cp.sum(u)
        )

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        return betas

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def fit(self, X, Y):
        """
        Fit model to inputs X and outputs Y.
        """
        N, d = X.shape
        Y = np.atleast_2d(Y)

        self.y_shift = np.mean(Y, axis=0)
        y_centered = Y - self.y_shift

        # Mercer eigenpairs
        eigenvalues = np.array([
            self.gaussian_eigen.eigenvalue(k)
            for k in range(self.eigen_K)
        ]) * N

        eigenvectors = np.stack(
            [self.gaussian_eigen.eigenfunction(k)(X)
             for k in range(self.eigen_K)],
            axis=2
        ).transpose(1, 0, 2) / np.sqrt(N)

        betas = [
            self._fit_rascutti(
                X,
                y_centered[:, i],
                eigenvalues,
                eigenvectors,
                self.eigen_K,
                self.lam,
                self.rho,
            )
            for i in range(Y.shape[1])
        ]

        # Store as (outputs, dims, K)
        self.beta = np.stack([
            np.vstack([b.value for b in betas_i]).T
            for betas_i in betas
        ])

        return self

    def predict(self, X):
        """
        Predict outputs for new inputs X.
        """
        N, d = X.shape

        eigenvectors = np.stack(
            [self.gaussian_eigen.eigenfunction(k)(X)
             for k in range(self.eigen_K)],
            axis=2
        ).transpose(1, 0, 2) / np.sqrt(N)

        fits = np.einsum("vdf,tfv->dt", eigenvectors, self.beta)
        return fits + self.y_shift
