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
    
class SOCPProblem:
    def __init__(self, N, d, eigen_K):
        self.key = (N, d, eigen_K)

        self._build_problem(N, eigen_K, d)

    def _build_problem(self, N, eigen_K, d):
        # Parameters
        Dinvsqroot = cp.Parameter((eigen_K, eigen_K))
        y = cp.Parameter(N)
        Q_k = [cp.Parameter((N, eigen_K)) for j in range(d)]
        # Q_k = cp.Parameter((d, N, eigen_K))
        lam = cp.Parameter(nonneg=True)
        rho = cp.Parameter(nonneg=True)

        # Variables
        betas = [cp.Variable(eigen_K) for j in range(d)]
        t = cp.Variable()
        u = [cp.Variable() for j in range(d)]
        v = [cp.Variable() for j in range(d)]

        Q_sum = cp.sum([Q_k[j]@betas[j] for j in range(d)], axis=0)

        socp_constraints = [
            *[cp.SOC(1/np.sqrt(2) * (0.5 + 1), cp.hstack([1/np.sqrt(2) * (0.5 - 1), Dinvsqroot@betas[j]])) for j in range(d)],
            *[cp.SOC(v[j], betas[j]) for j in range(d)],
            *[cp.SOC(u[j], Dinvsqroot@betas[j]) for j in range(d)],
            cp.SOC(np.sqrt(2)*(t + 0.5)/2, cp.hstack([np.sqrt(2)*(t - 0.5)/2, y - Q_sum]))
        ]

        sum_v = cp.Variable(nonneg=True)  # scalar surrogate
        sum_u = cp.Variable(nonneg=True)

        socp_constraints += [sum_v == cp.sum(v)]
        socp_constraints += [sum_u == cp.sum(u)]

        prob = cp.Problem(
            cp.Minimize(1/(2*N) * t + lam  * sum_v + rho * sum_u),            
            socp_constraints
        )

        self._Dinvsqroot = Dinvsqroot
        self._y = y
        self._Q_k = Q_k
        self._lam = lam
        self._rho = rho
        self._betas = betas
        self._prob = prob
    
    def solve_problem(self,
        Dinvsqroot,
        y,
        Q_k,
        lam,
        rho,            
    ):
        self._Dinvsqroot.value = Dinvsqroot
        self._y.value = y
        self._lam.value = lam
        self._rho.value = rho

        for j in range(len(self._Q_k)):
            self._Q_k[j].value = Q_k[j]

        try:
            self._prob.solve(solver=cp.CLARABEL)
        except:
            self._prob.solve(solver=cp.SCS, warm_start=False)

        return self._betas
    
class EigenGausRascutti:
    _problem_cache = {}

    def __init__(self, bandwidth, lam=0, rho=0, eigen_K = 10, problem_cache=None):
        self.bandwidth = bandwidth
        self.eigen_K = eigen_K
        self.lam = lam
        self.rho = rho
        self.gaussian_eigen = GaussianEigenfunctions(bandwidth)               

    def _fit_rascutti(self, X, y, eigenvalues, eigenvectors, eigen_K, lam, rho):
        N, d = X.shape

        key = (N, d, eigen_K)
        if key not in EigenGausRascutti._problem_cache:
            solver = SOCPProblem(N, d, eigen_K)
            EigenGausRascutti._problem_cache[key] = solver

        solver = EigenGausRascutti._problem_cache[key]

        Q_mercer_k = eigenvectors[:, :, :eigen_K] #[dim, N, k]
        Q_k = Q_mercer_k 
        D = np.diag(eigenvalues[:eigen_K])
        Dinvsqroot = np.sqrt(np.linalg.inv(D))

        betas = solver.solve_problem(
            Dinvsqroot=Dinvsqroot,
            y=y,
            Q_k=Q_k,
            lam=lam,
            rho=rho, 
        )

        return betas

    def fit(self, X, Y):
        if len(X.shape) == 1:
            N = len(X)
            d = 1
        else:
            N = X.shape[0]
            d = X.shape[-1]

        self.y_shift = np.mean(Y, axis=0)

        y_centered = Y - self.y_shift

        mercer_eigenvalues = np.array([self.gaussian_eigen.eigenvalue(k) for k in range(self.eigen_K)])
        mercer_eigenvectors = np.stack([self.gaussian_eigen.eigenfunction(k)(X) for k in range(self.eigen_K)], axis=2).transpose(1, 0, 2)

        eigenvalues = mercer_eigenvalues*N
        eigenvectors = mercer_eigenvectors/np.sqrt(N)

        betas = [
            self._fit_rascutti(
                X, 
                y_centered[:, i], 
                eigenvalues=eigenvalues, 
                eigenvectors=eigenvectors,
                eigen_K=self.eigen_K,
                lam = self.lam,
                rho = self.rho
            ) for i in range(Y.shape[1])
        ]

        self.beta = np.stack([np.vstack([b.value for b in betas[j]]).T for j in range(len(betas))])

    def predict(self, X):
        N, d= X.shape
        mercer_eigenvectors = np.stack([self.gaussian_eigen.eigenfunction(k)(X) for k in range(self.eigen_K)], axis=2).transpose(1, 0, 2)
        eigenvectors = mercer_eigenvectors/np.sqrt(N)

        Q_k = eigenvectors
        fits = np.einsum("vdf,tfv->dt", Q_k, self.beta) + self.y_shift
        return fits
    
    def inner_product(self, other):
        mercer_eigenvalues = np.array([self.gaussian_eigen.eigenvalue(k) for k in range(self.eigen_K)])
        D = np.diag(mercer_eigenvalues[:self.eigen_K])
        D_inv = np.linalg.inv(D)
        
        # Matrix multiplication over dimensions. 
        ip = np.einsum("ilj,ll,ilp->", self.beta, D_inv, other.beta)

        return ip


