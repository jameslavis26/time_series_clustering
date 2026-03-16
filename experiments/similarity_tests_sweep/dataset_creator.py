import numpy as np

def dynamics_sincos(
        theta: float,
        n_correlated_dimensions: int,
        n_uncorrelated_dimensions: int,
    ):

    n_dim = n_correlated_dimensions + n_uncorrelated_dimensions

    def f(x):
        # print(x.shape)
        x_next = np.zeros_like(x)

        for i in range(n_correlated_dimensions):
            x_next[i] += np.cos(theta * x[i]) 
            if i > 0:
                x_next[i] += -np.sin(theta*x[i-1])

            if i + 1 < n_dim:
                x_next[i] += np.sin(theta*x[i+1])

        # ------------------------------
        # Uncorrelated dimensions
        # ------------------------------
        for i in range(n_correlated_dimensions, n_dim):
            x_next[i] = np.cos(theta * x[i])

        return x_next
    return f

def time_series_generator(
    dynamics,
    x0: np.array,
    n_points: int,
    noise: float = 0.0,
):
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    rng = np.random.default_rng()
    n_dim = len(x0)

    X = np.empty((n_points + 1, n_dim))
    X[0] = x0

    for t in range(n_points):
        x = X[t]
        x_next = dynamics(x)

        x_next += rng.normal(0.0, noise, size=n_dim)

        X[t + 1] = x_next

    return X


def create_dataset(
    theta: float,
    n_points: int,
    n_correlated_dimensions: int,
    n_uncorrelated_dimensions: int,
    noise: float = 0.0,
    seed: int | None = None,
):
    """
    Generate a bounded nonlinear dynamical system dataset.

    """

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    rng = np.random.default_rng()

    n_dim = n_correlated_dimensions + n_uncorrelated_dimensions
    X = np.empty((n_points + 1, n_dim))
    X[0] = rng.uniform(-1.0, 1.0, size=n_dim)

    for t in range(n_points):
        x = X[t]
        x_next = np.zeros_like(x)

        # ------------------------------
        # Correlated nonlinear dynamics
        # ------------------------------
        for i in range(n_correlated_dimensions):
            x_next[i] += np.cos(theta * x[i]) 
            if i > 0:
                x_next[i] += -np.sin(theta*x[i-1])

            if i + 1 < n_dim:
                x_next[i] += np.sin(theta*x[i+1])

        # ------------------------------
        # Uncorrelated dimensions
        # ------------------------------
        for i in range(n_correlated_dimensions, n_dim):
            x_next[i] = np.cos(theta * x[i])

        # ------------------------------
        # Damping + noise (bounded step)
        # ------------------------------
        x_next += rng.normal(0.0, noise, size=n_dim)

        X[t + 1] = x_next

    return X
