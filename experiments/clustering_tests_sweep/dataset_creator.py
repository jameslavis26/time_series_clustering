import numpy as np

def create_dataset(
    theta: float,
    n_points: int,
    n_correlated_dimensions: int,
    n_uncorrelated_dimensions: int,
    noise: float = 0.0,
    damping: float = 0.9,
    seed: int | None = None,
):
    """
    Generate a bounded nonlinear dynamical system dataset.

    Dynamics are nonlinear but globally stable via tanh damping.
    """

    if not (0 < damping < 1):
        raise ValueError("damping must be in (0, 1)")
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    rng = np.random.default_rng(seed)

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
            x_next[i] += np.cos(theta * x[i]) * x[i]

            if i + 1 < n_correlated_dimensions:
                x_next[i] += np.sin(theta * x[i + 1])
                x_next[i + 1] += -np.sin(theta * x[i])

        # ------------------------------
        # Uncorrelated dimensions
        # ------------------------------
        for i in range(n_correlated_dimensions, n_dim):
            x_next[i] = np.cos(theta * x[i]) * x[i]

        # ------------------------------
        # Damping + noise (bounded step)
        # ------------------------------
        x_next = damping * np.tanh(x_next)
        x_next += rng.normal(0.0, noise, size=n_dim)

        X[t + 1] = x_next

    return X
