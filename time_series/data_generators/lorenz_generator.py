import numpy as np

def generate_lorenz_curve(
        x0 = [1,1,1], 
        noise_cov = 0*np.eye(3), 
        noise_mean=[0,0,0], 
        T = 100, 
        dt=1e-2, 
        rho=28, 
        sigma=12, 
        beta=8/3
    ):
    # Set initial conditionsfitted_time_series[1].self_kernel
    X = np.array([x0])

    N = int(T/dt)

    for n in range(N):
        A = np.array(
            [
                [-sigma, sigma, 0],
                [rho, -1, -X[-1, 0]],
                [0, X[-1, 0], -beta]
            ]
        )

        new_x = X[-1, :] + dt*A@X[-1, :] + np.random.multivariate_normal(noise_mean, noise_cov)
        X = np.vstack([X, new_x])

    t = np.linspace(0, T, N)

    return t, X