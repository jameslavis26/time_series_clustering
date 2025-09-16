import numpy as np

class LorenzGenerator:
    def __init__(self, **parameters):
        self.__dict__.update(parameters)

        # Process the noise mean parameter
        noise_mean = parameters["noise_mean"]
        if type(noise_mean) == list and len(noise_mean) != 3:
            raise Exception("If noise mean is a list, expecting a mean for [x, y, z]")
        elif type(noise_mean) in [int, float]:
            noise_mean = [noise_mean, noise_mean, noise_mean]
        self.noise_mean = noise_mean

        # Process the noise variance parameter
        noise_covariance = parameters["noise_covariance"]
        if type(noise_covariance) == list and (len(noise_covariance) != 3 or len(noise_covariance[0]) != 3):
            raise Exception("If noise covariance is a list, expecting a covariance matrix [[x11, y12, z13], [...], [...]]")
        elif type(noise_covariance) in [int, float]:
            noise_covariance = np.diag([noise_covariance, noise_covariance, noise_covariance])
        self.noise_covariance = noise_covariance

        # Process the x0  parameter
        x0 = parameters["x0"]
        if len(x0) != 3:
            raise Exception("Expecting a x0  [x, y, z]")
        self.x0 = x0    


    def __call__(self):
        return generate_lorenz_curve(
            x0 = self.x0,
            noise_mean=self.noise_mean,
            noise_cov = self.noise_covariance,
            T = self.T, 
            dt=self.dt, 
            rho=self.rho, 
            sigma=self.sigma, 
            beta=self.beta
        )

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