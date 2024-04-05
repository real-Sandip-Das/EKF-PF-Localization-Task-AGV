import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # MY IMPLEMENTATION HERE
        x_k_1_k_1 = self.mu.copy()
        P_k_1_k_1 = self.sigma.copy()

        #Computing Jacobians

        #Predict step
        F_k = env.G(x_k_1_k_1, u) #Jacobian of the dynamics w.r.t. the state
        x_k_k_1 = env.forward(x_k_1_k_1, u)
        Q_k_1 = env.noise_from_motion(u, self.alphas)
        P_k_k_1 = F_k@P_k_1_k_1@F_k.T + Q_k_1

        #Update step
        H_k = env.H(x_k_k_1, marker_id) #Jacobian of the observation w.r.t. the state
        R_k = self.beta #Covariance matrix of the Measurement Noise
        y_tilde_k = z - env.observe(x_k_k_1, marker_id) #Innovation(measurement residual)
        S_k = H_k@P_k_k_1@(H_k.T) + R_k #Innovation(or Residual) Covariance
        K_k = P_k_k_1@H_k.T@np.linalg.inv(S_k) #Near-optimal Kalman gain
        x_k_k = x_k_k_1 + K_k@y_tilde_k #Updated State estimate
        P_k_k = (np.eye(3) - K_k@H_k)@P_k_k_1 #Updated Covariance estimate
        x_k_k[2] = minimized_angle(x_k_k[2])

        self.mu = x_k_k.reshape((-1, 1))
        self.sigma = P_k_k
        return self.mu, self.sigma
