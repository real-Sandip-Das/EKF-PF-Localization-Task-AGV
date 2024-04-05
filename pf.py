import numpy as np

from utils import minimized_angle


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # MY IMPLEMENTATION HERE
        action = np.empty_like(self.particles)
        for i in range(self.num_particles):
            action[i] = env.sample_noisy_action(u, self.alphas).reshape(-1)
            self.particles[i] = env.forward(self.particles[i], action[i]).reshape(-1)
            #innovation = action[i].reshape(-1, 1)-u.reshape(-1, 1)
            #innovation[2] = minimized_angle(innovation[2])
            #self.weights[i] *= env.likelihood(action[i].reshape(-1, 1)-u.reshape(-1, 1), env.noise_from_motion(u, self.alphas)).reshape(1)
        for i in range(self.num_particles):
            self.weights[i] = env.likelihood(minimized_angle(env.observe(self.particles[i], marker_id)-z), self.beta).reshape(1)
        sum_weights = np.sum(self.weights)
        self.weights = self.weights / sum_weights
        self.particles, self.weights = self.resample(self.particles, self.weights)
        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        new_particles, new_weights = particles, weights
        # MY IMPLEMENTATION HERE
        N = self.num_particles
        new_weights = np.ones_like(weights)/N
        cumulative_sum = weights[0]
        r = np.random.uniform(np.finfo(np.float).eps, 1/N)
        j = 0
        for i in range(N):
            while r > cumulative_sum:
                j+=1
                cumulative_sum+=weights[j]
            new_particles[i] = particles[j]
            r += 1/N
        return new_particles, new_weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov
