import numpy as np

from utils.random import scaled_inverse_chi_squared


class ThompsonSamplingGaussianSicqPrior(object):
    def __init__(self, N):
        self.N = N
        self.k = np.ones(N)
        self.mu = np.zeros(N)
        self.v = np.ones(N)
        self.sigma = np.ones(N)

    def initialize(self):
        self.k = np.ones(self.N)
        self.mu = np.zeros(self.N)
        self.v = np.ones(self.N)
        self.sigma = np.ones(self.N)

    def estimate_mean(self):
        # For each arm i=1,...,N, sample random value from sicq distribution
        variance = [scaled_inverse_chi_squared(self.v[i], self.sigma[i])
                    for i in range(self.N)]
        # For each arm i=1,...,N, sample random value from normal distribution
        theta = [np.random.normal(self.mu[i],
                                  np.math.sqrt(variance[i] / self.k[i]))
                 for i in range(self.N)]
        return theta

    def update_param(self, arm_index, reward):
        sigma = self.sigma[arm_index]
        v = self.v[arm_index]
        mu = self.mu[arm_index]
        k = self.k[arm_index]
        d = reward - mu
        # update parameter of sicq distribution
        self.sigma[arm_index] = sigma * v / (v + 1.0) + d * d * k / ((v + 1.0) * (k + 1.0))
        self.v[arm_index] = v + 1.0
        # update parameter of normal distribution
        self.mu[arm_index] = (mu * k + reward) / (k + 1.0)
        self.k[arm_index] = k + 1.0
