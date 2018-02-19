import numpy as np


class ThompsonSamplingGaussianPrior(object):
    def __init__(self, N):
        self.N = N
        self.k = np.zeros(N)
        self.mu = np.zeros(N)

    def estimate_mean(self):
        # For each arm i=1,...,N, sample random value from normal distribution
        theta = [np.random.normal(self.mu[i], 1.0 / (self.k[i] + 1.0))
                 for i in range(self.N)]
        return theta

    def update_param(self, arm_index, reward):
        # update parameter of normal distribution
        self.mu[arm_index] = (self.mu[arm_index] * (self.k[arm_index] + 1.0)
                              + reward) / (self.k[arm_index] + 2.0)
        self.k[arm_index] += 1
