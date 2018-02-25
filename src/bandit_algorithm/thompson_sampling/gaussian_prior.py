import numpy as np
import pandas as pd


class ThompsonSamplingGaussianPrior(object):
    def __init__(self, N, save_log=False):
        self.N = N
        self.k = np.zeros(N)
        self.mu = np.zeros(N)
        self.save_log = save_log
        self.thetas = []

    def initialize(self):
        self.k = np.zeros(self.N)
        self.mu = np.zeros(self.N)
        self.thetas = []

    def estimate_mean(self):
        # For each arm i=1,...,N, sample random value from normal distribution
        theta = [np.random.normal(self.mu[i],
                                  np.math.sqrt(1.0 / (self.k[i] + 1.0)))
                 for i in range(self.N)]
        if self.save_log:
            self.thetas.append(theta)
        return theta

    def update_param(self, arm_index, reward):
        # update parameter of normal distribution
        self.mu[arm_index] = (self.mu[arm_index] * (self.k[arm_index] + 1.0)
                              + reward) / (self.k[arm_index] + 2.0)
        self.k[arm_index] += 1.0

    def save(self, save_path):
        if self.save_log:
            self.thetas = pd.DataFrame(self.thetas)
            self.thetas.to_csv(save_path + '/theta.csv')
