import numpy as np
import pandas as pd

from utils.random import scaled_inverse_chi_squared


class ThompsonSamplingGaussianSicqPrior(object):
    def __init__(self, N, save_log=False):
        self.N = N
        self.ks = np.ones(N)
        self.mus = np.zeros(N)
        self.vs = np.ones(N)
        self.sigmas = np.ones(N)
        self.save_log = save_log
        self.variances = []
        self.thetas = []

    def initialize(self):
        self.ks = np.ones(self.N)
        self.mus = np.zeros(self.N)
        self.vs = np.ones(self.N)
        self.sigmas = np.ones(self.N)
        self.variances = []
        self.thetas = []

    def select_arm(self):
        return np.argmax(self.estimate_mean())

    def estimate_mean(self):
        # For each arm i=1,...,N, sample random value from sicq distribution
        variance = [scaled_inverse_chi_squared(self.vs[i], self.sigmas[i])
                    for i in range(self.N)]
        # For each arm i=1,...,N, sample random value from normal distribution
        theta = [np.random.normal(self.mus[i],
                                  np.math.sqrt(variance[i] / self.ks[i]))
                 for i in range(self.N)]
        if self.save_log:
            self.thetas.append(theta)
            self.variances.append(variance)
        return theta

    def update_param(self, arm_id, reward):
        sigma = self.sigmas[arm_id]
        v = self.vs[arm_id]
        mu = self.mus[arm_id]
        k = self.ks[arm_id]
        d = reward - mu
        # update parameter of sicq distribution
        self.sigmas[arm_id] = sigma * v / (v + 1.0) + d * d * k / (
                    (v + 1.0) * (k + 1.0))
        self.vs[arm_id] = v + 1.0
        # update parameter of normal distribution
        self.mus[arm_id] = (mu * k + reward) / (k + 1.0)
        self.ks[arm_id] = k + 1.0

    def save(self, folder_name):
        if self.save_log:
            self.thetas = pd.DataFrame(self.thetas)
            self.thetas.to_csv(folder_name + '/theta.csv')
            self.variances = pd.DataFrame(self.variances)
            self.variances.to_csv(folder_name + '/variance.csv')
