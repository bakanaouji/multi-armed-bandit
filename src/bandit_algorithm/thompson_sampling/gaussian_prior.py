import numpy as np
import pandas as pd


class ThompsonSamplingGaussianPrior(object):
    def __init__(self, N, save_log=False):
        self.N = N
        self.ks = np.zeros(N)
        self.mus = np.zeros(N)
        self.save_log = save_log
        self.thetas = []

    def initialize(self):
        self.ks = np.zeros(self.N)
        self.mus = np.zeros(self.N)
        self.thetas = []

    def select_arm(self):
        return np.argmax(self.estimate_mean())

    def estimate_mean(self):
        # For each arm i=1,...,N, sample random value from normal distribution
        theta = [np.random.normal(self.mus[i],
                                  np.math.sqrt(1.0 / (self.ks[i] + 1.0)))
                 for i in range(self.N)]
        if self.save_log:
            self.thetas.append(theta)
        return theta

    def update_param(self, arm_id, reward):
        # update parameter of normal distribution
        self.mus[arm_id] = (self.mus[arm_id] * (self.ks[arm_id] + 1.0)
                            + reward) / (self.ks[arm_id] + 2.0)
        self.ks[arm_id] += 1.0

    def save(self, folder_name):
        if self.save_log:
            self.thetas = pd.DataFrame(self.thetas)
            self.thetas.to_csv(folder_name + '/theta.csv')
