import numpy as np


class NormalDistributionArm(object):
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def play(self):
        return np.random.normal(self.mean, self.sigma)

    def name(self):
        return 'N(' + str(self.mean) + ',' + str(self.sigma * self.sigma) + ')'
