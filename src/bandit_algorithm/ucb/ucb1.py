import numpy as np


class UCB1(object):
    def __init__(self, N, save_log=False):
        self.N = N
        self.means = np.zeros(N)
        self.ks = np.zeros(N)
        self.g = 0
        self.save_log = save_log

    def initialize(self):
        self.means = np.zeros(self.N)
        self.ks = np.zeros(self.N)

    def select_arm(self):
        arm_id = -1
        for i in range(self.N):
            if self.ks[i] == 0:
                arm_id = i
        if arm_id < 0:
            scores = self.means + np.sqrt(np.log(self.g) / (2.0 * self.ks))
            arm_id = np.argmax(scores)
        return arm_id

    def update_params(self, arm_id, reward):
        self.means[arm_id] = (self.means[arm_id] * self.ks[arm_id] + reward) / (
                self.ks[arm_id] + 1.0)
        self.ks[arm_id] += 1.0
        self.g += 1
