import numpy as np
import pandas as pd


class UCB1(object):
    def __init__(self, N, save_log=False):
        self.N = N
        self.means = np.zeros(N)
        self.ks = np.zeros(N)
        self.g = 0
        self.save_log = save_log
        self.scores = []

    def initialize(self):
        self.means = np.zeros(self.N)
        self.ks = np.zeros(self.N)
        self.g = 0
        self.scores = []

    def select_arm(self):
        arm_id = -1
        for i in range(self.N):
            if self.ks[i] == 0:
                arm_id = i
        if arm_id < 0:
            score = self.means + np.sqrt(np.log(self.g) / (2.0 * self.ks))
            arm_id = np.argmax(score)
        else:
            score = np.zeros(self.N)
        if self.save_log:
            self.scores.append(score)
        return arm_id

    def update_param(self, arm_id, reward):
        self.means[arm_id] = (self.means[arm_id] * self.ks[arm_id] + reward) / (
                self.ks[arm_id] + 1.0)
        self.ks[arm_id] += 1.0
        self.g += 1

    def save(self, folder_name):
        if self.save_log:
            self.scores = pd.DataFrame(self.scores)
            self.scores.to_csv(folder_name + '/score.csv')
