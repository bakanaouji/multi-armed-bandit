import matplotlib.pyplot as plt
import os
import pandas as pd

from utils.random import *


class BanditCore(object):
    def __init__(self, arms, algorithm):
        self.arms = arms
        self.algorithm = algorithm

    def experiment(self):
        N = len(self.arms)

        regret = 0.0
        regrets = []
        thetas = []

        # main loop
        for t in range(20000):
            # estimate mean of each arm
            theta = self.algorithm.estimate_mean()
            # select arm
            arm_index = np.argmax(theta)
            # play arm and observe reward
            reward = self.arms[arm_index].play()
            # update parameter of bandit algorithm
            self.algorithm.update_param(arm_index, reward)
            # update regret
            regret += self.arms[0].mean - self.arms[arm_index].mean
            # stock log
            regrets.append(regret)
            thetas.append([theta[i] for i in range(N)])
            # output
            if t % 500 == 0:
                s = "iteration: " + str(t) + ", regret: " + str(regret) + ", "
                for i in range(N):
                    s += "est_mean: " + str(theta[i]) + ", "
                print(s)

        # plot log
        plt.plot(regrets)
        plt.xscale('log')
        plt.show()

        # save log
        save_path = '../data/' + self.arms[0].__class__.__name__ \
                    + "/" + self.algorithm.__class__.__name__
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        regrets = pd.DataFrame(regrets)
        thetas = pd.DataFrame(thetas)
        regrets.to_csv(save_path + '/regret.csv')
        thetas.to_csv(save_path + '/theta.csv')
