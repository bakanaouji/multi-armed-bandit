import matplotlib.pyplot as plt
import os
import pandas as pd

from utils.random import *


class BanditCore(object):
    def __init__(self, arms, algorithm, args, save_path):
        self.arms = arms
        self.algorithm = algorithm
        self.show_log = args.show_log
        self.save_path = save_path

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
            if t % 1000 == 0:
                s = "iteration: " + str(t) + ", regret: " + str(regret) + ", "
                for i in range(N):
                    s += "est_mean: " + str(theta[i]) + ", "
                print(s)

        # plot log
        if self.show_log:
            plt.plot(regrets)
            plt.xscale('log')
            plt.show()

        # save log
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        regrets = pd.DataFrame(regrets)
        thetas = pd.DataFrame(thetas)
        regrets.to_csv(self.save_path + '/regret.csv')
        thetas.to_csv(self.save_path + '/theta.csv')
