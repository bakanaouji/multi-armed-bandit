import matplotlib.pyplot as plt
import os
import pandas as pd

from utils.random import *


class BanditCore(object):
    def __init__(self, arms, algorithm, args):
        self.arms = arms
        self.algorithm = algorithm
        self.show_log = args.show_log

    def experiment(self, save_path):
        self.algorithm.initialize()

        N = len(self.arms)

        regret = 0.0
        regrets = []
        thetas = []

        # main loop
        loop_num = 20000
        t = 0
        while True:
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
            if t % 5000 == 0:
                s = 'iteration: ' + str(t) + ', regret: ' + str(regret) + ', '
                for i in range(N):
                    s += 'est_mean: ' + str(theta[i]) + ', '
                print(s)
            t += 1
            if t > loop_num:
                break

        # plot log
        if self.show_log:
            plt.plot(regrets)
            plt.grid()
            plt.xscale('log')
            plt.xlim(0, loop_num)
            plt.ylim(0, 100)
            plt.show()

        # save log
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        regrets = pd.DataFrame(regrets)
        thetas = pd.DataFrame(thetas)
        regrets.to_csv(save_path + '/regret.csv')
        thetas.to_csv(save_path + '/theta.csv')
