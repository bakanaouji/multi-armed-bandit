import matplotlib.pyplot as plt
import os
import pandas as pd


class BanditCore(object):
    def __init__(self, arms, algorithm, args):
        self.arms = arms
        self.algorithm = algorithm
        self.play_num = args.play_num
        self.save_log = args.save_log
        self.show_log = args.show_log

    def experiment(self, folder_name):
        self.algorithm.initialize()

        regret = 0.0
        regrets = []

        # main loop
        t = 0
        while True:
            # select arm
            arm_id = self.algorithm.select_arm()
            # play arm and observe reward
            reward = self.arms[arm_id].play()
            # update parameter of bandit algorithm
            self.algorithm.update_param(arm_id, reward)
            # update regret
            regret += self.arms[0].mean - self.arms[arm_id].mean
            # stock log
            regrets.append(regret)
            # output
            if t % 5000 == 0:
                s = 'iteration: ' + str(t) + ', selected arm: ' + str(arm_id) \
                    + ', regret: ' + str(regret)
                print(s)
            t += 1
            if t > self.play_num:
                break

        # plot log
        if self.show_log:
            plt.plot(regrets)
            plt.grid()
            plt.xscale('log')
            plt.xlim(0, self.play_num)
            plt.ylim(0, 100)
            plt.show()

        # save log
        if self.save_log:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            regrets = pd.DataFrame(regrets)
            regrets.to_csv(folder_name + '/regret.csv')
            self.algorithm.save(folder_name)
