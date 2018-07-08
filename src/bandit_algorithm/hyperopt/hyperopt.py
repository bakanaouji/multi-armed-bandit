import matplotlib.pyplot as plt
import os
import pandas as pd

from hyperopt import fmin, tpe, Trials


class HyperOpt(object):
    def __init__(self, arms, hyper_params, play_num, save_log=False, show_log=False):
        self.arms = arms
        self.N = len(arms)
        self.play_num = play_num
        self.hyper_params = hyper_params
        self.save_log = save_log
        self.show_log = show_log
        self.trials = Trials()
        self.regret = 0.0
        self.regrets = []
        self.t = 0

    def initialize(self):
        self.trials = Trials()
        self.regret = 0.0
        self.regrets = []
        self.t = 0

    def one_iteration(self, params):
        # select arm
        arm_id = params['arm']
        # play arm and observe reward
        reward = self.arms[arm_id].play()
        # update regret
        self.regret += self.arms[0].mean - self.arms[arm_id].mean
        # stock log
        self.regrets.append(self.regret)
        # output
        if self.t % 1000 == 0:
            s = 'iteration: ' + str(self.t) + ', selected arm:' + str(arm_id) \
                + ', regret: ' + str(self.regret)
            print(s)
        self.t += 1
        return -reward

    def experiment(self, folder_name):
        self.initialize()
        fmin(self.one_iteration, self.hyper_params,
             algo=tpe.suggest, max_evals=self.play_num,
             trials=self.trials)
        # plot log
        if self.show_log:
            plt.plot(self.regrets)
            plt.grid()
            plt.xscale('log')
            plt.xlim(0, self.play_num)
            plt.ylim(0, 100)
            plt.show()
        # save log
        if self.save_log:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            regrets = pd.DataFrame(self.regrets)
            regrets.to_csv(folder_name + '/regret.csv')
            scores = self.trials.losses()
            log = {'fval': []}
            ts = self.trials.trials
            keys = ts[0]['misc']['vals'].keys()
            for key in keys:
                log[key] = []
            for i in range(len(ts)):
                log['fval'].append(scores[i])
                for key in keys:
                    if len(ts[i]['misc']['vals'][key]) > 0:
                        log[key].append(ts[i]['misc']['vals'][key][0])
                    else:
                        log[key].append(None)
            df = pd.DataFrame(log)
            df.index.name = '#index'
            df.to_csv(folder_name + '/log.csv', sep=',')
