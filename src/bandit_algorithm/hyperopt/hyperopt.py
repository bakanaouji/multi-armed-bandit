import os
import pandas as pd

from hyperopt import fmin, tpe, Trials


class HyperOpt(object):
    def __init__(self, arms, hyper_params, save_log=False):
        self.arms = arms
        self.N = len(arms)
        self.hyper_params = hyper_params
        self.save_log = save_log
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
        if self.t % 5000 == 0:
            s = 'iteration: ' + str(self.t) + ', selected arm:' + str(arm_id) \
                + ', regret: ' + str(self.regret)
            print(s)
        self.t += 1
        return -reward

    def experiment(self, folder_name):
        fmin(self.one_iteration, self.hyper_params,
             algo=tpe.suggest, max_evals=20000,
             trials=self.trials)
        # save log
        if self.save_log:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            regrets = pd.DataFrame(self.regrets)
            regrets.to_csv(folder_name + '/regret.csv')
