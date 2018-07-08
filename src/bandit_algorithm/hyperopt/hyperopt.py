from hyperopt import fmin, tpe, Trials


class HyperOpt(object):
    def __init__(self, arms, hyper_params):
        self.arms = arms
        self.N = len(arms)
        self.hyper_params = hyper_params
        self.trials = Trials()
        self.regret = 0.0
        self.t = 0

    def one_iteration(self, params):
        # select arm
        arm_id = params['arm']
        # play arm and observe reward
        reward = self.arms[arm_id].play()
        # update regret
        self.regret += self.arms[0].mean - self.arms[arm_id].mean
        # output
        if self.t % 5000 == 0:
            s = 'iteration: ' + str(self.t) + ', selected arm:' + str(arm_id) \
                + ', regret: ' + str(self.regret)
            print(s)
        self.t += 1
        return -reward

    def optimize(self):
        fmin(self.one_iteration, self.hyper_params,
             algo=tpe.suggest, max_evals=20000,
             trials=self.trials)
