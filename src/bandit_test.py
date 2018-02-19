import matplotlib.pyplot as plt
import pandas as pd

from arm.arm import NormalDistributionArm
from bandit_algorithm.thompson_sampling.gaussian_prior import ThompsonSamplingGaussianPrior
from utils.random import *


def main():
    arms = [NormalDistributionArm(1.0, 3.0), NormalDistributionArm(0.0, 0.3)]
    N = len(arms)
    algorithm = ThompsonSamplingGaussianPrior(N)

    regret = 0.0
    regrets = []
    thetas = []

    # main loop
    for t in range(20000):
        # estimate mean of each arm
        theta = algorithm.estimate_mean()
        # select arm
        arm_index = np.argmax(theta)
        # play arm and observe reward
        reward = arms[arm_index].play()
        # update parameter of bandit algorithm
        algorithm.update_param(arm_index, reward)
        # update regret
        regret += arms[0].mean - arms[arm_index].mean
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

    regrets = pd.DataFrame(regrets)
    thetas = pd.DataFrame(thetas)
    regrets.to_csv('../data/regret.csv')
    thetas.to_csv('../data/theta.csv')


if __name__ == '__main__':
    main()
