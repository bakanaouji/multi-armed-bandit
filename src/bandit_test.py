import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.random import *


def main():
    arms = [{"mean": 1.0, "sigma": 3.0}, {"mean": 0.0, "sigma": 0.3}]
    N = len(arms)
    k = np.zeros(N)
    mu = np.zeros(N)
    regret = 0.0
    regrets = []
    thetas = []
    for t in range(20000):
        # For each arm i=1,...,N, sample random value from normal distribution
        theta = [np.random.normal(mu[i], 1.0 / (k[i] + 1.0)) for i in range(N)]
        # select arm
        arm_index = np.argmax(theta)
        # play arm and observe reward
        reward = np.random.normal(arms[arm_index]["mean"],
                                  arms[arm_index]["sigma"])
        # update parameter of normal distribution
        mu[arm_index] = (mu[arm_index] * (k[arm_index] + 1.0) + reward) /\
                        (k[arm_index] + 2.0)
        k[arm_index] += 1
        # update regret
        regret += arms[0]["mean"] - arms[arm_index]["mean"]
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
