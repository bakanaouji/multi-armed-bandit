import argparse

from arm.arm import NormalDistributionArm
from bandit_algorithm.bandit_core import BanditCore
from bandit_algorithm.thompson_sampling.gaussian_prior import \
    ThompsonSamplingGaussianPrior


def main():
    parser = argparse.ArgumentParser(description='Bandit Experiment')

    # setting of experiment
    parser.add_argument('--exp_num', type=int, default=1,
                        help='Number of experiments')
    parser.add_argument('--show_log', action='store_true',
                        help='Whether to show log')
    parser.set_defaults(test=False)

    args = parser.parse_args()

    # define arms
    arms = [NormalDistributionArm(1.0, 3.0),
            NormalDistributionArm(0.0, 0.3)]

    # define bandit algorithm
    algorithm = ThompsonSamplingGaussianPrior(len(arms))

    # run experiment
    for i in range(args.exp_num):
        print("Run Exp" + str(i))
        save_path = '../data/' + arms[0].__class__.__name__ \
                    + "/" + algorithm.__class__.__name__ \
                    + "/Exp" + str(i)
        core = BanditCore(arms, algorithm, args, save_path)
        core.experiment()
        print("Finish Exp" + str(i))
        print("")


if __name__ == '__main__':
    main()
