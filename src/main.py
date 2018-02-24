import argparse

from arm.arm import NormalDistributionArm
from bandit_algorithm.bandit_core import BanditCore
from bandit_algorithm.thompson_sampling.gaussian_prior import \
    ThompsonSamplingGaussianPrior
from bandit_algorithm.thompson_sampling.gaussian_sicq_prior import \
    ThompsonSamplingGaussianSicqPrior
from utils.data_processing import calc_mean_log


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
    # algorithm = ThompsonSamplingGaussianPrior(len(arms))
    algorithm = ThompsonSamplingGaussianSicqPrior(len(arms))
    save_path_root = '../data/' + arms[0].__class__.__name__ \
                     + '/' + algorithm.__class__.__name__
    core = BanditCore(arms, algorithm, args)

    # run experiment
    print('----------Run Exp----------')
    for i in range(args.exp_num):
        print('Run Exp' + str(i))

        # define bandit algorithm
        save_path = save_path_root + '/Exp' + str(i)
        core.experiment(save_path)
        print('Finish Exp' + str(i))
        print('')

    # calculate mean values of log
    print('----------Calc Mean of Log----------')
    calc_mean_log(save_path_root, 'regret')


if __name__ == '__main__':
    main()
