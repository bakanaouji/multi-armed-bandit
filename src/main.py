from arm.arm import NormalDistributionArm
from bandit_algorithm.bandit_core import BanditCore
from bandit_algorithm.thompson_sampling.gaussian_prior import \
    ThompsonSamplingGaussianPrior


def main():
    arms = [NormalDistributionArm(1.0, 3.0),
            NormalDistributionArm(0.0, 0.3)]
    algorithm = ThompsonSamplingGaussianPrior(len(arms))
    core = BanditCore(arms, algorithm)
    core.experiment()


if __name__ == '__main__':
    main()
