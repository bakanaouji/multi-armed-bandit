import matplotlib.pyplot as plt

from utils.random import *


def main():
    samples = [scaled_inverse_chi_squared(10, 1.0) for _ in range(5000)]
    # show histogram
    plt.hist(samples, bins=100, normed=True, range=(0.0, 5.0))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
