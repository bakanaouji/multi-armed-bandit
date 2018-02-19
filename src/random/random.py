import numpy as np


def standard_gamma(shape, scale=1.0):
    # calc n
    if shape <= 0.4:
        n = 1.0 / shape
    elif shape <= 4.0:
        n = 1.0 / shape + (shape - 0.4) / (3.6 * shape)
    else:
        n = 1.0 / np.math.sqrt(shape)

    # calc b
    b1 = shape - 1.0 / n
    b2 = shape + 1.0 / n

    # calc c
    if shape <= 0.4:
        c1 = 0.0
    else:
        c1 = b1 * (np.math.log(b1) - 1.0) / 2.0
    c2 = b2 * (np.math.log(b2) - 1.0) / 2.0

    while True:
        v1 = np.random.rand()
        v2 = np.random.rand()
        w1 = c1 + np.math.log(v1)
        w2 = c2 + np.math.log(v2)
        y = n * (b1 * w2 - b2 * w1)
        if y >= 0 and np.math.log(y) >= n * (w2 - w1):
            break
    return np.math.exp(n * (w2 - w1)) * scale


def chisquare(df):
    return standard_gamma(df / 2.0, 2.0)
