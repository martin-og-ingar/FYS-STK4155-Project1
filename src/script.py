import numpy as np


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def Z():
    np.random.seed(42)
    degree = range(18)
    n = 1000
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    noise = 0.1 * np.random.normal(0, 1, n)
    Z = FrankeFunction(x, y) + noise

    return Z
