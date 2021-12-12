import numpy as np


def mean_squared_error(y_hat, y):
    return np.mean(np.power(y_hat - y, 2))


def d_mean_squared_error(y_hat, y):
    return 2 * (y - y_hat) / np.size(y_hat)
