import numpy as np


def tanh(z):
    return np.tanh(z)


def d_tanh(z):
    return 1 - np.tanh(z) ** 2
