import numpy as np


def ReLU(z):
    return np.maximum(z, 0)


def d_ReLU(z):
    return np.array(z > 0, dtype='float32')
