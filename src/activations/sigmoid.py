import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))
