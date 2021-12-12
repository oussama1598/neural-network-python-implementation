from typing import Type

import numpy as np

from src.initializers import Initializer
from src.layers.layer import Layer


class LinearLayer(Layer):
    def __init__(self, n: int, m: int, initializer: Type[Initializer]):
        self.n = n
        self.m = m

        self.w = initializer.initialize(m, n)
        self.b = initializer.initialize(m, 1)

    def feed_forward(self, x: np.array):
        self.x = x

        return np.dot(self.w, x) + self.b

    def back_propagate(self, nabla_y, lr=0.01):
        nabla_w = np.dot(nabla_y, self.x.T)
        self.w = self.w - lr * nabla_w
        self.b = self.b - lr * nabla_y
        return np.dot(self.w.T, nabla_y)
