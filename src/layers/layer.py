import numpy as np


class Layer:
    def feed_forward(self, x: np.array):
        raise NotImplemented()

    def back_propagate(self, nabla_y, lr=0.01):
        raise NotImplemented()
