import numpy as np

from src.layers.layer import Layer


class ActivationLayer(Layer):
    def __init__(self, f, df):
        self.f = f
        self.df = df

    def feed_forward(self, x: np.array):
        self.x = x

        return self.f(x)

    def back_propagate(self, nabla_y, lr=0.01):
        # Nothing to update here, just pass the gradient to the previous layer
        return np.multiply(nabla_y, self.df(self.x))
