import numpy as np


class XavierInitializer:
    def initialize(self, n, m):
        alpha = np.sqrt(6 / (n + m))
        w = np.random.uniform(-alpha, alpha, size=(n, m))
        return w
