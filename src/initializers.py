import numpy as np


class Initializer:
    @staticmethod
    def initialize(n, m):
        raise NotImplemented()


class UniformInitializer(Initializer):
    @staticmethod
    def initialize(n, m):
        return np.random.uniform(n, m)


class NormalInitializer(Initializer):
    @staticmethod
    def initialize(n, m):
        return np.random.randn(n, m)


class OnesInitializer(Initializer):
    @staticmethod
    def initialize(n, m):
        return np.ones((n, m))


class ZerosInitializer(Initializer):
    @staticmethod
    def initialize(n, m):
        return np.zeros((n, m))


class XavierInitializer(Initializer):
    @staticmethod
    def initialize(n, m):
        alpha = np.sqrt(6 / (n + m))

        return np.random.uniform(-alpha, alpha, size=(n, m))


class HeInitializer(Initializer):
    @staticmethod
    def initialize(n, m):
        alpha = np.sqrt(6 / n)

        return np.random.uniform(-alpha, alpha, size=(n, m))
