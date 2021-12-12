import numpy as np


class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def run(self, model, training_data_x: np.array, training_data_y: np.array):
        raise NotImplemented()
