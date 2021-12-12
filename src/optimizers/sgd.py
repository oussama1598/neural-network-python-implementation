import numpy as np

from src.losses.mean_squared_error import d_mean_squared_error
from src.neural_network import NeuralNetwork
from src.optimizers.optimizer import Optimizer


class SGDOptimizer(Optimizer):
    def run(self, model: NeuralNetwork, training_data_x: np.array, training_data_y: np.array):
        for x, y in zip(training_data_x, training_data_y):
            output = model.feed_forward(x)

            grad = d_mean_squared_error(y, output)

            for layer in reversed(model.layers):
                grad = layer.back_propagate(grad, self.learning_rate)
