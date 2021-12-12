from typing import List

import pandas as pd
from numpy import ndarray

from src.layers.layer import Layer
from src.losses.mean_squared_error import *
from src.optimizers.optimizer import Optimizer


class NeuralNetwork:
    def __init__(self, *args):
        self.layers: List[Layer] = args

    def feed_forward(self, x: np.array) -> np.array:
        output = x
        for layer in self.layers:
            output = layer.feed_forward(output)

        return output

    def calculate_loss(self, x_data: np.array, y_data: np.array) -> ndarray:
        return np.mean([
            mean_squared_error(y, self.feed_forward(x))
            for x, y in zip(x_data, y_data)
        ])

    def optimize(self,
                 training_data_x: pd.DataFrame, training_data_y: pd.DataFrame,
                 validation_data_x: pd.DataFrame, validation_data_y: pd.DataFrame,
                 optimizer: Optimizer, epochs: int = 1000, verbose: bool = False):
        training_data_x: np.array = training_data_x.values
        training_data_y: np.array = training_data_y.values

        validation_data_x: np.array = validation_data_x.values
        validation_data_y: np.array = validation_data_y.values

        t_loss_history = []
        v_loss_history = []

        for epoch in range(epochs):
            optimizer.run(self, training_data_x, training_data_y)

            t_loss_history.append(self.calculate_loss(training_data_x, training_data_y))
            v_loss_history.append(self.calculate_loss(validation_data_x, validation_data_y))

            if verbose:
                print(f'Epoch {epoch + 1}/{epochs} ::: t_loss = {t_loss_history[-1]}, v_loss = {v_loss_history[-1]}')

        return t_loss_history, v_loss_history

    def predict(self, data_x: pd.DataFrame):
        return [
            self.feed_forward(x)
            for x in data_x.values
        ]
