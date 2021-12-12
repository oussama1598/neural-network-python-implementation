import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

from src.activations import ReLU, d_ReLU, sigmoid, d_sigmoid
from src.initializers import XavierInitializer
from src.layers.activation_layer import ActivationLayer
from src.layers.linear_layer import LinearLayer
from src.neural_network import NeuralNetwork
from src.optimizers.sgd import SGDOptimizer


def main():
    # Dataset loading and preprocessing
    x, y = load_iris(return_X_y=True, as_frame=True)

    x = shuffle(x)

    x_scaled = pd.DataFrame(StandardScaler().fit_transform(x))
    y_scaled = pd.DataFrame(MinMaxScaler().fit_transform(y.to_frame()))

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2)

    model = NeuralNetwork(
        LinearLayer(4, 4, XavierInitializer),
        ActivationLayer(ReLU, d_ReLU),

        LinearLayer(4, 1, XavierInitializer),
        ActivationLayer(sigmoid, d_sigmoid)
    )

    t_loss, v_loss = model.optimize(
        x_train, y_train, x_test, y_test, epochs=100,
        optimizer=SGDOptimizer(learning_rate=.01),
        verbose=True)

    # # Plot the loss history
    x1 = [i for i in range(len(t_loss))]
    y1 = t_loss
    y2 = v_loss
    plt.plot(x1, y1, x1, y2)
    plt.show()


if __name__ == "__main__":
    main()
