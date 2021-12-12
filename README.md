# Deep Neural Network Implementation From Scratch (Numpy only)
A very basic implementation of deep neural networks, written in python using only numpy

## How it looks
Building the neural network
```python
model = NeuralNetwork(
      LinearLayer(64, 32),
      ActivationLayer(ReLU, d_ReLU),
      
      LinearLayer(32, 32),
      ActivationLayer(ReLU, d_ReLU),
      
      
      LinearLayer(32, 10),
      ActivationLayer(sigmoid, d_sigmoid)
  )
```
Training it
```python
  t_loss, v_loss = model.optimize(x_train, y_train, x_test, y_test, lr = .01, epochs = 128, verbose = True)
```
Using it
```python
    y = model.predict(x)
```