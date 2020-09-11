import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# set random func to repeat random value each time
np.random.seed(0)


# samples
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]


X, y = spiral_data(100, 3)

#hiden layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # np.random.randn - array with normalize data from Gause

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases



class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)

activation1 = Activation_Relu()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)