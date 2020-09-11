import numpy as np

# set random func to repeat random value each time
np.random.seed(0)


# samples
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]


#hiden layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        print("init",n_inputs,n_neurons)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        print("init_weights", n_inputs, n_neurons,self.weights)
        self.biases = np.zeros((1, n_neurons))
        print("init_biases", n_inputs, n_neurons,self.biases)
        # np.random.randn - array with normalize data from Gause

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        print("ouput", self.output)


layer1 = Layer_Dense(len(X[0]), 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)