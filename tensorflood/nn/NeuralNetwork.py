import numpy as np

from tensorflood.nn import Linear

class NeuralNetwork:

    def __init__(self, input_dim, units):
        sizes = [input_dim] + units
        self.layers = [
            Linear(
                'Linear_{}'.format(i),
                sizes[i],
                sizes[i + 1],
                last_layer=(i == len(units) - 1))
            for i in range(len(units))
        ]

    def __call__(self, X):
        Z = X
        for layer in self.layers:
            Z = layer(Z)
        return Z

    def get_params(self):
        params = []
        for layer in self.layers:
            params.append(layer.W)
            params.append(layer.b)
        return params