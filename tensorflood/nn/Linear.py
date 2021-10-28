import numpy as np

from tensorflood.engine import Variable

class Linear:
    
    def __init__(self, name, in_dim, out_dim, last_layer=False):
        self.name = name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.last_layer = last_layer

        weights = np.random.uniform(-1, 1, size=(in_dim, out_dim))
        bias = np.zeros((out_dim))

        self.W = Variable(weights, name=self.name + '_W')
        self.b = Variable(bias, name=self.name + '_b')

    def __call__(self, X):
        Z = X @ self.W + self.b
        return Z.relu() if not self.last_layer else Z.sigmoid()
        
    def __repr__(self):
        return self.name