import numpy as np

class Optimizer:
    
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad