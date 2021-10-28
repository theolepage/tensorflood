from .GlobalState import GlobalState
from .Node import Node

import numpy as np

class Variable(Node):
    count = 0

    def __init__(self, data, name=None):
        super().__init__()

        GlobalState.graph.variables.add(self)

        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.zero_grad()

        self.name = 'Var/{}'.format(Variable.count) if name is None else name
        
        Variable.count += 1

    def __repr__(self):
        if self.data.ndim == 1:
            return 'Variable: {}={}'.format(self.name, self.data)
        return 'Variable: {}'.format(self.name)

    def zero_grad(self):
        self.grad = 0