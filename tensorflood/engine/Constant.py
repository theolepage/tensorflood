from .GlobalState import GlobalState
from .Node import Node

import numpy as np

class Constant(Node):
    count = 0

    def __init__(self, data, name=None):
        super().__init__()

        GlobalState.graph.constants.add(self)

        self._data = data if isinstance(data, np.ndarray) else np.array(data)
        self.zero_grad()

        self.name = 'Const/{}'.format(Constant.count) if name is None else name
        
        Constant.count += 1
        
    def __repr__(self):
        if self.data.ndim == 1:
            return 'Constant: {}={}'.format(self.name, self.data)
        return 'Constant: {}'.format(self.name)

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, val):
        raise ValueError('Cannot reassign constant')

    def zero_grad(self):
        self.grad = 0