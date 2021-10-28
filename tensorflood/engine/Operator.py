import numpy as np

from .GlobalState import GlobalState
from .Node import Node
from .Constant import Constant

def adapt_grad_shape(grad, node_data):
    # Considering grad has n more dimensions than node_data,
    # remove the first n dimensions of grad by summing values.
    # This operation does not impact gradient.
    for _ in range(grad.ndim - node_data.ndim): 
        grad = grad.sum(axis=0)
    return grad


class Operator(Node):

    def __init__(self, input_nodes, name='Operator'):
        super().__init__()

        GlobalState.graph.operators.add(self)

        self.name = name

        self.input_nodes = input_nodes
        self.input_data = [input_node.data for input_node in self.input_nodes]
        
        self.data = self.forward(*self.input_data)

        self.zero_grad()
    
    def __repr__(self):
        return 'Operator: {}'.format(self.name)

    def zero_grad(self):
        self.grad = 0


class add(Operator):
    count = 0

    def __init__(self, input_nodes, name=None):
        super().__init__(input_nodes, name)

        self.name = 'add/{}'.format(add.count) if name is None else name

        add.count += 1
        
    def forward(self, a, b):
        return a + b
    
    def backward_(self, grad):
        grad_a = adapt_grad_shape(grad, self.input_nodes[0].data)
        grad_b = adapt_grad_shape(grad, self.input_nodes[1].data)
        return grad_a, grad_b


class multiply(Operator):
    count = 0

    def __init__(self, input_nodes, name=None):
        super().__init__(input_nodes, name)

        self.name = 'mul/{}'.format(multiply.count) if name is None else name

        multiply.count += 1
        
    def forward(self, a, b):
        return a * b
    
    def backward_(self, grad):
        grad_a = adapt_grad_shape(grad, self.input_nodes[0].data)
        grad_b = adapt_grad_shape(grad, self.input_nodes[1].data)
        a, b = self.input_data
        return grad_b * b, grad_a * a


class divide(Operator):
    count = 0

    def __init__(self, input_nodes, name=None):
        super().__init__(input_nodes, name)

        self.name = 'div/{}'.format(divide.count) if name is None else name

        divide.count += 1
   
    def forward(self, a, b):
        return a / b
    
    def backward_(self, grad):
        grad_a = adapt_grad_shape(grad, self.input_nodes[0].data)
        grad_b = adapt_grad_shape(grad, self.input_nodes[1].data)
        a, b = self.input_data
        return grad_a / b, grad_b * -a / (b ** 2)
    
    
class power(Operator):
    count = 0

    def __init__(self, input_nodes, name=None):
        super().__init__(input_nodes, name)

        self.name = 'pow/{}'.format(power.count) if name is None else name
        
        power.count += 1
   
    def forward(self, a, b):
        return a ** b
    
    def backward_(self, grad):
        a, b = self.input_data
        return grad * b * (a ** (b - 1)), grad * np.log(a) * (a ** b)


class matmul(Operator):
    count = 0

    def __init__(self, input_nodes, name=None):
        super().__init__(input_nodes, name)

        self.name = 'matmul/{}'.format(matmul.count) if name is None else name
        
        matmul.count += 1
        
    def forward(self, a, b):
        return a @ b
    
    def backward_(self, grad):
        a, b = self.input_data
        return grad @ b.T, a.T @ grad


class neg(Operator):
    count = 0

    def __init__(self, input_nodes, name=None):
        super().__init__(input_nodes, name)

        self.name = 'neg/{}'.format(neg.count) if name is None else name
        
        neg.count += 1
        
    def forward(self, a):
        return -a
    
    def backward_(self, grad):
        a = self.input_data[0]
        return -grad * np.ones_like(a)


class log(Operator):
    count = 0

    def __init__(self, input_nodes, name=None):
        super().__init__(input_nodes, name)

        self.name = 'log/{}'.format(log.count) if name is None else name
        
        log.count += 1
        
    def forward(self, a):
        return np.log(a)
    
    def backward_(self, grad):
        a = self.input_data[0]
        return 1/a * grad


class sum(Operator):
    count = 0

    def __init__(self, input_nodes, name=None):
        super().__init__(input_nodes, name)

        self.name = 'sum/{}'.format(sum.count) if name is None else name
        
        sum.count += 1
        
    def forward(self, a):
        return np.sum(a)
    
    def backward_(self, grad):
        a = self.input_data[0]
        return grad * np.ones_like(a)


class relu(Operator):
    count = 0

    def __init__(self, input_nodes, name=None):
        super().__init__(input_nodes, name)

        self.name = 'relu/{}'.format(relu.count) if name is None else name
        
        relu.count += 1
        
    def forward(self, a):
        return a * (a > 0)
    
    def backward_(self, grad):
        a = self.input_data[0]
        return grad * (a > 0)


class sigmoid(Operator):
    count = 0

    def __init__(self, input_nodes, name=None):
        super().__init__(input_nodes, name)

        self.name = 'sigmoid/{}'.format(sigmoid.count) if name is None else name
        
        sigmoid.count += 1
        
    def forward(self, a):
        return 1. / (1. + np.exp(-a))
    
    def backward_(self, grad):
        a = self.input_data[0]
        tmp = 1. / (1. + np.exp(-a))
        return grad * tmp * (1 - tmp)


class OperatorBinder:

    @staticmethod
    def bind_(method, lhs, rhs):
        if isinstance(rhs, Node):
            return method([lhs, rhs])
        if isinstance(rhs, float) or isinstance(rhs, int):
            return method([lhs, Constant(rhs)])
        raise TypeError('Incompatible types')

    @staticmethod
    def bind():
        Node.__add__ = lambda self, other: OperatorBinder.bind_(add, self, other)
        Node.__radd__ = lambda self, other: OperatorBinder.bind_(add, self, other)
        Node.__sub__ = lambda self, other: OperatorBinder.bind_(add, self, -other)
        Node.__rsub__ = lambda self, other: OperatorBinder.bind_(add, -self, other)
        Node.__mul__ = lambda self, other: OperatorBinder.bind_(multiply, self, other)
        Node.__truediv__ = lambda self, other: OperatorBinder.bind_(divide, self, other)
        Node.__pow__ = lambda self, other: OperatorBinder.bind_(power, self, other)
        Node.__matmul__ = lambda self, other: OperatorBinder.bind_(matmul, self, other)

        Node.__neg__ = lambda self: neg([self])
        Node.log = lambda self: log([self])
        Node.sum = lambda self: sum([self])

        Node.relu = lambda self: relu([self])
        Node.sigmoid = lambda self: sigmoid([self])