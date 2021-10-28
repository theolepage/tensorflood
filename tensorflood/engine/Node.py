import numpy as np

from .utils import topological_sort

class Node:
    def __init__(self):
        pass

    def backward_(self, grad=None):
        pass

    def backward(self):
        order = topological_sort(self)

        order[-1].grad = np.array(1)

        for node in reversed(order):
            grads = node.backward_(node.grad)

            if not hasattr(node, 'input_nodes'):
                continue

            for input_node, grad in zip(node.input_nodes, grads):
                if len(node.input_nodes) == 1: grad = grads
                input_node.grad += grad