
import numpy as np

from .GlobalState import GlobalState
from .Node import Node
from .Operator import Operator, OperatorBinder

class Graph():
    def __init__(self):
        GlobalState.graph = self

        self.operators = set()
        self.constants = set()
        self.variables = set()

        OperatorBinder.bind()

    def reset_counts(self, root):
        if hasattr(root, 'count'):
            root.count = 0
        else:
            for child in root.__subclasses__():
                self.reset_counts(child)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try: del GlobalState.graph
        except: pass
        self.reset_counts(Node)

    def reset(self):
        self.reset_counts(Node)
        self.operators = set()
        self.constants = set()
        self.variables = set()