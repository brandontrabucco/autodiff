"""Author: Brandon Trabucco, Copyright 2019
Implements a jacobian gradient calculator using numpy.
"""


import numpy as np
import autodiff.tensor


class Operation(object):

    def __init__(self, name, result, args, kwargs):
        """Creates an operation node for the graph.
        Args: name: a string name for the operation.
              result: the Tensor resulting from the operation.
              args: a list of Tensor inputs.
              kwargs: a dict of keyword arguments."""
        self.name = name
        self.result = result
        self.args = args
        self.kwargs = kwargs


class Graph(object):

    def __init__(self):
        """Create a differentiable graph of operations.
        """
        self.operation_map = {}

    def register_operation(self, name, result, args, kwargs):
        """Adds a new tensor to the graph with a new unique identifier.
        Args: name: the name of the operation.
              result: a Tensor object resulting from the operation.
              args: a list of Tensor objects input to the operation.
              kwargs: a dict of non Tensor keyword arguments."""
        if not isinstance(result, autodiff.tensor.Tensor):
            result = autodiff.tensor.Tensor(result, graph=self)
        args = [x if isinstance(x, autodiff.tensor.Tensor) 
            else autodiff.tensor.Tensor(x, graph=self) for x in args]
        self.operation_map[result.id] = Operation(name, result, args, kwargs)

    def lookup_operation(self, result):
        """Fetches the operation and inputs that led to this result.
        Args: result: a Tensor object.
        Returns: a matching operation object."""
        if (not isinstance(result, autodiff.tensor.Tensor) or 
                result.id not in self.operation_map):
            return None
        return self.operation_map[result.id]
