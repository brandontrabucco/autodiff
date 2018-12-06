"""Author: Brandon Trabucco, Copyright 2019
Implements a jacobian gradient calculator using numpy.
"""


import numpy as np
import autodiff
import autodiff.graph


DEFAULT_GRAPH = autodiff.graph.Graph()


class Identifier(object):

    def __init__(self):
        """Class for managing Tensor identifiers."""
        self.x = 0

    def get(self):
        """Get the next available tensor identifier.
        Returns: an integer."""
        y = self.x
        self.x += 1
        return y


ID_MANAGER = Identifier()


class Tensor(np.ndarray):

    def __new__(cls, input_array, graph=None):
        """Creates a tensor object.
        Args: input_array: numpy array to store the tensor data.
              graph: a container to store tensors."""
        prototype = np.asarray(input_array).view(cls)
        prototype.graph = (graph if graph is not None else DEFAULT_GRAPH)
        return prototype

    def __array_finalize__(self, prototype):
        """Creates a tensor object.
        Args: input_array: numpy array to store the tensor data.
              graph: a container to store tensors."""
        self.graph = getattr(prototype, 'graph', DEFAULT_GRAPH)
        self.id = ID_MANAGER.get()

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """Applies a numpy ufunc operation to the tensor and 
        adds the operation to the current graph."""
        in_args = [x.view(np.ndarray) if isinstance(x, Tensor) else x for x in args]
        if kwargs.pop('out', None) is not None or ufunc.nout != 1:
            raise NotImplementedError()
        result = Tensor(np.asarray(super(Tensor, self).__array_ufunc__(ufunc, method, 
            *in_args, **kwargs)), graph=self.graph)
        self.graph.register_operation(ufunc.__name__, result, args, kwargs)
        return result

    def dot(self, b):
        """Computes the dot product of this tensor with b.
        Args: b: a Tensor object.
        Returns: the result of a dot product."""
        return autodiff.dot(self, b)

    def reshape(self, shape):
        """Reshapes the tensor into shape.
        Args: shape: a list of ints.
        Returns: a tensor object."""
        return autodiff.reshape(self, shape)
