"""Author: Brandon Trabucco, Copyright 2019
Implements a jacobian gradient calculator using numpy.
"""


import numpy as np
import autodiff.tensor


def dot(a, b):
    """Computes the differentiable dot product.
    Args: a: a tensor object.
          b: a tensor object.
    Returns: a tensor object."""
    result = np.tensordot(a.view(np.ndarray), b.view(np.ndarray), axes=1)
    result = autodiff.tensor.Tensor(result, graph=a.graph)
    a.graph.register_operation("dot", result, [a, b], {"axes": 1})
    return result


def reshape(a, shape):
    """Reshapes the tensor into shape.
    Args: a: a tensor object.
          shape: a list of ints.
    Returns: a tensor object."""
    result = np.reshape(a.view(np.ndarray), shape)
    result = autodiff.tensor.Tensor(result, graph=a.graph)
    a.graph.register_operation("reshape", result, [a], {"shape": shape})
    return result