"""Author: Brandon Trabucco, Copyright 2019
Implements a jacobian gradient calculator using numpy.
"""


import numpy as np
from autodiff.tensor import Tensor
from autodiff.tensor import DEFAULT_GRAPH
from autodiff.derivative import derivative


def jacobian(x, y, graph=None):
    """Computes the jacobian of x with respect to y in the graph.
    Args: x: a tensor object.
          y: a tensor object.
          graph: a Graph object containing x and y.
    Returns: the jacobian of x with respect to y."""
    graph = (graph if graph is not None else DEFAULT_GRAPH)
    if x is y:
        if x.ndim > 0:
            return Tensor(np.eye(x.size).reshape(x.shape + x.shape), graph=graph)
        else:
            return Tensor(1.0, graph=graph)
    operation = graph.lookup_operation(x)
    if operation is None:
        if x.ndim > 0:
            return Tensor(np.zeros(x.shape + y.shape), graph=graph)
        else:
            return Tensor(0.0, graph=graph)
    result = Tensor(np.array([0.0]), graph=graph)
    for i, z in enumerate(operation.args):
        ddx = derivative(jacobian(z, y), operation, i, graph=graph)
        while result.ndim < ddx.ndim:
            result = np.repeat(result[..., np.newaxis], ddx.shape[result.ndim], axis=result.ndim)
        while ddx.ndim < result.ndim:
            ddx = np.repeat(ddx[..., np.newaxis], result.shape[ddx.ndim], axis=ddx.ndim)
        result = result + ddx
    return result