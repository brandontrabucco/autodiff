"""Author: Brandon Trabucco, Copyright 2019
Implements a jacobian gradient calculator using numpy.
"""


import numpy as np
import autodiff
import autodiff.tensor


def derivative(jacobian, operation, i, graph=None):
    """Computes the dx/dz of the given operation.
    Args: jacobian: the jacobian matrix of z.
          operation: an Operation object.
          i: the position of the current Tensor object in the args.
          graph: a Graph object containing the inputs.
    Returns: the resulting derivative matrix."""

    if operation.name == "add":
        return jacobian

    elif operation.name == "subtract":
        return jacobian * autodiff.tensor.Tensor(1.0 if i == 0 else -1.0, graph=graph)

    elif operation.name == "multiply":
        x = operation.args[(i + 1) % 2]
        while x.ndim < jacobian.ndim:
            x = np.repeat(x[..., np.newaxis], jacobian.shape[x.ndim], axis=x.ndim)
        return jacobian * x

    elif operation.name == "divide" or operation.name =="true_divide":
        x, y = operation.args
        while x.ndim < jacobian.ndim:
            x = np.repeat(x[..., np.newaxis], jacobian.shape[x.ndim], axis=x.ndim)
        while y.ndim < jacobian.ndim:
            y = np.repeat(y[..., np.newaxis], jacobian.shape[y.ndim], axis=y.ndim)
        if i == 0:
            return jacobian * autodiff.tensor.Tensor(1.0, graph=graph) / y
        elif i == 1:
            return jacobian * autodiff.tensor.Tensor(-1.0, graph=graph) * x / y / y

    elif operation.name == "negative":
        return jacobian * autodiff.tensor.Tensor(-1.0, graph=graph)

    elif operation.name == "power":
        x, y = operation.args
        while x.ndim < jacobian.ndim:
            x = np.repeat(x[..., np.newaxis], jacobian.shape[x.ndim], axis=x.ndim)
        while y.ndim < jacobian.ndim:
            y = np.repeat(y[..., np.newaxis], jacobian.shape[y.ndim], axis=y.ndim)
        if i == 0:
            return jacobian * y * np.power(x, y - autodiff.tensor.Tensor(1.0, graph=graph))
        elif i == 1:
            return jacobian * np.log(np.absolute(x)) * np.power(x, y)

    elif operation.name == "absolute":
        x = operation.args[i]
        while x.ndim < jacobian.ndim:
            x = np.repeat(x[..., np.newaxis], jacobian.shape[x.ndim], axis=x.ndim)
        return jacobian * x / np.absolute(x)

    elif operation.name == "exp":
        x = operation.args[i]
        while x.ndim < jacobian.ndim:
            x = np.repeat(x[..., np.newaxis], jacobian.shape[x.ndim], axis=x.ndim)
        return jacobian * np.exp(x)
        
    elif operation.name == "log":
        x = operation.args[i]
        while x.ndim < jacobian.ndim:
            x = np.repeat(x[..., np.newaxis], jacobian.shape[x.ndim], axis=x.ndim)
        return jacobian * autodiff.tensor.Tensor(1.0, graph=graph) / x

    elif operation.name == "sqrt":
        x = operation.args[i]
        while x.ndim < jacobian.ndim:
            x = np.repeat(x[..., np.newaxis], jacobian.shape[x.ndim], axis=x.ndim)
        return jacobian * autodiff.tensor.Tensor(0.5, graph=graph) / np.sqrt(x)

    elif operation.name == "square":
        x = operation.args[i]
        while x.ndim < jacobian.ndim:
            x = np.repeat(x[..., np.newaxis], jacobian.shape[x.ndim], axis=x.ndim)
        return jacobian * autodiff.tensor.Tensor(2.0, graph=graph) * x

    elif operation.name == "dot":
        x, y = operation.args
        if i == 0:
            length1, length2, length3 = len(x.shape), len(jacobian.shape) - len(x.shape), len(y.shape)
            jacobian = jacobian.transpose(list(range(length1, length1 + length2)) + list(range(length1)))
            return jacobian.dot(y).transpose(list(range(length2, 
                length1 + length2 + length3 - 2)) + list(range(length2)))
        elif i == 1:
            return x.dot(jacobian)

    elif operation.name == "reshape":
        shape1 = operation.args[0].shape
        shape2 = operation.kwargs["shape"]
        shape3 = jacobian.shape
        shape2 += shape3[len(shape1):]
        return jacobian.reshape(shape2)

    print("Error: operation {0} is not differentiable.".format(operation.name))
    raise NotImplementedError()