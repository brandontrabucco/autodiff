"""Author: Brandon Trabucco, Copyright 2019
Implements a jacobian gradient calculator using numpy.
"""


import numpy as np
from autodiff.tensor import Tensor
from autodiff.jacobian import jacobian


def sigmoid(x):

    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":


    w1 = Tensor(np.random.normal(0, 1, [10, 3]))
    b1 = Tensor(np.random.normal(0, 1, [3]))
    
    w2 = Tensor(np.random.normal(0, 1, [3, 2]))
    b2 = Tensor(np.random.normal(0, 1, [2]))
    
    w3 = Tensor(np.random.normal(0, 1, [2, 1]))
    b3 = Tensor(np.random.normal(0, 1, [1]))

    x = Tensor(np.random.normal(0, 1, [5, 2]))
    h1 = sigmoid(x.reshape([10]).dot(w1) + b1)
    h2 = sigmoid(h1.dot(w2) + b2)
    h3 = sigmoid(h2.dot(w3) + b3)

    t4 = jacobian(h3, w1)
    t5 = jacobian(t4, w1)
    t6 = jacobian(t5, w1)
    print(t6.shape)
