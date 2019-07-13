from __future__ import absolute_import
from .util import as_variable
from .model import Model, Param
from torch.autograd import Variable
import torch as th
from numpy.polynomial.hermite import hermval
import numpy as np
import bpflow as bf

TensorType = th.DoubleTensor


class MeanFunction(Model):
    """
    The base mean function class.
    To implement a mean function, write the __call__ method. This takes a
    tensor X and returns a tensor m(X). In accordance with the GPflow/GPtorch
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.
    MeanFunction classes can have parameters, see the Linear class for an
    example.
    """
    def __init__(self):
        super(MeanFunction, self).__init__()

    def __call__(self, X):
        raise NotImplementedError("Implement the __call__\
                                  method for this mean function")

    def __add__(self, other):
        return Additive(self, other)

    def __mul__(self, other):
        return Product(self, other)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n\n'
        for name, param in self._parameters.items():
            tmpstr = tmpstr + name + str(param.data) + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr



class Identity(Linear):
    """
    y_i = x_i
    """
    def __init__(self, input_dim=None):
        Linear.__init__(self)
        self.input_dim = input_dim

    def __call__(self, X):
        return X



class Linear(MeanFunction):
  #yi =AXi +b format
        
         def __init__(self, A=None, b=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        A = np.ones((1, 1)) if A is None else A
        b = np.zeros(1) if b is None else b
        MeanFunction.__init__(self)
        self.A = Parameter(np.atleast_2d(A), dtype=settings.float_type)
        self.b = Parameter(b, dtype=settings.float_type)

    def __call__(self, X):
        return variable(bf.tensordot(X, self.A, [[-1], [0]]) + self.b)  #tensordot Alt
    



class Zero(MeanFunction):
    def __call__(self, X):
        # return Variable(th.zeros(len(X), 1), requires_grad=False)
        # print(th.zeros(10, 1))
        return Variable(th.zeros(X.size(0), 1).type(TensorType),
                        requires_grad=False)


class Constant(MeanFunction):
    """
    Just a constant
    """
    def __init__(self):
        super(Constant, self).__init__()

    def __call__(self, x):
        if isinstance(x, Variable):
            return Variable(th.ones(x.size(0), 1).type(TensorType))
        else:
            raise NotImplementedError