"""The AQUA reverse-mode automatic differentiation engine.

Public surface
--------------
:class:`~aqua.engine.tensor.Tensor`
    The graph node type.
:class:`~aqua.engine.tensor.Parameter`
    A tensor that is meant to be optimised.
:mod:`aqua.engine.ops`
    Every differentiable operation, plus the masking primitive
    :func:`~aqua.engine.ops.applyMask`.
"""

from aqua.engine.ops import (
    applyMask,
    crossEntropyFromLogits,
    divide,
    exp,
    log,
    logSoftmax,
    matmul,
    mean,
    multiply,
    negate,
    reduceGradToShape,
    relu,
    reshape,
    sigmoid,
    softmax,
    subtract,
    sum,
    tanh,
)
from aqua.engine.tensor import Parameter, Tensor

__all__ = [
    "Tensor",
    "Parameter",
    "add",
    "applyMask",
    "crossEntropyFromLogits",
    "divide",
    "exp",
    "log",
    "logSoftmax",
    "matmul",
    "mean",
    "multiply",
    "negate",
    "reduceGradToShape",
    "relu",
    "reshape",
    "sigmoid",
    "softmax",
    "subtract",
    "sum",
    "tanh",
]
