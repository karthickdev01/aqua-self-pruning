"""Differentiable operations for the AQUA autodiff engine.

Every public function in this module follows the same contract:

* it accepts :class:`~aqua.engine.tensor.Tensor` objects or plain NumPy
  scalars/arrays (the latter are promoted to constants),
* it returns a new :class:`~aqua.engine.tensor.Tensor`,
* it installs a local backward rule that *accumulates* into the gradients of
  its inputs.

Gradients w.r.t. broadcast inputs are reduced back to the input shape by
:func:`reduceGradToShape`, which is the single place in the code base where
the NumPy broadcasting rules are translated into gradient rules.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np

from aqua.engine.tensor import Tensor

__all__ = [
    "reduceGradToShape",
    "add",
    "subtract",
    "multiply",
    "divide",
    "negate",
    "matmul",
    "sum",
    "mean",
    "reshape",
    "relu",
    "leakyRelu",
    "tanh",
    "sigmoid",
    "softmax",
    "logSoftmax",
    "crossEntropyFromLogits",
    "applyMask",
    "exp",
    "log",
    "square",
    "maximum",
    "clip",
]

Numeric = Union[int, float, np.ndarray]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _toTensor(value: Union[Tensor, Numeric], requiresGrad: bool = False) -> Tensor:
    """Promote ``value`` to a tensor if it is not one already."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requiresGrad=requiresGrad)


def _accumulate(target: Optional[Tensor], contribution: np.ndarray) -> None:
    """Add ``contribution`` to ``target.grad``, allocating the buffer on demand."""
    if target is None:
        return
    if target.grad is None:
        target.grad = np.array(contribution, dtype=target.data.dtype, copy=True)
    else:
        target.grad = target.grad + contribution


def reduceGradToShape(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``gradient`` back to ``shape`` following NumPy broadcasting rules.

    If ``gradient`` was produced by broadcasting an input of shape ``shape`` up
    to a larger shape, the gradient must be summed over exactly the axes that
    were broadcast.  Concretely:

    * axes that were *added* (leading axes) are summed away entirely,
    * axes where the input had size ``1`` and the output had size ``n > 1`` are
      summed with ``keepdims=True`` so the rank is preserved,
    * axes where both sizes agree are left untouched.

    Examples
    --------
    >>> reduceGradToShape(np.ones((4, 3)), (3,)).tolist()
    [[4.0, 4.0, 4.0]]
    >>> reduceGradToShape(np.ones((4, 3)), (1, 3)).tolist()
    [[4.0, 4.0, 4.0]]
    """
    grad = np.asarray(gradient)
    targetShape = tuple(shape)

    if grad.shape == targetShape:
        return grad

    # Drop the leading axes that were created by broadcasting.
    while grad.ndim > len(targetShape):
        grad = grad.sum(axis=0)

    # Sum over the axes that were stretched from size 1.  The reduction
    # uses keepdims=True so that the axis indices of the *remaining* axes do
    # not shift while the loop is running.
    for axis, size in enumerate(targetShape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    # The gradient may now have a *trailing* axis of size 1 where the target
    # has a larger one.  Broadcast the result to the target shape before
    # reshaping.
    if grad.shape != targetShape:
        grad = np.broadcast_to(grad, targetShape)

    return np.ascontiguousarray(grad).reshape(targetShape)


def _makeNode(
    data: np.ndarray,
    parents: Sequence[Tensor],
    requiresGrad: bool,
) -> Tensor:
    """Create a graph node and wire it to its parents."""
    node = Tensor(data, requiresGrad=requiresGrad)
    node._parents = tuple(parents)
    return node


def _nodeRequiresGrad(parents: Sequence[Tensor]) -> bool:
    """A node requires a gradient if any of its parents does."""
    return any(parent.requiresGrad for parent in parents)


# ----------------------------------------------------------------------
# Elementwise arithmetic
# ----------------------------------------------------------------------
def add(left: Union[Tensor, Numeric], right: Union[Tensor, Numeric]) -> Tensor:
    """Elementwise addition with broadcasting."""
    a = _toTensor(left)
    b = _toTensor(right)
    out = _makeNode(a.data + b.data, (a, b), _nodeRequiresGrad((a, b)))

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        if a.requiresGrad:
            _accumulate(a, reduceGradToShape(upstream, a.shape))
        if b.requiresGrad:
            _accumulate(b, reduceGradToShape(upstream, b.shape))

    out._backwardFn = backwardFn
    return out


def subtract(left: Union[Tensor, Numeric], right: Union[Tensor, Numeric]) -> Tensor:
    """Elementwise subtraction with broadcasting."""
    a = _toTensor(left)
    b = _toTensor(right)
    out = _makeNode(a.data - b.data, (a, b), _nodeRequiresGrad((a, b)))

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        if a.requiresGrad:
            _accumulate(a, reduceGradToShape(upstream, a.shape))
        if b.requiresGrad:
            _accumulate(b, reduceGradToShape(-upstream, b.shape))

    out._backwardFn = backwardFn
    return out


def multiply(left: Union[Tensor, Numeric], right: Union[Tensor, Numeric]) -> Tensor:
    """Elementwise multiplication with broadcasting."""
    a = _toTensor(left)
    b = _toTensor(right)
    out = _makeNode(a.data * b.data, (a, b), _nodeRequiresGrad((a, b)))

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        if a.requiresGrad:
            _accumulate(a, reduceGradToShape(upstream * b.data, a.shape))
        if b.requiresGrad:
            _accumulate(b, reduceGradToShape(upstream * a.data, b.shape))

    out._backwardFn = backwardFn
    return out


def divide(left: Union[Tensor, Numeric], right: Union[Tensor, Numeric]) -> Tensor:
    """Elementwise division with broadcasting."""
    a = _toTensor(left)
    b = _toTensor(right)
    if np.any(b.data == 0):
        raise ValueError("divide() received a zero denominator.")
    out = _makeNode(a.data / b.data, (a, b), _nodeRequiresGrad((a, b)))

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        if a.requiresGrad:
            _accumulate(a, reduceGradToShape(upstream / b.data, a.shape))
        if b.requiresGrad:
            contribution = -upstream * a.data / (b.data * b.data)
            _accumulate(b, reduceGradToShape(contribution, b.shape))

    out._backwardFn = backwardFn
    return out


def negate(value: Union[Tensor, Numeric]) -> Tensor:
    """Elementwise negation."""
    a = _toTensor(value)
    out = _makeNode(-a.data, (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        if a.requiresGrad:
            _accumulate(a, -out.grad)

    out._backwardFn = backwardFn
    return out


# ----------------------------------------------------------------------
# Linear algebra
# ----------------------------------------------------------------------
def matmul(left: Union[Tensor, Numeric], right: Union[Tensor, Numeric]) -> Tensor:
    """Matrix multiplication of two 2-D tensors.

    Only the 2-D case is supported on purpose.  The reverse rule for the
    batched case is a genuine source of subtle bugs and the AQUA network
    never needs it; raising loudly is safer than guessing.
    """
    a = _toTensor(left)
    b = _toTensor(right)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            "matmul() currently supports 2-D operands only, got "
            f"shapes {a.shape} and {b.shape}."
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul() shape mismatch: {a.shape} @ {b.shape} is not defined."
        )
    out = _makeNode(a.data @ b.data, (a, b), _nodeRequiresGrad((a, b)))

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        if a.requiresGrad:
            _accumulate(a, upstream @ b.data.T)
        if b.requiresGrad:
            _accumulate(b, a.data.T @ upstream)

    out._backwardFn = backwardFn
    return out


# ----------------------------------------------------------------------
# Reductions
# ----------------------------------------------------------------------
def sum(
    value: Union[Tensor, Numeric],
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum reduction over ``axis`` (all axes when ``axis`` is ``None``)."""
    a = _toTensor(value)
    out = _makeNode(a.data.sum(axis=axis, keepdims=keepdims), (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        if axis is None:
            contribution = np.broadcast_to(upstream, a.shape).copy()
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            if not keepdims:
                for reducedAxis in sorted(axes):
                    upstream = np.expand_dims(upstream, axis=reducedAxis)
            contribution = np.broadcast_to(upstream, a.shape).copy()
        if a.requiresGrad:
            _accumulate(a, contribution)

    out._backwardFn = backwardFn
    return out


def mean(
    value: Union[Tensor, Numeric],
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Mean reduction over ``axis`` (all axes when ``axis`` is ``None``)."""
    a = _toTensor(value)
    if axis is None:
        count = a.data.size
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        count = int(np.prod([a.data.shape[anAxis] for anAxis in axes], dtype=np.int64))
    out = _makeNode(a.data.mean(axis=axis, keepdims=keepdims), (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        if axis is None:
            contribution = np.broadcast_to(upstream, a.shape).copy() / count
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            if not keepdims:
                for reducedAxis in sorted(axes):
                    upstream = np.expand_dims(upstream, axis=reducedAxis)
            contribution = np.broadcast_to(upstream, a.shape).copy() / count
        if a.requiresGrad:
            _accumulate(a, contribution)

    out._backwardFn = backwardFn
    return out


def reshape(value: Union[Tensor, Numeric], shape: Union[int, tuple[int, ...]]) -> Tensor:
    """Reshape a tensor.  The gradient is reshaped back."""
    a = _toTensor(value)
    targetShape = (shape,) if isinstance(shape, int) else tuple(shape)
    out = _makeNode(a.data.reshape(targetShape), (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        if a.requiresGrad:
            _accumulate(a, out.grad.reshape(a.shape))

    out._backwardFn = backwardFn
    return out


# ----------------------------------------------------------------------
# Elementwise maths
# ----------------------------------------------------------------------
def exp(value: Union[Tensor, Numeric]) -> Tensor:
    """Elementwise exponential."""
    a = _toTensor(value)
    out = _makeNode(np.exp(a.data), (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        if a.requiresGrad:
            _accumulate(a, out.grad * out.data)

    out._backwardFn = backwardFn
    return out


def log(value: Union[Tensor, Numeric]) -> Tensor:
    """Elementwise natural logarithm (input must be strictly positive)."""
    a = _toTensor(value)
    if np.any(a.data <= 0):
        raise ValueError("log() requires strictly positive inputs.")
    out = _makeNode(np.log(a.data), (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        if a.requiresGrad:
            _accumulate(a, out.grad / a.data)

    out._backwardFn = backwardFn
    return out


def square(value: Union[Tensor, Numeric]) -> Tensor:
    """Elementwise square."""
    a = _toTensor(value)
    out = _makeNode(a.data * a.data, (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        if a.requiresGrad:
            _accumulate(a, out.grad * 2.0 * a.data)

    out._backwardFn = backwardFn
    return out


def maximum(value: Union[Tensor, Numeric], other: Union[Tensor, Numeric]) -> Tensor:
    """Elementwise maximum of two tensors (ties go to the left operand)."""
    a = _toTensor(value)
    b = _toTensor(other)
    mask = a.data >= b.data
    out = _makeNode(np.maximum(a.data, b.data), (a, b), _nodeRequiresGrad((a, b)))

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        if a.requiresGrad:
            _accumulate(a, reduceGradToShape(upstream * mask, a.shape))
        if b.requiresGrad:
            _accumulate(b, reduceGradToShape(upstream * (~mask), b.shape))

    out._backwardFn = backwardFn
    return out


def clip(
    value: Union[Tensor, Numeric],
    lower: float,
    upper: float,
) -> Tensor:
    """Elementwise clipping to ``[lower, upper]``."""
    a = _toTensor(value)
    out = _makeNode(np.clip(a.data, lower, upper), (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        if a.requiresGrad:
            mask = ((a.data >= lower) & (a.data <= upper)).astype(a.data.dtype)
            _accumulate(a, out.grad * mask)

    out._backwardFn = backwardFn
    return out


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------
def relu(value: Union[Tensor, Numeric]) -> Tensor:
    """Rectified linear unit."""
    a = _toTensor(value)
    out = _makeNode(np.maximum(a.data, 0.0), (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        if a.requiresGrad:
            _accumulate(a, out.grad * (a.data > 0.0))

    out._backwardFn = backwardFn
    return out


def leakyRelu(value: Union[Tensor, Numeric], negativeSlope: float = 0.01) -> Tensor:
    """Leaky rectified linear unit."""
    a = _toTensor(value)
    out = _makeNode(
        np.where(a.data > 0.0, a.data, negativeSlope * a.data),
        (a,),
        a.requiresGrad,
    )

    def backwardFn() -> None:
        if out.grad is None:
            return
        if a.requiresGrad:
            slope = np.where(a.data > 0.0, 1.0, negativeSlope)
            _accumulate(a, out.grad * slope)

    out._backwardFn = backwardFn
    return out


def tanh(value: Union[Tensor, Numeric]) -> Tensor:
    """Hyperbolic tangent."""
    a = _toTensor(value)
    out = _makeNode(np.tanh(a.data), (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        if a.requiresGrad:
            _accumulate(a, out.grad * (1.0 - out.data * out.data))

    out._backwardFn = backwardFn
    return out


def sigmoid(value: Union[Tensor, Numeric]) -> Tensor:
    """Logistic sigmoid, computed in a numerically stable way."""
    a = _toTensor(value)
    positive = a.data >= 0
    outData = np.empty_like(a.data, dtype=np.float64)
    outData[positive] = 1.0 / (1.0 + np.exp(-a.data[positive]))
    expNeg = np.exp(a.data[~positive])
    outData[~positive] = expNeg / (1.0 + expNeg)
    out = _makeNode(outData, (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        if a.requiresGrad:
            _accumulate(a, out.grad * out.data * (1.0 - out.data))

    out._backwardFn = backwardFn
    return out


# ----------------------------------------------------------------------
# Softmax family
# ----------------------------------------------------------------------
def _softmaxData(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def softmax(value: Union[Tensor, Numeric], axis: int = -1) -> Tensor:
    """Numerically stable softmax.

    The backward rule is the standard Jacobian contraction

    ``dL/dz = p * (dL/dp - sum(dL/dp * p))``
    """
    a = _toTensor(value)
    out = _makeNode(_softmaxData(a.data, axis=axis), (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        projected = upstream - np.sum(upstream * out.data, axis=axis, keepdims=True)
        if a.requiresGrad:
            _accumulate(a, out.data * projected)

    out._backwardFn = backwardFn
    return out


def logSoftmax(value: Union[Tensor, Numeric], axis: int = -1) -> Tensor:
    """Numerically stable log-softmax.

    ``dL/dz = dL/dl - sum(dL/dl) * softmax(z)``
    """
    a = _toTensor(value)
    shifted = a.data - np.max(a.data, axis=axis, keepdims=True)
    out = _makeNode(shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)), (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        probabilities = _softmaxData(a.data, axis=axis)
        if a.requiresGrad:
            _accumulate(a, upstream - np.sum(upstream, axis=axis, keepdims=True) * probabilities)

    out._backwardFn = backwardFn
    return out


def crossEntropyFromLogits(
    logits: Union[Tensor, Numeric],
    targets: np.ndarray,
    axis: int = -1,
) -> Tensor:
    """Mean cross-entropy over integer class targets, computed from raw logits.

    The forward value is ``-mean(log p[target])`` with ``p = softmax(logits)``
    evaluated in a numerically stable way (the maximum is subtracted before
    exponentiating).

    The gradient is the textbook

    ``dL/dlogits = (softmax(logits) - onehot(targets)) / batchSize``

    which is exact for the *mean* reduction and never forms a log of a
    probability that underflowed to zero.
    """
    a = _toTensor(logits)
    if a.ndim < 1:
        raise ValueError("crossEntropyFromLogits() requires logits with at least one axis.")
    targetArray = np.asarray(targets)
    if targetArray.shape != a.shape[:-1]:
        raise ValueError(
            f"Target shape {targetArray.shape} does not match logits shape {a.shape}."
        )
    if not np.issubdtype(targetArray.dtype, np.integer):
        raise ValueError("crossEntropyFromLogits() requires integer class indices.")

    batchSize = int(a.shape[0]) if a.ndim > 1 else 1
    shifted = a.data - np.max(a.data, axis=axis, keepdims=True)
    logSumExp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    logProbabilities = shifted - logSumExp

    indexArray = np.expand_dims(targetArray, axis)
    picked = np.take_along_axis(logProbabilities, indexArray, axis=axis)
    lossValue = -picked.mean()

    out = _makeNode(lossValue, (a,), a.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        probabilities = np.exp(logProbabilities)
        oneHot = np.zeros_like(probabilities)
        np.put_along_axis(oneHot, indexArray, 1.0, axis=axis)
        contribution = (probabilities - oneHot) / batchSize
        if a.requiresGrad:
            _accumulate(a, contribution * out.grad)

    out._backwardFn = backwardFn
    return out


# ----------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------
def applyMask(weight: Tensor, mask: np.ndarray) -> Tensor:
    """Apply a binary mask to a weight tensor: ``W_effective = W_raw * mask``.

    This is the single point in the engine where masking enters the
    computational model, and it is where the two distinct gradient semantics
    are separated:

    ``dLoss/dW_raw = (dLoss/dW_effective) * mask``
        Masked connections receive **exactly zero** gradient with respect to
        the raw parameter.  The optimiser therefore has nothing to act on,
        which is what we want.

    ``dLoss/dW_effective``
        The upstream gradient *before* the mask is applied.  It is generally
        **non-zero** even at masked positions, because it only depends on the
        layer input and the downstream gradient, not on the weight value.  It
        quantifies "how sensitive would the loss be to this connection if it
        were active?", which is exactly the signal needed to score candidate
        re-grown connections.  The value is stored on the produced node as
        ``effectiveGrad`` and mirrored onto the owning layer.

    Parameters
    ----------
    weight:
        The raw weight tensor (a leaf parameter).
    mask:
        A binary array broadcastable to the weight shape.  Values are treated
        as multiplicative factors; a mask of ``0`` removes the connection and
        a mask of ``1`` keeps it.

    Returns
    -------
    Tensor
        The effective weight tensor, wired so that the backward pass reduces
        the upstream gradient through the mask.
    """
    maskArray = np.asarray(mask, dtype=np.float64)
    if maskArray.shape != weight.shape:
        if np.broadcast_shapes(maskArray.shape, weight.shape) != weight.shape:
            raise ValueError(
                f"Mask shape {maskArray.shape} is not broadcastable to weight shape {weight.shape}."
            )
    effectiveData = weight.data * maskArray
    out = _makeNode(effectiveData, (weight,), weight.requiresGrad)

    def backwardFn() -> None:
        if out.grad is None:
            return
        upstream = out.grad
        # Capture the *unmasked* sensitivity before reducing through the mask.
        out.effectiveGrad = np.array(upstream, copy=True)
        if weight.requiresGrad:
            _accumulate(weight, upstream * maskArray)

    out._backwardFn = backwardFn
    return out
