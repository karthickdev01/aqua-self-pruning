"""Gradient checks for every operation implemented in :mod:`aqua.engine.ops`.

Every test follows the same shape: build a small forward function, run the
finite-difference checker, and assert the analytic gradient matches.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqua.engine import ops
from aqua.engine.tensor import Parameter, Tensor
from tests.gradient_checks import checkGradients


def _random(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape)


# ----------------------------------------------------------------------
# Elementwise arithmetic
# ----------------------------------------------------------------------
def test_add_scalar_gradient() -> None:
    left = Parameter(_random((3, 4), 1))
    right = Parameter(_random((3, 4), 2))

    def forward() -> Tensor:
        return ops.sum(ops.add(left, right))

    result = checkGradients(forward, [left, right], "add")
    assert result.passed, result


def test_subtract_gradient() -> None:
    left = Parameter(_random((3, 4), 3))
    right = Parameter(_random((3, 4), 4))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.subtract(left, right), _random((3, 4), 5)))

    result = checkGradients(forward, [left, right], "subtract")
    assert result.passed, result


def test_multiply_gradient() -> None:
    left = Parameter(_random((2, 3), 6))
    right = Parameter(_random((3,), 7))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(left, right))

    result = checkGradients(forward, [left, right], "multiply")
    assert result.passed, result


def test_divide_gradient() -> None:
    left = Parameter(_random((2, 3), 8) + 3.0)
    right = Parameter(_random((2, 3), 9) + 3.0)

    def forward() -> Tensor:
        return ops.sum(ops.divide(left, right))

    result = checkGradients(forward, [left, right], "divide")
    assert result.passed, result


def test_divide_rejects_zero_denominator() -> None:
    left = Parameter(np.ones((2,)))
    right = Parameter(np.zeros((2,)))
    with pytest.raises(ValueError):
        ops.divide(left, right)


def test_negate_gradient() -> None:
    value = Parameter(_random((4,), 10))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.negate(value), _random((4,), 11)))

    result = checkGradients(forward, [value], "negate")
    assert result.passed, result


# ----------------------------------------------------------------------
# Broadcasting
# ----------------------------------------------------------------------
def test_broadcast_row_vector_gradient() -> None:
    """(N, D) + (D,) must backpropagate correctly to both operands."""
    matrix = Parameter(_random((5, 3), 20))
    row = Parameter(_random((3,), 21))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.add(matrix, row), _random((5, 3), 22)))

    result = checkGradients(forward, [matrix, row], "broadcast-row")
    assert result.passed, result


def test_broadcast_column_vector_gradient() -> None:
    matrix = Parameter(_random((5, 3), 23))
    column = Parameter(_random((5, 1), 24))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.add(matrix, column), _random((5, 3), 25)))

    result = checkGradients(forward, [matrix, column], "broadcast-column")
    assert result.passed, result


def test_broadcast_scalar_gradient() -> None:
    matrix = Parameter(_random((4, 2), 26))
    scalar = Parameter(np.array(2.5))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.add(matrix, scalar), _random((4, 2), 27)))

    result = checkGradients(forward, [matrix, scalar], "broadcast-scalar")
    assert result.passed, result


def test_broadcast_keeps_operand_shape() -> None:
    matrix = Parameter(_random((4, 3), 28))
    row = Parameter(_random((3,), 29))
    out = ops.add(matrix, row)
    ops.sum(out).backward()
    assert matrix.grad is not None and matrix.grad.shape == (4, 3)
    assert row.grad is not None and row.grad.shape == (3,)


# ----------------------------------------------------------------------
# Matrix multiplication
# ----------------------------------------------------------------------
def test_matmul_gradient() -> None:
    left = Parameter(_random((4, 5), 30))
    right = Parameter(_random((5, 3), 31))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.matmul(left, right), _random((4, 3), 32)))

    result = checkGradients(forward, [left, right], "matmul")
    assert result.passed, result


def test_matmul_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        ops.matmul(Parameter(np.ones((3, 4))), Parameter(np.ones((5, 6))))
    with pytest.raises(ValueError):
        ops.matmul(Parameter(np.ones((3,))), Parameter(np.ones((3, 2))))


# ----------------------------------------------------------------------
# Reductions
# ----------------------------------------------------------------------
def test_sum_gradient_all_axes() -> None:
    value = Parameter(_random((3, 4), 33))

    def forward() -> Tensor:
        return ops.sum(value)

    result = checkGradients(forward, [value], "sum-all")
    assert result.passed, result


def test_sum_gradient_over_axis() -> None:
    value = Parameter(_random((3, 4), 34))

    def forward() -> Tensor:
        return ops.sum(ops.sum(value, axis=1), axis=0)

    result = checkGradients(forward, [value], "sum-axis")
    assert result.passed, result


def test_sum_gradient_keepdims() -> None:
    value = Parameter(_random((3, 4), 35))

    def forward() -> Tensor:
        return ops.sum(ops.sum(value, axis=1, keepdims=True))

    result = checkGradients(forward, [value], "sum-keepdims")
    assert result.passed, result


def test_mean_gradient() -> None:
    value = Parameter(_random((3, 4), 36))

    def forward() -> Tensor:
        return ops.mean(value)

    result = checkGradients(forward, [value], "mean-all")
    assert result.passed, result


def test_mean_gradient_over_axis() -> None:
    value = Parameter(_random((3, 4), 37))

    def forward() -> Tensor:
        return ops.mean(ops.sum(value, axis=1))

    result = checkGradients(forward, [value], "mean-axis")
    assert result.passed, result


def test_reshape_gradient() -> None:
    value = Parameter(_random((2, 6), 38))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.reshape(value, (3, 4)), _random((3, 4), 39)))

    result = checkGradients(forward, [value], "reshape")
    assert result.passed, result


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------
def test_relu_gradient() -> None:
    value = Parameter(_random((5, 4), 40) + 0.1)

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.relu(value), _random((5, 4), 41)))

    result = checkGradients(forward, [value], "relu")
    assert result.passed, result


def test_leaky_relu_gradient() -> None:
    value = Parameter(_random((5, 4), 42))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.leakyRelu(value, 0.05), _random((5, 4), 43)))

    result = checkGradients(forward, [value], "leakyRelu")
    assert result.passed, result


def test_tanh_gradient() -> None:
    value = Parameter(_random((5, 4), 44))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.tanh(value), _random((5, 4), 45)))

    result = checkGradients(forward, [value], "tanh")
    assert result.passed, result


def test_sigmoid_gradient() -> None:
    value = Parameter(_random((5, 4), 46))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.sigmoid(value), _random((5, 4), 47)))

    result = checkGradients(forward, [value], "sigmoid")
    assert result.passed, result


def test_softmax_gradient() -> None:
    logits = Parameter(_random((6, 3), 48))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.softmax(logits), _random((6, 3), 49)))

    result = checkGradients(forward, [logits], "softmax")
    assert result.passed, result


def test_log_softmax_gradient() -> None:
    logits = Parameter(_random((6, 3), 50))

    def forward() -> Tensor:
        return ops.sum(ops.multiply(ops.logSoftmax(logits), _random((6, 3), 51)))

    result = checkGradients(forward, [logits], "logSoftmax")
    assert result.passed, result


def test_cross_entropy_gradient() -> None:
    logits = Parameter(_random((7, 4), 52))
    targets = np.array([0, 3, 1, 2, 0, 3, 1])

    def forward() -> Tensor:
        return ops.crossEntropyFromLogits(logits, targets)

    result = checkGradients(forward, [logits], "crossEntropy")
    assert result.passed, result


def test_cross_entropy_matches_closed_form() -> None:
    """The gradient must equal (softmax - onehot) / batchSize exactly."""
    logits = Parameter(_random((5, 3), 53))
    targets = np.array([0, 1, 2, 1, 0])
    loss = ops.crossEntropyFromLogits(logits, targets)
    loss.backward()
    probabilities = ops.softmax(logits).data
    oneHot = np.eye(3)[targets]
    expected = (probabilities - oneHot) / 5.0
    assert np.allclose(logits.grad, expected, atol=1e-12)


def test_cross_entropy_is_numerically_stable() -> None:
    """Large logits must not produce NaN/inf losses."""
    logits = Parameter(np.array([[1000.0, 1001.0, 1002.0]]))
    targets = np.array([2])
    loss = ops.crossEntropyFromLogits(logits, targets)
    assert np.isfinite(loss.data).all()
    loss.backward()
    assert np.isfinite(logits.grad).all()


# ----------------------------------------------------------------------
# Misc maths
# ----------------------------------------------------------------------
def test_exp_log_square_gradient() -> None:
    value = Parameter(_random((4,), 54) + 2.0)

    def forward() -> Tensor:
        return ops.sum(ops.square(ops.log(ops.exp(value))))

    result = checkGradients(forward, [value], "exp-log-square")
    assert result.passed, result


def test_maximum_gradient() -> None:
    left = Parameter(_random((4,), 55))
    right = Parameter(_random((4,), 56))

    def forward() -> Tensor:
        return ops.sum(ops.maximum(left, right))

    result = checkGradients(forward, [left, right], "maximum")
    assert result.passed, result


def test_clip_gradient() -> None:
    value = Parameter(_random((4,), 57))

    def forward() -> Tensor:
        return ops.sum(ops.clip(value, -0.5, 0.5))

    result = checkGradients(forward, [value], "clip")
    assert result.passed, result
