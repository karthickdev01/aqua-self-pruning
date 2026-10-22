"""Finite-difference gradient checking utilities.

The checker is deliberately independent of the engine: it perturbs the raw
NumPy data of the input tensors and re-evaluates the forward function, so a
bug in the backward rules cannot hide behind a shared helper.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from aqua.engine.tensor import Tensor

__all__ = ["numericGradient", "checkGradients", "GradCheckResult"]


def numericGradient(
    forwardFn: Callable[[], Tensor],
    inputs: Sequence[Tensor],
    epsilon: float = 1e-6,
) -> list[np.ndarray]:
    """Central-difference numerical gradient of ``forwardFn`` w.r.t. ``inputs``.

    ``forwardFn`` must rebuild and return the scalar loss tensor from the
    *current* contents of ``inputs`` (they are mutated in place by this
    function).
    """
    gradients: list[np.ndarray] = []
    for tensor in inputs:
        gradient = np.zeros_like(tensor.data, dtype=np.float64)
        flatGradient = gradient.reshape(-1)
        flatData = tensor.data.reshape(-1)
        for index in range(flatData.size):
            original = flatData[index]

            flatData[index] = original + epsilon
            plusValue = float(np.asarray(forwardFn().data).reshape(-1)[0])

            flatData[index] = original - epsilon
            minusValue = float(np.asarray(forwardFn().data).reshape(-1)[0])

            flatData[index] = original
            flatGradient[index] = (plusValue - minusValue) / (2.0 * epsilon)
        gradients.append(gradient)
    return gradients


class GradCheckResult:
    """Outcome of a gradient check."""

    def __init__(self, name: str, maxAbsError: float, maxRelError: float, passed: bool) -> None:
        self.name = name
        self.maxAbsError = maxAbsError
        self.maxRelError = maxRelError
        self.passed = passed

    def __repr__(self) -> str:
        return (
            f"GradCheckResult(name={self.name!r}, passed={self.passed}, "
            f"maxAbsError={self.maxAbsError:.3e}, maxRelError={self.maxRelError:.3e})"
        )


def checkGradients(
    forwardFn: Callable[[], Tensor],
    inputs: Sequence[Tensor],
    name: str = "op",
    epsilon: float = 1e-6,
    tolerance: float = 1e-6,
    relativeTolerance: float = 1e-5,
) -> GradCheckResult:
    """Compare analytic gradients against finite differences.

    The check passes when the absolute error is below ``tolerance`` *or* the
    relative error is below ``relativeTolerance``.  The relative test is what
    makes the check meaningful for gradients that are legitimately tiny.

    The analytic gradients are read from ``inputs[i].grad`` after the backward
    pass, which the caller must have triggered.  This function runs the
    backward pass itself so that the gradient buffers are fresh.
    """
    loss = forwardFn()
    loss.backward()

    analytic = [None if tensor.grad is None else tensor.grad for tensor in inputs]
    numerical = numericGradient(forwardFn, inputs, epsilon=epsilon)

    maxAbsError = 0.0
    maxRelError = 0.0
    for analyticGradient, numericGradientValue in zip(analytic, numerical):
        if analyticGradient is None:
            # A constant input legitimately has no gradient.
            if np.any(numericGradientValue != 0.0):
                return GradCheckResult(name, float("inf"), float("inf"), False)
            continue
        difference = np.abs(analyticGradient - numericGradientValue)
        maxAbsError = max(maxAbsError, float(difference.max()) if difference.size else 0.0)
        denominator = np.maximum(np.abs(analyticGradient), np.abs(numericGradientValue))
        relative = np.where(denominator > 1e-12, difference / np.maximum(denominator, 1e-12), 0.0)
        maxRelError = max(maxRelError, float(relative.max()) if relative.size else 0.0)

    passed = maxAbsError < tolerance or maxRelError < relativeTolerance
    return GradCheckResult(name, maxAbsError, maxRelError, passed)
