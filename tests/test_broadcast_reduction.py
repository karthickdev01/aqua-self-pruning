"""Exhaustive verification of the broadcasting gradient reduction.

The reduction rule is the single most error-prone piece of a NumPy
autodiff engine, so it is verified against an *independent* reference
implementation that literally replays NumPy's broadcasting index mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqua.engine.ops import reduceGradToShape


def _referenceReduction(gradient: np.ndarray, targetShape: tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcast gradient using an explicit index replay.

    This mirrors what NumPy does when it broadcasts ``targetShape`` up to
    ``gradient.shape``: the operand's axes are aligned with the *trailing*
    axes of the output, and an axis of size 1 is read at index 0 for every
    position along that axis.
    """
    padded = (1,) * (gradient.ndim - len(targetShape)) + tuple(targetShape)
    result = np.zeros(targetShape, dtype=np.float64)
    for index in np.ndindex(gradient.shape):
        mapped = tuple(
            0 if padded[axis] == 1 and gradient.shape[axis] > 1 else index[axis]
            for axis in range(gradient.ndim)
        )[-len(targetShape) :]
        result[mapped] += gradient[index]
    return result


BROADCAST_CASES = [
    ((4, 3), (3,)),
    ((4, 3), (1,)),
    ((4, 3), (1, 3)),
    ((4, 3, 5), (3, 5)),
    ((4, 3, 5), (5,)),
    ((4, 3, 5), (1, 5)),
    ((4, 3, 5), (3, 1)),
    ((4, 3, 5), (1, 1, 5)),
    ((4, 3, 5), (4, 3, 5)),
    ((4, 3, 5), (1, 3, 1)),
    ((7,), (7,)),
    ((7,), (1,)),
]


@pytest.mark.parametrize("outputShape,targetShape", BROADCAST_CASES)
def test_reduce_matches_index_replay(outputShape, targetShape) -> None:
    """The reduction must match an explicit replay of the broadcast map."""
    # Every case must be a legal NumPy broadcast.
    assert np.broadcast_shapes(targetShape, outputShape) == outputShape

    gradient = np.random.default_rng(0).normal(size=outputShape)
    reduced = reduceGradToShape(gradient, targetShape)
    expected = _referenceReduction(gradient, targetShape)
    assert reduced.shape == targetShape
    assert np.allclose(reduced, expected, atol=1e-12)


def test_reduce_is_identity_when_shapes_match() -> None:
    gradient = np.random.default_rng(1).normal(size=(3, 4))
    assert reduceGradToShape(gradient, (3, 4)) is gradient


def test_reduce_sums_the_broadcast_axis() -> None:
    """The challenge's example: (N, D) + (D,) must sum over N."""
    gradient = np.ones((5, 3))
    reduced = reduceGradToShape(gradient, (3,))
    assert np.allclose(reduced, np.array([5.0, 5.0, 5.0]))


def test_reduce_keeps_the_rank() -> None:
    """A stretched axis must be summed with keepdims so the rank survives."""
    gradient = np.ones((4, 3))
    reduced = reduceGradToShape(gradient, (1, 3))
    assert reduced.shape == (1, 3)
    assert np.allclose(reduced, np.array([[4.0, 4.0, 4.0]]))
