"""Optimisers for the AQUA project.

Only Adam is implemented; it is written from scratch in
:mod:`aqua.optim.adam` so that the mask-aware state semantics are explicit
and testable.
"""

from aqua.optim.adam import (
    DEFAULT_BETA1,
    DEFAULT_BETA2,
    DEFAULT_EPSILON,
    DEFAULT_LEARNING_RATE,
    AdamOptimizer,
    AdamState,
)

__all__ = [
    "AdamOptimizer",
    "AdamState",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BETA1",
    "DEFAULT_BETA2",
    "DEFAULT_EPSILON",
]
