"""Activation functions used by the AQUA MLP.

Each activation is a small callable object with a stable ``name`` attribute
so that the initialisation scheme, the JSON serialisation and the plotting
code can refer to it without stringly-typed lookups everywhere.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from aqua.engine import ops
from aqua.engine.tensor import Tensor

__all__ = ["Activation", "ReLU", "LeakyReLU", "Tanh", "Sigmoid", "Identity"]


class Activation:
    """Base class for elementwise, differentiable activations."""

    name: str = "identity"
    initialisationKey: str = "relu"

    def __call__(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{type(self).__name__}()"


class ReLU(Activation):
    """``max(0, x)``."""

    name = "relu"
    initialisationKey = "relu"

    def __call__(self, value: Tensor) -> Tensor:
        return ops.relu(value)


class LeakyReLU(Activation):
    """``x`` for ``x > 0``, ``negativeSlope * x`` otherwise."""

    name = "leakyRelu"
    initialisationKey = "relu"

    def __init__(self, negativeSlope: float = 0.01) -> None:
        self.negativeSlope = float(negativeSlope)

    def __call__(self, value: Tensor) -> Tensor:
        return ops.leakyRelu(value, negativeSlope=self.negativeSlope)


class Tanh(Activation):
    """``tanh(x)``."""

    name = "tanh"
    initialisationKey = "tanh"

    def __call__(self, value: Tensor) -> Tensor:
        return ops.tanh(value)


class Sigmoid(Activation):
    """``1 / (1 + exp(-x))`` evaluated in a numerically stable way."""

    name = "sigmoid"
    initialisationKey = "sigmoid"

    def __call__(self, value: Tensor) -> Tensor:
        return ops.sigmoid(value)


class Identity(Activation):
    """The identity function."""

    name = "identity"
    initialisationKey = "relu"

    def __call__(self, value: Tensor) -> Tensor:
        return value


def buildActivation(name: str) -> Activation:
    """Create an activation object from its string name."""
    lookup: dict[str, Callable[[], Activation]] = {
        "relu": ReLU,
        "leakyRelu": LeakyReLU,
        "tanh": Tanh,
        "sigmoid": Sigmoid,
        "identity": Identity,
    }
    if name not in lookup:
        raise ValueError(f"Unknown activation '{name}'. Known: {sorted(lookup)}")
    return lookup[name]()
