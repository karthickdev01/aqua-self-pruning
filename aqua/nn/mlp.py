"""Factories that assemble a multi-layer perceptron.

The MLP is intentionally small: a list of :class:`~aqua.nn.layers.Linear`
layers, each with its own activation, plus an optional terminal activation
(``None`` means raw logits, which is what the cross-entropy loss expects).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from aqua.nn.activations import Activation, buildActivation
from aqua.nn.layers import DenseNetwork, Linear

__all__ = ["buildMlp", "mlpFromConfig", "DEFAULT_HIDDEN_SIZES"]

DEFAULT_HIDDEN_SIZES = (64, 64)


def buildMlp(
    inputSize: int,
    hiddenSizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    outputSize: int = 2,
    hiddenActivation: str = "relu",
    outputActivation: Optional[str] = None,
    useBias: bool = True,
    rng: Optional[np.random.Generator] = None,
    name: Optional[str] = None,
) -> DenseNetwork:
    """Build a fully connected MLP.

    Parameters
    ----------
    inputSize:
        Number of input features.
    hiddenSizes:
        Widths of the hidden layers.  An empty sequence produces a linear
        model.
    outputSize:
        Number of output classes.
    hiddenActivation:
        Activation used after every hidden layer.
    outputActivation:
        Activation used after the last layer.  ``None`` (the default) leaves
        raw logits, which is what the cross-entropy loss consumes.
    useBias:
        Whether hidden/output layers carry a bias.  Biases are never pruned.
    rng:
        NumPy generator used for the initialisation.  Passing a seeded
        generator makes the network bit-for-bit reproducible.
    """
    generator = rng if rng is not None else np.random.default_rng(0)

    sizes = [int(inputSize)] + [int(size) for size in hiddenSizes] + [int(outputSize)]
    layers: list[Linear] = []

    for index in range(len(sizes) - 1):
        isLast = index == len(sizes) - 2
        if isLast:
            activation = buildActivation(outputActivation) if outputActivation else None
        else:
            activation = buildActivation(hiddenActivation)
        layers.append(
            Linear(
                inFeatures=sizes[index],
                outFeatures=sizes[index + 1],
                bias=useBias,
                activation=activation,
                rng=generator,
                name=f"fc{index + 1}",
            )
        )

    return DenseNetwork(layers, name=name or "mlp")


def mlpFromConfig(config: dict, rng: Optional[np.random.Generator] = None) -> DenseNetwork:
    """Build an MLP from a plain configuration dictionary.

    The dictionary is expected to contain the keys ``inputSize``,
    ``hiddenSizes`` and ``outputSize``; ``hiddenActivation``,
    ``outputActivation`` and ``useBias`` are optional.
    """
    return buildMlp(
        inputSize=int(config["inputSize"]),
        hiddenSizes=config.get("hiddenSizes", DEFAULT_HIDDEN_SIZES),
        outputSize=int(config.get("outputSize", 2)),
        hiddenActivation=str(config.get("hiddenActivation", "relu")),
        outputActivation=config.get("outputActivation", None),
        useBias=bool(config.get("useBias", True)),
        rng=rng,
        name=config.get("name", None),
    )
