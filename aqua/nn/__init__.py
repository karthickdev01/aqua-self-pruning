"""Neural-network building blocks for the AQUA project.

* :mod:`aqua.nn.initialization` -- He and Glorot initialisation.
* :mod:`aqua.nn.activations` -- activation functions.
* :mod:`aqua.nn.layers` -- the prunable :class:`~aqua.nn.layers.Linear` layer
  and the :class:`~aqua.nn.layers.DenseNetwork` container.
* :mod:`aqua.nn.mlp` -- a factory that assembles an MLP with a chosen
  activation and initialisation scheme.
"""

from aqua.nn.activations import (
    Activation,
    Identity,
    LeakyReLU,
    ReLU,
    Sigmoid,
    Tanh,
    buildActivation,
)
from aqua.nn.initialization import (
    glorotNormal,
    glorotUniform,
    heNormal,
    heUniform,
    initializeWeight,
    zeros,
)
from aqua.nn.layers import DenseNetwork, Linear
from aqua.nn.mlp import buildMlp, mlpFromConfig

__all__ = [
    "Activation",
    "DenseNetwork",
    "Identity",
    "LeakyReLU",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "buildActivation",
    "buildMlp",
    "glorotNormal",
    "glorotUniform",
    "heNormal",
    "heUniform",
    "initializeWeight",
    "mlpFromConfig",
    "zeros",
]
