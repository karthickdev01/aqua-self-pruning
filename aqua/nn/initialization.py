"""Weight initialisation schemes with their mathematical justification.

Why initialisation matters
--------------------------
At initialisation every pre-activation is the sum of ``fanIn`` i.i.d. terms.
If the weights are too small the forward signal decays layer by layer and
gradients vanish; if they are too large the signal explodes and the network
diverges.  The variance of the pre-activation is therefore

``Var(z) = fanIn * Var(w) * Var(activationInput)``

and the schemes below pick ``Var(w)`` so that the signal neither decays nor
grows as it passes through a layer.

He / Kaiming (for ReLU)
-----------------------
ReLU halves the variance of a Gaussian pre-activation (``E[relu(x)^2] =
0.5 Var(x)`` for symmetric zero-mean ``x``).  To keep the variance of the
pre-activation at ``1`` we need

``Var(w) = 2 / fanIn``      =>   ``std = sqrt(2 / fanIn)``

Glorot / Xavier (for tanh and sigmoid)
--------------------------------------
For a symmetric, approximately linear activation the variance is preserved
when both the forward and backward variances are equal.  Balancing the two
gives

``Var(w) = 2 / (fanIn + fanOut)``   =>   ``std = sqrt(2 / (fanIn + fanOut))``

which is the standard Glorot normal initialisation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "heNormal",
    "heUniform",
    "glorotNormal",
    "glorotUniform",
    "zeros",
    "initializeWeight",
]

DEFAULT_RNG_SEED = 0


def heNormal(
    shape: tuple[int, ...],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """He normal initialisation: ``N(0, 2 / fanIn)``.

    Use for layers followed by ReLU (or leaky ReLU with a small slope).
    """
    generator = rng if rng is not None else np.random.default_rng(DEFAULT_RNG_SEED)
    fanIn = int(shape[0])
    std = np.sqrt(2.0 / fanIn)
    return generator.normal(loc=0.0, scale=std, size=shape)


def heUniform(
    shape: tuple[int, ...],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """He uniform initialisation: ``U(-sqrt(6 / fanIn), sqrt(6 / fanIn))``."""
    generator = rng if rng is not None else np.random.default_rng(DEFAULT_RNG_SEED)
    fanIn = int(shape[0])
    limit = np.sqrt(6.0 / fanIn)
    return generator.uniform(low=-limit, high=limit, size=shape)


def glorotNormal(
    shape: tuple[int, ...],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Glorot normal initialisation: ``N(0, 2 / (fanIn + fanOut))``.

    Use for layers followed by tanh or sigmoid.
    """
    generator = rng if rng is not None else np.random.default_rng(DEFAULT_RNG_SEED)
    fanIn = int(shape[0])
    fanOut = int(shape[-1])
    std = np.sqrt(2.0 / (fanIn + fanOut))
    return generator.normal(loc=0.0, scale=std, size=shape)


def glorotUniform(
    shape: tuple[int, ...],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Glorot uniform initialisation: ``U(-sqrt(6/(fanIn+fanOut)), ...)``."""
    generator = rng if rng is not None else np.random.default_rng(DEFAULT_RNG_SEED)
    fanIn = int(shape[0])
    fanOut = int(shape[-1])
    limit = np.sqrt(6.0 / (fanIn + fanOut))
    return generator.uniform(low=-limit, high=limit, size=shape)


def zeros(shape: tuple[int, ...]) -> np.ndarray:
    """All-zero initialisation (used for biases)."""
    return np.zeros(shape, dtype=np.float64)


def initializeWeight(
    shape: tuple[int, ...],
    activation: str = "relu",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Dispatch to the right scheme for ``activation``.

    Parameters
    ----------
    shape:
        ``(fanIn, fanOut)`` weight shape.
    activation:
        ``"relu"``, ``"leakyRelu"``, ``"tanh"`` or ``"sigmoid"``.
    """
    key = activation.lower()
    if key in {"relu", "leakyrelu"}:
        return heNormal(shape, rng)
    if key in {"tanh", "sigmoid"}:
        return glorotNormal(shape, rng)
    raise ValueError(
        f"Unknown activation '{activation}'; expected one of relu, leakyRelu, tanh, sigmoid."
    )
