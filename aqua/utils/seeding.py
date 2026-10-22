"""Seeding helpers.

Reproducibility is a hard requirement of the challenge, so every source of
randomness in the project is funnelled through this module.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["setGlobalSeed", "makeGenerator", "DEFAULT_SEED"]

DEFAULT_SEED = 0


def setGlobalSeed(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Seed Python, NumPy and the OS entropy pool.

    Returns a fresh NumPy generator seeded with ``seed``.  Passing the
    returned generator into the model initialisation makes the whole
    pipeline independent of global state.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def makeGenerator(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a NumPy generator seeded with ``seed``."""
    return np.random.default_rng(seed)
