"""Deterministic synthetic datasets (pure NumPy, no scikit-learn)."""

from aqua.data.datasets import (
    Dataset,
    makeSpiralDataset,
    makeSyntheticDataset,
    makeTwoMoonsDataset,
)

__all__ = [
    "Dataset",
    "makeSpiralDataset",
    "makeSyntheticDataset",
    "makeTwoMoonsDataset",
]
