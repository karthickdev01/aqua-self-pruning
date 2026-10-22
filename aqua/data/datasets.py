"""Deterministic synthetic datasets.

Why a synthetic dataset
-----------------------
The challenge allows any classification task.  A synthetic two-class spiral
is used here for four reasons:

1. **Reproducibility.**  It is generated from a seeded generator, so a clean
   clone reproduces the exact same numbers.
2. **Debuggability.**  The decision boundary is known analytically, so a
   training bug is visible immediately in the accuracy curve.
3. **Speed.**  A few thousand samples of two features means a full experiment
   sweep over five sparsities x five seeds x several criteria finishes in
   minutes, which is what makes a genuine multi-seed study affordable.
4. **A non-linear but *not* adversarially non-linear problem.**  A spiral
   needs a hidden layer but does not need a huge one, so the pruning study
   measures the pruning, not the capacity.

The dataset is generated in pure NumPy.  **scikit-learn is not used
anywhere in this project** -- not for data loading, not for models, not for
gradients.  See ``README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["Dataset", "makeSpiralDataset", "makeTwoMoonsDataset", "makeSyntheticDataset"]


@dataclass
class Dataset:
    """A train/validation split of a labelled classification dataset."""

    trainFeatures: np.ndarray
    trainLabels: np.ndarray
    validationFeatures: np.ndarray
    validationLabels: np.ndarray
    featureCount: int
    classCount: int
    name: str = "dataset"

    def describe(self) -> str:
        """Return a short human readable summary."""
        return (
            f"{self.name}: train={self.trainFeatures.shape}, "
            f"validation={self.validationFeatures.shape}, "
            f"features={self.featureCount}, classes={self.classCount}"
        )


def _standardise(trainFeatures: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Zero-mean/unit-variance standardisation fitted on the training split.

    Fitting the statistics on the *training* split only is a deliberate
    guard against data leakage: the validation set never influences any
    statistic that the model sees.
    """
    mean = trainFeatures.mean(axis=0, keepdims=True)
    std = trainFeatures.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return tuple((array - mean) / std for array in (trainFeatures, *others))


def _splitIndices(
    totalCount: int,
    validationFraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a random train/validation index split."""
    indices = rng.permutation(totalCount)
    validationCount = int(round(totalCount * validationFraction))
    validationCount = max(1, min(validationCount, totalCount - 1))
    return indices[validationCount:], indices[:validationCount]


def makeSpiralDataset(
    sampleCount: int = 1200,
    validationFraction: float = 0.25,
    noiseStd: float = 0.35,
    turns: float = 1.6,
    radius: float = 6.0,
    seed: int = 0,
    name: str = "spiral",
) -> Dataset:
    """Generate a two-class spiral classification dataset.

    Parameters
    ----------
    sampleCount:
        Total number of samples (train + validation).
    validationFraction:
        Fraction of samples held out for validation.
    noiseStd:
        Standard deviation of the isotropic Gaussian noise added to each
        point.  Larger values make the task harder and the decision boundary
        less trivial.
    turns:
        Number of turns of the spiral.
    radius:
        Radius of the outermost turn.
    seed:
        Seed for the NumPy generator.

    Returns
    -------
    Dataset
        A standardised train/validation split.
    """
    if sampleCount < 4:
        raise ValueError("sampleCount must be at least 4.")
    if not 0.0 < validationFraction < 1.0:
        raise ValueError("validationFraction must lie in (0, 1).")

    rng = np.random.default_rng(seed)
    halfCount = sampleCount // 2
    counts = (halfCount, sampleCount - halfCount)

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for classIndex, classCount in enumerate(counts):
        # The radius must be *correlated* with the angle, otherwise the two
        # "classes" are just two annuli and the task is not separable.
        # Parameterising by t in [0, 1] gives a genuine spiral arm: the point
        # sweeps `turns` full revolutions while moving outwards from the
        # origin to `radius`.
        t = rng.uniform(0.0, 1.0, size=classCount)
        angle = t * turns * 2.0 * np.pi + classIndex * np.pi
        distance = radius * t
        points = np.stack(
            [distance * np.cos(angle), distance * np.sin(angle)],
            axis=1,
        )
        points = points + rng.normal(0.0, noiseStd, size=points.shape)
        features.append(points)
        labels.append(np.full(classCount, classIndex, dtype=np.int64))

    allFeatures = np.concatenate(features, axis=0)
    allLabels = np.concatenate(labels, axis=0)

    trainIndices, validationIndices = _splitIndices(sampleCount, validationFraction, rng)
    trainFeatures = allFeatures[trainIndices]
    validationFeatures = allFeatures[validationIndices]
    trainLabels = allLabels[trainIndices]
    validationLabels = allLabels[validationIndices]

    trainFeatures, validationFeatures = _standardise(trainFeatures, validationFeatures)

    return Dataset(
        trainFeatures=trainFeatures.astype(np.float64),
        trainLabels=trainLabels,
        validationFeatures=validationFeatures.astype(np.float64),
        validationLabels=validationLabels,
        featureCount=allFeatures.shape[1],
        classCount=2,
        name=name,
    )


def makeTwoMoonsDataset(
    sampleCount: int = 1200,
    validationFraction: float = 0.25,
    noiseStd: float = 0.2,
    seed: int = 0,
    name: str = "twoMoons",
) -> Dataset:
    """Generate the classic two-moons dataset (an easy sanity-check task)."""
    if sampleCount < 4:
        raise ValueError("sampleCount must be at least 4.")
    rng = np.random.default_rng(seed)
    halfCount = sampleCount // 2

    firstAngle = np.linspace(0.0, np.pi, halfCount)
    first = np.stack([np.cos(firstAngle), np.sin(firstAngle)], axis=1)
    secondAngle = np.linspace(0.0, np.pi, halfCount)
    second = np.stack([1.0 - np.cos(secondAngle), 0.5 - np.sin(secondAngle)], axis=1)

    features = np.concatenate([first, second], axis=0)
    labels = np.concatenate(
        [np.zeros(halfCount, dtype=np.int64), np.ones(halfCount, dtype=np.int64)],
        axis=0,
    )
    features = features + rng.normal(0.0, noiseStd, size=features.shape)

    trainIndices, validationIndices = _splitIndices(sampleCount, validationFraction, rng)
    trainFeatures, validationFeatures = _standardise(
        features[trainIndices], features[validationIndices]
    )

    return Dataset(
        trainFeatures=trainFeatures.astype(np.float64),
        trainLabels=labels[trainIndices],
        validationFeatures=validationFeatures.astype(np.float64),
        validationLabels=labels[validationIndices],
        featureCount=2,
        classCount=2,
        name=name,
    )


def makeSyntheticDataset(
    kind: str = "spiral",
    **kwargs: object,
) -> Dataset:
    """Factory dispatching to the concrete dataset builders."""
    key = kind.lower()
    if key == "spiral":
        return makeSpiralDataset(**kwargs)  # type: ignore[arg-type]
    if key in {"twomoons", "two_moons"}:
        return makeTwoMoonsDataset(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown dataset kind '{kind}'. Known: spiral, twoMoons")
