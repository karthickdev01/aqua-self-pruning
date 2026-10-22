"""Metrics and loss helpers.

Everything here is pure NumPy.  The loss is the mean cross-entropy over a
mini-batch, computed from raw logits in a numerically stable way (see
:func:`aqua.engine.ops.crossEntropyFromLogits`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from aqua.engine import ops
from aqua.engine.tensor import Tensor

__all__ = [
    "BatchMetrics",
    "computeLoss",
    "computeAccuracy",
    "predictLabels",
    "accuracyFromPredictions",
]


@dataclass
class BatchMetrics:
    """Metrics for one mini-batch or one evaluation pass."""

    loss: float = 0.0
    accuracy: float = 0.0
    sampleCount: int = 0

    def asDict(self) -> Dict[str, float]:
        """Return the metrics as a plain dictionary."""
        return {
            "loss": float(self.loss),
            "accuracy": float(self.accuracy),
            "sampleCount": int(self.sampleCount),
        }


def computeLoss(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Mean cross-entropy loss of ``logits`` against integer ``labels``."""
    return ops.crossEntropyFromLogits(logits, labels)


def predictLabels(logits: Tensor) -> np.ndarray:
    """Return the arg-max class index for every row of ``logits``."""
    return np.argmax(logits.data, axis=1)


def accuracyFromPredictions(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of correct predictions."""
    if labels.size == 0:
        return 0.0
    return float(np.mean(predictions == labels))


def computeAccuracy(logits: Tensor, labels: np.ndarray) -> float:
    """Accuracy of ``logits`` against ``labels``."""
    return accuracyFromPredictions(predictLabels(logits), labels)
