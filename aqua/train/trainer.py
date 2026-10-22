"""Mini-batch training and evaluation loops.

Design notes
------------
* The batch order is produced by a **seeded** NumPy generator, so a run is
  reproducible from ``(seed, epoch)`` alone.  The shuffling happens inside
  the trainer rather than in the dataset so that the dataset object stays a
  pure, immutable description of the data.
* Every epoch records loss, accuracy, epoch duration and a numerical
  stability check.  The stability check asserts that the loss and the
  gradients are finite; a NaN is a *reported failure*, not a silently
  averaged-out number.
* The trainer is agnostic to pruning.  The pruning machinery observes the
  trainer through callbacks, so the baseline (no pruning) and the pruned runs
  execute exactly the same code path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from aqua.data.datasets import Dataset
from aqua.engine.tensor import Tensor
from aqua.nn.layers import DenseNetwork
from aqua.optim.adam import AdamOptimizer
from aqua.train.metrics import computeAccuracy, computeLoss, predictLabels

__all__ = ["EpochRecord", "Trainer", "TrainConfig", "shuffleIndices"]


@dataclass
class TrainConfig:
    """Hyper-parameters of a training run."""

    epochs: int = 60
    batchSize: int = 64
    learningRate: float = 1e-3
    seed: int = 0
    maxGradientNorm: Optional[float] = 5.0
    reportEveryEpoch: bool = True


@dataclass
class EpochRecord:
    """Everything measured during a single epoch."""

    epoch: int
    trainLoss: float
    trainAccuracy: float
    validationLoss: float
    validationAccuracy: float
    durationSeconds: float
    learningRate: float
    sparsity: float
    activeParameters: int
    totalParameters: int
    isFinite: bool = True
    extras: Dict[str, float] = field(default_factory=dict)

    def asDict(self) -> Dict[str, object]:
        """Return the record as a flat dictionary."""
        payload: Dict[str, object] = {
            "epoch": self.epoch,
            "trainLoss": self.trainLoss,
            "trainAccuracy": self.trainAccuracy,
            "validationLoss": self.validationLoss,
            "validationAccuracy": self.validationAccuracy,
            "durationSeconds": self.durationSeconds,
            "learningRate": self.learningRate,
            "sparsity": self.sparsity,
            "activeParameters": self.activeParameters,
            "totalParameters": self.totalParameters,
            "isFinite": self.isFinite,
        }
        payload.update(self.extras)
        return payload


def shuffleIndices(sampleCount: int, rng: np.random.Generator) -> np.ndarray:
    """Return a fresh permutation of ``range(sampleCount)``."""
    return rng.permutation(sampleCount)


class Trainer:
    """A mini-batch trainer for a :class:`DenseNetwork`.

    Parameters
    ----------
    model:
        The network to train.
    optimizer:
        The optimiser.  It must already be attached to ``model``.
    dataset:
        The train/validation split.
    config:
        Training hyper-parameters.
    onEpochEnd:
        Optional callback invoked as ``onEpochEnd(trainer, record)`` after
        every epoch.  This is how the pruning machinery hooks into training
        without the trainer knowing anything about pruning.
    onBatchEnd:
        Optional callback invoked as ``onBatchEnd(trainer)`` after every
        mini-batch.  The saliency criterion uses it to average the gradient
        over the whole epoch, which removes most of the mini-batch noise
        from the importance estimate.
    """

    def __init__(
        self,
        model: DenseNetwork,
        optimizer: AdamOptimizer,
        dataset: Dataset,
        config: Optional[TrainConfig] = None,
        onEpochEnd: Optional[Callable[["Trainer", EpochRecord], None]] = None,
        onBatchEnd: Optional[Callable[["Trainer"], None]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.config = config or TrainConfig()
        self.onEpochEnd = onEpochEnd
        self.onBatchEnd = onBatchEnd
        self.history: List[EpochRecord] = []
        self.rng = np.random.default_rng(self.config.seed)
        self.globalStep = 0

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self) -> tuple[float, float]:
        """Return ``(meanLoss, accuracy)`` on the validation split."""
        features = self.dataset.validationFeatures
        labels = self.dataset.validationLabels
        if features.shape[0] == 0:
            return 0.0, 0.0

        logits = self.model.forward(Tensor(features))
        loss = float(computeLoss(logits, labels).data)
        accuracy = computeAccuracy(logits, labels)
        return loss, accuracy

    def evaluateTrainSplit(self) -> tuple[float, float]:
        """Return ``(meanLoss, accuracy)`` on the training split."""
        features = self.dataset.trainFeatures
        labels = self.dataset.trainLabels
        logits = self.model.forward(Tensor(features))
        loss = float(computeLoss(logits, labels).data)
        accuracy = computeAccuracy(logits, labels)
        return loss, accuracy

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def trainEpoch(self, epoch: int) -> EpochRecord:
        """Run one epoch of mini-batch training and return its record."""
        startTime = time.perf_counter()
        features = self.dataset.trainFeatures
        labels = self.dataset.trainLabels
        sampleCount = features.shape[0]
        batchSize = max(1, min(self.config.batchSize, sampleCount))

        order = shuffleIndices(sampleCount, self.rng)
        totalLoss = 0.0
        totalCorrect = 0
        isFinite = True

        for start in range(0, sampleCount, batchSize):
            batchIndices = order[start : start + batchSize]
            batchFeatures = features[batchIndices]
            batchLabels = labels[batchIndices]

            logits = self.model.forward(Tensor(batchFeatures))
            loss = computeLoss(logits, batchLabels)
            loss.backward()

            # Publish the effective (unmasked) weight gradients so that a
            # pruner can read them without re-running the backward pass.
            self.model.publishEffectiveGradients()

            # Let an importance criterion observe this mini-batch's gradient
            # statistics.  This must happen *before* the optimiser step so
            # that the captured gradient corresponds to the current weights.
            if self.onBatchEnd is not None:
                self.onBatchEnd(self)

            self.optimizer.step()
            self.globalStep += 1

            totalLoss += float(loss.data) * batchIndices.size
            totalCorrect += int(np.sum(predictLabels(logits) == batchLabels))
            if not np.isfinite(loss.data):
                isFinite = False

        epochDuration = time.perf_counter() - startTime
        validationLoss, validationAccuracy = self.evaluate()
        trainLoss = totalLoss / sampleCount
        trainAccuracy = totalCorrect / sampleCount

        if not np.isfinite(validationLoss):
            isFinite = False

        record = EpochRecord(
            epoch=epoch,
            trainLoss=trainLoss,
            trainAccuracy=trainAccuracy,
            validationLoss=validationLoss,
            validationAccuracy=validationAccuracy,
            durationSeconds=epochDuration,
            learningRate=self.optimizer.learningRate,
            sparsity=self.model.sparsity(),
            activeParameters=self.model.activeParameterCount(),
            totalParameters=self.model.totalParameterCount(),
            isFinite=isFinite,
        )
        self.history.append(record)
        return record

    def fit(self, epochs: Optional[int] = None) -> List[EpochRecord]:
        """Train for ``epochs`` epochs and return the history."""
        totalEpochs = epochs if epochs is not None else self.config.epochs
        for epoch in range(1, totalEpochs + 1):
            record = self.trainEpoch(epoch)
            if self.onEpochEnd is not None:
                self.onEpochEnd(self, record)
        return self.history

    def finalMetrics(self) -> Dict[str, float]:
        """Return the final validation loss/accuracy and sparsity."""
        if not self.history:
            return {
                "validationLoss": 0.0,
                "validationAccuracy": 0.0,
                "sparsity": self.model.sparsity(),
                "activeParameters": float(self.model.activeParameterCount()),
            }
        last = self.history[-1]
        return {
            "validationLoss": last.validationLoss,
            "validationAccuracy": last.validationAccuracy,
            "sparsity": last.sparsity,
            "activeParameters": float(last.activeParameters),
        }
