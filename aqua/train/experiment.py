"""The experiment driver.

An experiment run is fully described by a small configuration dictionary so
that every number in the results directory can be traced back to a named
configuration plus a seed.  The driver is deterministic: the same
configuration and seed produce the same history, the same masks and the same
final metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from aqua.data.datasets import Dataset, makeSyntheticDataset
from aqua.nn.layers import DenseNetwork
from aqua.nn.mlp import buildMlp
from aqua.optim.adam import AdamOptimizer
from aqua.prune.criteria import buildCriterion
from aqua.prune.pruner import Pruner
from aqua.prune.regrowth import Regrowth
from aqua.prune.schedule import buildSchedule
from aqua.train.trainer import EpochRecord, TrainConfig, Trainer
from aqua.utils.seeding import setGlobalSeed

__all__ = ["ExperimentConfig", "ExperimentResult", "runExperiment", "DEFAULT_CONFIG"]


DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": "spiral",
    "datasetKwargs": {
        "sampleCount": 1200,
        "validationFraction": 0.25,
        "noiseStd": 0.20,
        "turns": 1.2,
    },
    "hiddenSizes": [32, 32],
    "hiddenActivation": "relu",
    "outputSize": 2,
    "epochs": 60,
    "batchSize": 64,
    "learningRate": 1e-3,
    "maxGradientNorm": 5.0,
    "criterion": "saliency",
    "targetSparsity": 0.0,
    "schedule": "cubic",
    "scheduleKwargs": {"startFraction": 0.2, "endFraction": 0.8},
    "enableRegrowth": False,
    "regrowthKwargs": {"maxCandidatesPerEvent": 256, "maxReviveFraction": 0.05},
    "seed": 0,
}


@dataclass
class ExperimentConfig:
    """A complete, serialisable description of one experiment."""

    dataset: str = "spiral"
    datasetKwargs: Dict[str, Any] = field(default_factory=dict)
    hiddenSizes: List[int] = field(default_factory=lambda: [32, 32])
    hiddenActivation: str = "relu"
    outputSize: int = 2
    epochs: int = 60
    batchSize: int = 64
    learningRate: float = 1e-3
    maxGradientNorm: Optional[float] = 5.0
    criterion: str = "saliency"
    targetSparsity: float = 0.0
    schedule: str = "cubic"
    scheduleKwargs: Dict[str, Any] = field(default_factory=dict)
    enableRegrowth: bool = False
    regrowthKwargs: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    name: str = "experiment"

    @staticmethod
    def fromDict(payload: Dict[str, Any]) -> "ExperimentConfig":
        """Build a config from a plain dictionary, filling in the defaults."""
        merged = dict(DEFAULT_CONFIG)
        merged.update(payload)
        known = {
            key: merged[key] for key in ExperimentConfig.__dataclass_fields__ if key in merged
        }
        known["scheduleKwargs"] = dict(known.get("scheduleKwargs") or {})
        known["regrowthKwargs"] = dict(known.get("regrowthKwargs") or {})
        known["datasetKwargs"] = dict(known.get("datasetKwargs") or {})
        known.setdefault("name", "experiment")
        return ExperimentConfig(**known)  # type: ignore[arg-type]

    def asDict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return {
            "dataset": self.dataset,
            "datasetKwargs": dict(self.datasetKwargs),
            "hiddenSizes": list(self.hiddenSizes),
            "hiddenActivation": self.hiddenActivation,
            "outputSize": self.outputSize,
            "epochs": self.epochs,
            "batchSize": self.batchSize,
            "learningRate": self.learningRate,
            "maxGradientNorm": self.maxGradientNorm,
            "criterion": self.criterion,
            "targetSparsity": self.targetSparsity,
            "schedule": self.schedule,
            "scheduleKwargs": dict(self.scheduleKwargs),
            "enableRegrowth": self.enableRegrowth,
            "regrowthKwargs": dict(self.regrowthKwargs),
            "seed": self.seed,
            "name": self.name,
        }


@dataclass
class ExperimentResult:
    """Everything produced by one experiment run."""

    config: ExperimentConfig
    history: List[Dict[str, Any]]
    pruningEvents: List[Dict[str, Any]]
    regrowthEvents: List[Dict[str, Any]]
    finalMetrics: Dict[str, float]
    costMetrics: Dict[str, Any]
    wallClockSeconds: float
    seed: int

    def asDict(self) -> Dict[str, Any]:
        """Return the whole result as a nested dictionary."""
        return {
            "config": self.config.asDict(),
            "seed": self.seed,
            "history": self.history,
            "pruningEvents": self.pruningEvents,
            "regrowthEvents": self.regrowthEvents,
            "finalMetrics": self.finalMetrics,
            "costMetrics": self.costMetrics,
            "wallClockSeconds": self.wallClockSeconds,
        }


def buildModel(config: ExperimentConfig, dataset: Dataset, seed: int) -> DenseNetwork:
    """Create the model described by ``config`` with a seeded initialisation."""
    generator = setGlobalSeed(seed)
    return buildMlp(
        inputSize=dataset.featureCount,
        hiddenSizes=config.hiddenSizes,
        outputSize=config.outputSize,
        hiddenActivation=config.hiddenActivation,
        outputActivation=None,
        useBias=True,
        rng=generator,
        name="mlp",
    )


def runExperiment(config: ExperimentConfig) -> ExperimentResult:
    """Run one full experiment and return its result.

    The run is deterministic given ``config``: the dataset, the model
    initialisation and the batch shuffling all derive from ``config.seed``.
    """
    from aqua.train.cost import measureCost

    startTime = time.perf_counter()

    setGlobalSeed(config.seed)
    dataset = makeSyntheticDataset(config.dataset, seed=config.seed, **config.datasetKwargs)
    model = buildModel(config, dataset, config.seed)

    optimizer = AdamOptimizer(
        learningRate=config.learningRate,
        maxGradientNorm=config.maxGradientNorm,
    )
    optimizer.attachModel(model)

    criterion = buildCriterion(config.criterion)
    schedule = buildSchedule(
        config.schedule,
        finalSparsity=config.targetSparsity,
        **config.scheduleKwargs,
    )
    pruner = Pruner(model, optimizer, criterion=criterion, schedule=schedule)

    regrowth: Optional[Regrowth] = None
    if config.enableRegrowth and config.targetSparsity > 0.0:
        regrowth = Regrowth(model, optimizer, criterion, **config.regrowthKwargs)

    trainConfig = TrainConfig(
        epochs=config.epochs,
        batchSize=config.batchSize,
        learningRate=config.learningRate,
        seed=config.seed,
        maxGradientNorm=config.maxGradientNorm,
    )

    def onEpochEnd(trainer: Trainer, record: EpochRecord) -> None:
        if regrowth is not None:
            regrowth.regrow(
                epoch=record.epoch,
                targetSparsity=schedule.targetSparsity(
                    record.epoch / max(1, config.epochs)
                ),
                budgetNeutral=True,
            )
        pruner.onEpochEnd(record.epoch, config.epochs)

    trainer = Trainer(
        model,
        optimizer,
        dataset,
        trainConfig,
        onEpochEnd=onEpochEnd,
        onBatchEnd=lambda trainer: pruner.observeBatch(),
    )
    trainer.fit()

    wallClock = time.perf_counter() - startTime
    costMetrics = measureCost(model, dataset)

    finalMetrics = {
        "trainLoss": trainer.history[-1].trainLoss if trainer.history else 0.0,
        "trainAccuracy": trainer.history[-1].trainAccuracy if trainer.history else 0.0,
        "validationLoss": trainer.history[-1].validationLoss if trainer.history else 0.0,
        "validationAccuracy": trainer.history[-1].validationAccuracy if trainer.history else 0.0,
        "achievedSparsity": model.sparsity(),
        "activeParameters": float(model.activeParameterCount()),
        "totalParameters": float(model.totalParameterCount()),
        "totalScalarParameters": float(model.parameterCount()),
    }

    return ExperimentResult(
        config=config,
        history=[record.asDict() for record in trainer.history],
        pruningEvents=[event.asDict() for event in pruner.events],
        regrowthEvents=[event.asDict() for event in (regrowth.events if regrowth else [])],
        finalMetrics=finalMetrics,
        costMetrics=costMetrics,
        wallClockSeconds=wallClock,
        seed=config.seed,
    )
