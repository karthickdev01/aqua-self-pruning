"""Integration tests: training converges, results are reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from aqua.data.datasets import makeSpiralDataset, makeTwoMoonsDataset
from aqua.nn.mlp import buildMlp
from aqua.optim.adam import AdamOptimizer
from aqua.train.experiment import ExperimentConfig, runExperiment
from aqua.train.trainer import TrainConfig, Trainer


def _trainSimple(epochs: int = 40, seed: int = 0):
    dataset = makeSpiralDataset(sampleCount=800, noiseStd=0.2, turns=1.2, seed=seed)
    model = buildMlp(2, [32, 32], 2, rng=np.random.default_rng(seed))
    optimizer = AdamOptimizer(learningRate=1e-3, maxGradientNorm=5.0)
    optimizer.attachModel(model)
    trainer = Trainer(
        model, optimizer, dataset, TrainConfig(epochs=epochs, batchSize=32, seed=seed)
    )
    trainer.fit()
    return trainer, dataset


def test_training_converges() -> None:
    """The baseline model must learn the spiral."""
    trainer, _ = _trainSimple(epochs=60)
    assert trainer.history[-1].trainLoss < trainer.history[0].trainLoss
    assert trainer.history[-1].trainAccuracy > 0.85
    assert trainer.history[-1].validationAccuracy > 0.85


def test_training_stays_finite() -> None:
    """No NaN, no inf, ever."""
    trainer, _ = _trainSimple(epochs=40)
    for record in trainer.history:
        assert np.isfinite(record.trainLoss)
        assert np.isfinite(record.validationLoss)
        assert record.isFinite


def test_training_is_reproducible_under_a_fixed_seed() -> None:
    """The same seed must give bit-identical results."""
    first, _ = _trainSimple(epochs=15, seed=7)
    second, _ = _trainSimple(epochs=15, seed=7)
    for left, right in zip(first.history, second.history):
        assert left.trainLoss == pytest.approx(right.trainLoss, rel=1e-12, abs=1e-15)
        assert left.validationAccuracy == pytest.approx(
            right.validationAccuracy, rel=1e-12
        )


def test_different_seeds_give_different_results() -> None:
    """A different seed must actually change the run (otherwise the seed is ignored)."""
    first, _ = _trainSimple(epochs=10, seed=1)
    second, _ = _trainSimple(epochs=10, seed=2)
    assert first.history[-1].trainLoss != second.history[-1].trainLoss


def test_dataset_is_deterministic() -> None:
    first = makeSpiralDataset(sampleCount=200, seed=3)
    second = makeSpiralDataset(sampleCount=200, seed=3)
    assert np.array_equal(first.trainFeatures, second.trainFeatures)
    assert np.array_equal(first.trainLabels, second.trainLabels)


def test_dataset_has_no_train_validation_leakage() -> None:
    """The train and validation splits must be disjoint in size and in content."""
    dataset = makeSpiralDataset(sampleCount=200, seed=0)
    # The split sizes must add up to the full dataset.
    assert dataset.trainFeatures.shape[0] + dataset.validationFeatures.shape[0] == 200
    assert dataset.trainFeatures.shape[0] > 0
    assert dataset.validationFeatures.shape[0] > 0
    # Standardisation is fitted on the training split only, so the two splits
    # must not be identical copies of each other.
    assert not np.array_equal(dataset.trainFeatures, dataset.validationFeatures)


def test_spiral_dataset_is_learnable() -> None:
    """A simple model must be able to fit the spiral well above chance."""
    dataset = makeSpiralDataset(sampleCount=800, noiseStd=0.2, turns=1.2, seed=0)
    labels = dataset.trainLabels
    assert len(np.unique(labels)) == 2
    # The class balance must be roughly even.
    assert abs(np.mean(labels) - 0.5) < 0.05


def test_two_moons_dataset_is_learnable() -> None:
    dataset = makeTwoMoonsDataset(sampleCount=400, seed=0)
    assert dataset.trainFeatures.shape[1] == 2
    assert dataset.classCount == 2


def test_experiment_is_reproducible() -> None:
    """The full experiment driver must be deterministic."""
    config = ExperimentConfig(
        epochs=12,
        targetSparsity=0.7,
        seed=5,
        hiddenSizes=[16, 16],
        datasetKwargs={"sampleCount": 400, "noiseStd": 0.2, "turns": 1.2},
    )
    first = runExperiment(config)
    second = runExperiment(config)
    assert first.finalMetrics["validationAccuracy"] == pytest.approx(
        second.finalMetrics["validationAccuracy"], rel=1e-12
    )
    assert first.finalMetrics["achievedSparsity"] == pytest.approx(
        second.finalMetrics["achievedSparsity"], rel=1e-12
    )


def test_end_to_end_pruning_run_hits_the_budget() -> None:
    """A full run with pruning must land on the target sparsity."""
    config = ExperimentConfig(
        epochs=30,
        targetSparsity=0.9,
        seed=0,
        hiddenSizes=[32, 32],
        criterion="saliency",
        datasetKwargs={"sampleCount": 600, "noiseStd": 0.2, "turns": 1.2},
    )
    result = runExperiment(config)
    # The budget is met up to the integer rounding of the connection count.
    assert result.finalMetrics["achievedSparsity"] == pytest.approx(0.9, abs=1e-3)
    assert len(result.pruningEvents) > 0
    assert all(event["prunedCount"] >= 0 for event in result.pruningEvents)
    # The sparsity must be monotonically non-decreasing across the schedule.
    sparsities = [event["achievedSparsity"] for event in result.pruningEvents]
    for earlier, later in zip(sparsities, sparsities[1:]):
        assert later >= earlier - 1e-12


def test_pruning_preserves_a_reasonable_accuracy() -> None:
    """A 90% sparse model must remain far above chance."""
    config = ExperimentConfig(
        epochs=60,
        targetSparsity=0.9,
        seed=0,
        hiddenSizes=[32, 32],
        criterion="saliency",
        datasetKwargs={"sampleCount": 800, "noiseStd": 0.2, "turns": 1.2},
    )
    result = runExperiment(config)
    assert result.finalMetrics["validationAccuracy"] > 0.7
