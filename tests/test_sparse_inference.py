"""Tests for the sparse-aware inference path and the cost measurement."""

from __future__ import annotations

import numpy as np
import pytest

from aqua.data.datasets import makeSpiralDataset
from aqua.nn.mlp import buildMlp
from aqua.optim.adam import AdamOptimizer
from aqua.prune.criteria import SaliencyPruner
from aqua.prune.pruner import Pruner
from aqua.train.cost import measureCost, measureLatency
from aqua.train.sparse import (
    buildSparseStructure,
    denseForward,
    inferenceAccuracy,
    maskedDenseForward,
    sparseAwareForward,
)


def _makePrunedModel(sparsity: float = 0.9, seed: int = 0):
    model = buildMlp(2, [16, 16], 2, rng=np.random.default_rng(seed))
    optimizer = AdamOptimizer(learningRate=1e-3)
    optimizer.attachModel(model)
    pruner = Pruner(model, optimizer, criterion=SaliencyPruner())
    pruner.pruneTo(sparsity)
    return model


def test_sparse_aware_forward_matches_dense_forward() -> None:
    """The sparse path must be numerically equivalent to the dense path."""
    model = _makePrunedModel(0.9)
    features = np.random.default_rng(1).normal(size=(16, 2))

    dense = denseForward(model, features)
    sparse = sparseAwareForward(model, features)
    assert np.allclose(dense, sparse, atol=1e-12)


def test_masked_dense_forward_matches_dense_forward() -> None:
    """The "dense times zero" path is numerically identical to the dense path."""
    model = _makePrunedModel(0.8)
    features = np.random.default_rng(2).normal(size=(8, 2))
    assert np.allclose(
        denseForward(model, features), maskedDenseForward(model, features), atol=1e-14
    )


def test_sparse_structure_counts_only_active_connections() -> None:
    model = _makePrunedModel(0.75)
    for layer in model.layers:
        structure = buildSparseStructure(layer)
        assert structure.nonZeroCount == layer.activeParameterCount()
        assert structure.denseCount == layer.totalParameterCount()


def test_sparse_structure_gathers_only_active_inputs() -> None:
    model = _makePrunedModel(0.9)
    layer = model.layers[0]
    structure = buildSparseStructure(layer)
    weights = layer.applyMaskToWeights()
    for outputIndex in range(structure.outputCount):
        active = structure.inputIndices[outputIndex]
        expected = np.flatnonzero(weights[:, outputIndex] != 0.0)
        assert np.array_equal(active, expected)


def test_sparse_path_performs_fewer_multiply_accumulates() -> None:
    """The theoretical MAC reduction must match the achieved sparsity."""
    model = _makePrunedModel(0.9)
    dataset = makeSpiralDataset(sampleCount=200, seed=0)
    cost = measureCost(model, dataset, batchSizes=(8,), repetitions=10)

    assert cost["denseMacs"] == cost["denseTimesZeroMacs"]
    assert cost["activeMacs"] < cost["denseMacs"]
    assert cost["theoreticalMacReduction"] == pytest.approx(model.sparsity(), abs=1e-9)


def test_dense_times_zero_does_not_reduce_work() -> None:
    """The honest point: multiplying by zero performs the same MACs."""
    model = _makePrunedModel(0.95)
    dataset = makeSpiralDataset(sampleCount=200, seed=0)
    cost = measureCost(model, dataset, batchSizes=(8,), repetitions=10)
    assert cost["denseTimesZeroMacs"] == cost["denseMacs"]


def test_cost_measurement_reports_sparse_equals_dense_accuracy() -> None:
    model = _makePrunedModel(0.9)
    dataset = makeSpiralDataset(sampleCount=200, seed=0)
    cost = measureCost(model, dataset, batchSizes=(8,), repetitions=10)
    assert cost["sparseMatchesDense"]
    assert cost["sparseMaxAbsDifference"] < 1e-9
    assert cost["sparseAccuracy"] == pytest.approx(cost["denseAccuracy"])


def test_cost_measurement_reports_parameter_counts() -> None:
    model = _makePrunedModel(0.9)
    dataset = makeSpiralDataset(sampleCount=200, seed=0)
    cost = measureCost(model, dataset, batchSizes=(8,), repetitions=10)
    assert cost["totalConnections"] == model.totalParameterCount()
    assert cost["activeConnections"] == model.activeParameterCount()
    assert cost["biasParameters"] == sum(
        layer.bias.size for layer in model.layers if layer.bias is not None
    )


def test_latency_measurement_returns_positive_values() -> None:
    model = _makePrunedModel(0.9)
    features = np.random.default_rng(3).normal(size=(32, 2))
    result = measureLatency(lambda: denseForward(model, features), repetitions=20, warmup=2)
    assert result.medianSeconds > 0.0
    assert result.minimumSeconds <= result.medianSeconds
    assert len(result.samples) == 20


def test_sparse_forward_uses_no_dense_matmul_for_masked_columns() -> None:
    """A masked connection must not influence the output at all.

    Perturbing the raw weight of a masked connection must leave the sparse
    forward output completely unchanged.
    """
    model = _makePrunedModel(0.9)
    features = np.random.default_rng(4).normal(size=(8, 2))
    before = sparseAwareForward(model, features)

    layer = model.layers[0]
    dead = np.argwhere(layer.mask == 0)
    assert dead.size > 0
    layer.rawWeights.data[dead[0][0], dead[0][1]] = 1234.5

    after = sparseAwareForward(model, features)
    assert np.array_equal(before, after)
