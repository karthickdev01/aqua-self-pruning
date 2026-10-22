"""Tests for the pruning machinery: budget, determinism, masks, regrowth."""

from __future__ import annotations

import numpy as np
import pytest

from aqua.engine import ops
from aqua.engine.tensor import Tensor
from aqua.nn.mlp import buildMlp
from aqua.optim.adam import AdamOptimizer
from aqua.prune.allocator import MaskAllocator
from aqua.prune.criteria import MagnitudePruner, SaliencyPruner, SignedTaylorPruner
from aqua.prune.pruner import Pruner
from aqua.prune.regrowth import Regrowth
from aqua.prune.schedule import ConstantSparsitySchedule, CubicSparsitySchedule


def _makeModel(seed: int = 0, hiddenSizes: list[int] | None = None):
    rng = np.random.default_rng(seed)
    return buildMlp(4, hiddenSizes or [8, 8], 3, rng=rng)


def _makePruner(model, criterion=None, schedule=None):
    optimizer = AdamOptimizer(learningRate=1e-3)
    optimizer.attachModel(model)
    pruner = Pruner(
        model,
        optimizer,
        criterion=criterion if criterion is not None else MagnitudePruner(),
        schedule=schedule,
    )
    return pruner, optimizer


# ----------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------
def test_cubic_schedule_reaches_final_sparsity() -> None:
    schedule = CubicSparsitySchedule(finalSparsity=0.9, startFraction=0.2, endFraction=0.8)
    assert schedule.targetSparsity(0.0) == 0.0
    assert schedule.targetSparsity(0.2) == 0.0
    assert schedule.targetSparsity(0.8) == pytest.approx(0.9)
    assert schedule.targetSparsity(1.0) == pytest.approx(0.9)


def test_cubic_schedule_is_monotonically_increasing() -> None:
    schedule = CubicSparsitySchedule(finalSparsity=0.95, startFraction=0.2, endFraction=0.8)
    values = [schedule.targetSparsity(index / 100.0) for index in range(0, 101)]
    inside = [value for index, value in enumerate(values) if 0.2 <= index / 100.0 <= 0.8]
    for earlier, later in zip(inside, inside[1:]):
        assert later >= earlier - 1e-12


def test_cubic_schedule_window_is_closed() -> None:
    schedule = CubicSparsitySchedule(finalSparsity=0.9, startFraction=0.2, endFraction=0.8)
    assert schedule.shouldPrune(0.2)
    assert schedule.shouldPrune(0.8)
    assert not schedule.shouldPrune(0.19)
    assert not schedule.shouldPrune(0.81)


def test_constant_schedule_is_one_shot() -> None:
    schedule = ConstantSparsitySchedule(finalSparsity=0.9)
    assert schedule.shouldPrune(0.0)
    # A one-shot schedule must never fire twice.
    assert not schedule.shouldPrune(0.5)
    assert not schedule.shouldPrune(0.9)
    assert schedule.targetSparsity(0.0) == pytest.approx(0.9)


def test_schedule_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        CubicSparsitySchedule(finalSparsity=1.0)
    with pytest.raises(ValueError):
        CubicSparsitySchedule(finalSparsity=0.5, startFraction=0.8, endFraction=0.2)
    with pytest.raises(ValueError):
        CubicSparsitySchedule(finalSparsity=0.5, initialSparsity=0.9)


# ----------------------------------------------------------------------
# Allocator
# ----------------------------------------------------------------------
def test_allocator_hits_the_budget_exactly() -> None:
    model = _makeModel()
    allocator = MaskAllocator(model)
    scores = {
        id(layer): np.random.default_rng(0).random(layer.mask.shape) for layer in model.layers
    }

    allocation = allocator.allocate(scores, 0.9)
    total = allocator.totalConnections()
    expectedActive = int(round(total * 0.1))
    assert allocation.prunedCount == total - expectedActive
    assert model.activeParameterCount() == expectedActive
    assert model.sparsity() == pytest.approx(0.9, abs=1e-9)


def test_allocator_is_monotone() -> None:
    """Pruning to a lower sparsity must do nothing."""
    model = _makeModel()
    allocator = MaskAllocator(model)
    scores = {
        id(layer): np.random.default_rng(0).random(layer.mask.shape) for layer in model.layers
    }
    allocator.allocate(scores, 0.5)
    activeAfterFirst = model.activeParameterCount()
    allocator.allocate(scores, 0.2)
    assert model.activeParameterCount() == activeAfterFirst


def test_allocator_removes_the_least_important_connections() -> None:
    model = _makeModel()
    allocator = MaskAllocator(model)
    layer = model.layers[0]
    score = np.zeros(layer.mask.shape)
    score[0, 0] = 1.0  # the single most important connection
    scores = {id(layer): score}
    for other in model.layers[1:]:
        scores[id(other)] = np.zeros(other.mask.shape)

    allocator.allocate(scores, 0.5)
    # The high-score connection must survive.
    assert layer.mask[0, 0] == 1.0


def test_allocator_is_deterministic_under_ties() -> None:
    """Equal scores must break ties identically on every run."""
    masks = []
    for _ in range(3):
        model = _makeModel()
        allocator = MaskAllocator(model)
        scores = {id(layer): np.zeros(layer.mask.shape) for layer in model.layers}
        allocator.allocate(scores, 0.5)
        masks.append([layer.mask.copy() for layer in model.layers])
    for other in masks[1:]:
        for firstMask, otherMask in zip(masks[0], other):
            assert np.array_equal(firstMask, otherMask)


def test_allocator_sparsity_is_global_not_per_layer() -> None:
    """The budget must be global: layers may end up with different sparsity."""
    model = _makeModel()
    allocator = MaskAllocator(model)
    scores = {
        id(layer): np.random.default_rng(layer.outFeatures).random(layer.mask.shape)
        for layer in model.layers
    }
    allocator.allocate(scores, 0.5)
    sparsities = [layer.sparsity() for layer in model.layers]
    assert len(set(round(value, 6) for value in sparsities)) > 1


# ----------------------------------------------------------------------
# Pruner
# ----------------------------------------------------------------------
def test_pruner_hits_the_target_sparsity() -> None:
    model = _makeModel()
    pruner, _ = _makePruner(model, schedule=CubicSparsitySchedule(finalSparsity=0.9))
    allocation = pruner.pruneTo(0.9)
    assert allocation.prunedCount > 0
    assert model.sparsity() == pytest.approx(0.9, abs=1e-9)


def test_pruner_zeroes_the_pruned_weights() -> None:
    model = _makeModel()
    pruner, _ = _makePruner(model)
    pruner.pruneTo(0.7)
    for layer in model.layers:
        dead = layer.mask == 0
        assert np.all(layer.rawWeights.data[dead] == 0.0)


def test_pruner_resets_the_optimizer_state_of_pruned_connections() -> None:
    model = _makeModel()
    pruner, optimizer = _makePruner(model)
    # Give the weights some momentum first.
    for _ in range(3):
        inputs = Tensor(np.random.default_rng(0).normal(size=(4, 4)))
        ops.sum(model.forward(inputs)).backward()
        optimizer.step()
    pruner.pruneTo(0.7)
    for layer in model.layers:
        state = optimizer.getState(layer.rawWeights)
        dead = layer.mask == 0
        assert np.all(state.firstMoment[dead] == 0.0)
        assert np.all(state.secondMoment[dead] == 0.0)
        assert np.all(state.timestep[dead] == 0)


def test_pruner_is_idempotent() -> None:
    model = _makeModel()
    pruner, _ = _makePruner(model)
    pruner.pruneTo(0.8)
    active = model.activeParameterCount()
    pruner.pruneTo(0.8)
    assert model.activeParameterCount() == active


def test_pruner_does_not_prune_biases() -> None:
    model = _makeModel()
    pruner, _ = _makePruner(model)
    pruner.pruneTo(0.9)
    for layer in model.layers:
        # The bias tensor carries no mask at all -- biases are never pruned.
        assert not hasattr(layer, "biasMask")
        assert layer.bias is not None
        assert layer.bias.data.size == layer.outFeatures


def test_pruner_event_records_the_achieved_sparsity() -> None:
    model = _makeModel()
    schedule = CubicSparsitySchedule(finalSparsity=0.9, startFraction=0.2, endFraction=0.8)
    pruner, _ = _makePruner(model, schedule=schedule)
    event = pruner.onEpochEnd(epoch=8, totalEpochs=10)
    assert event is not None
    assert event.achievedSparsity == pytest.approx(0.9, abs=1e-9)
    assert event.prunedCount > 0
    assert event.activeParameters == model.activeParameterCount()


def test_pruner_does_nothing_outside_the_schedule_window() -> None:
    model = _makeModel()
    schedule = CubicSparsitySchedule(finalSparsity=0.9, startFraction=0.2, endFraction=0.8)
    pruner, _ = _makePruner(model, schedule=schedule)
    assert pruner.onEpochEnd(epoch=1, totalEpochs=10) is None
    assert model.sparsity() == 0.0


# ----------------------------------------------------------------------
# Criteria
# ----------------------------------------------------------------------
def test_magnitude_criterion_scores_absolute_effective_weights() -> None:
    model = _makeModel()
    layer = model.layers[0]
    score = MagnitudePruner().score(layer)
    assert np.allclose(score, np.abs(layer.applyMaskToWeights()))


def test_saliency_criterion_uses_the_effective_gradient() -> None:
    model = _makeModel()
    layer = model.layers[0]
    inputs = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
    ops.sum(model.forward(inputs)).backward()
    model.publishEffectiveGradients()

    pruner = SaliencyPruner()
    pruner.accumulate(layer)
    expected = np.abs(layer.applyMaskToWeights() * layer.effectiveWeightGrad)
    assert np.allclose(pruner.score(layer), expected)


def test_saliency_accumulator_averages_over_batches() -> None:
    model = _makeModel()
    layer = model.layers[0]
    pruner = SaliencyPruner()

    firstGradient = None
    for _ in range(3):
        inputs = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
        ops.sum(model.forward(inputs)).backward()
        model.publishEffectiveGradients()
        if firstGradient is None:
            firstGradient = layer.effectiveWeightGrad.copy()
        pruner.accumulate(layer)

    assert pruner.observationCount(layer) == 3
    # With identical gradients the accumulated mean must equal the single score.
    single = np.abs(layer.applyMaskToWeights() * firstGradient)
    assert np.allclose(pruner.score(layer), single, atol=1e-12)


def test_saliency_accumulator_reset() -> None:
    model = _makeModel()
    layer = model.layers[0]
    pruner = SaliencyPruner()
    inputs = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
    ops.sum(model.forward(inputs)).backward()
    model.publishEffectiveGradients()
    pruner.accumulate(layer)
    assert pruner.observationCount(layer) == 1
    pruner.reset()
    assert pruner.observationCount(layer) == 0


def test_signed_taylor_criterion_is_the_negated_product() -> None:
    model = _makeModel()
    layer = model.layers[0]
    inputs = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
    ops.sum(model.forward(inputs)).backward()
    model.publishEffectiveGradients()
    pruner = SignedTaylorPruner()
    pruner.accumulate(layer)
    expected = -(layer.applyMaskToWeights() * layer.effectiveWeightGrad)
    assert np.allclose(pruner.score(layer), expected)


# ----------------------------------------------------------------------
# Regrowth
# ----------------------------------------------------------------------
def test_regrowth_revives_connections_and_keeps_the_budget() -> None:
    """A re-growth event must swap connections while preserving the budget."""
    model = _makeModel()
    optimizer = AdamOptimizer(learningRate=1e-3)
    optimizer.attachModel(model)
    criterion = SaliencyPruner()
    pruner = Pruner(model, optimizer, criterion=criterion)
    pruner.pruneTo(0.9)

    prunedCount = model.totalParameterCount() - model.activeParameterCount()
    assert prunedCount > 0
    activeBefore = model.activeParameterCount()

    # A gradient is needed to score the candidates.  Accumulate a few
    # mini-batches so the criterion has an averaged gradient to work with.
    for _ in range(3):
        inputs = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
        ops.sum(model.forward(inputs)).backward()
        model.publishEffectiveGradients()
        for layer in model.layers:
            criterion.accumulate(layer)

    regrowth = Regrowth(
        model, optimizer, criterion, maxCandidatesPerEvent=64, maxReviveFraction=0.5
    )
    event = regrowth.regrow(epoch=1, targetSparsity=0.9, budgetNeutral=True)

    assert event.revivedCount > 0
    assert event.prunedCount == event.revivedCount, "the swap must be budget neutral"
    # The active count is unchanged, so the sparsity is exactly preserved.
    assert model.activeParameterCount() == activeBefore
    assert model.sparsity() == pytest.approx(0.9, abs=1e-9)


def test_regrowth_resets_the_optimizer_state_of_revived_connections() -> None:
    """A revived connection must start from a clean Adam state and zero weight."""
    model = _makeModel()
    optimizer = AdamOptimizer(learningRate=1e-3)
    optimizer.attachModel(model)
    criterion = SaliencyPruner()
    pruner = Pruner(model, optimizer, criterion=criterion)
    pruner.pruneTo(0.9)

    inputs = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
    ops.sum(model.forward(inputs)).backward()
    model.publishEffectiveGradients()
    for layer in model.layers:
        criterion.accumulate(layer)

    # Record which connections are dead before the swap.
    deadBefore = {
        id(layer): set(np.flatnonzero((layer.mask == 0).reshape(-1))) for layer in model.layers
    }

    regrowth = Regrowth(
        model, optimizer, criterion, maxCandidatesPerEvent=64, maxReviveFraction=0.5
    )
    regrowth.regrow(epoch=1, targetSparsity=0.9, budgetNeutral=True)

    for layer in model.layers:
        state = optimizer.getState(layer.rawWeights)
        aliveFlat = layer.mask.reshape(-1) == 1
        # Every connection that was dead before the swap and is alive now
        # must have a clean optimiser state.
        revived = np.array(
            [
                index
                for index in range(layer.mask.size)
                if aliveFlat[index] and index in deadBefore[id(layer)]
            ],
            dtype=np.int64,
        )
        if revived.size:
            assert np.all(state.firstMoment.reshape(-1)[revived] == 0.0)
            assert np.all(state.secondMoment.reshape(-1)[revived] == 0.0)
            assert np.all(state.timestep.reshape(-1)[revived] == 0)
            # A revived connection starts from exactly zero.
            assert np.all(layer.rawWeights.data.reshape(-1)[revived] == 0.0)


def test_regrowth_is_budget_neutral() -> None:
    """The swap must never change the number of active connections."""
    model = _makeModel()
    optimizer = AdamOptimizer(learningRate=1e-3)
    optimizer.attachModel(model)
    criterion = SaliencyPruner()
    pruner = Pruner(model, optimizer, criterion=criterion)
    pruner.pruneTo(0.9)

    inputs = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
    ops.sum(model.forward(inputs)).backward()
    model.publishEffectiveGradients()
    for layer in model.layers:
        criterion.accumulate(layer)

    activeBefore = model.activeParameterCount()
    sparsityBefore = model.sparsity()

    regrowth = Regrowth(
        model, optimizer, criterion, maxCandidatesPerEvent=1000, maxReviveFraction=1.0
    )
    regrowth.regrow(epoch=1, targetSparsity=0.9, budgetNeutral=True)
    assert model.activeParameterCount() == activeBefore
    assert model.sparsity() == pytest.approx(sparsityBefore, abs=1e-12)
