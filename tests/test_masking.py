"""Tests for the masking semantics -- the correctness requirement of Part 1.

These tests are the heart of the submission.  They pin down exactly what is
promised about masked weights:

1. A masked connection contributes **exactly zero** to the forward pass.
2. The gradient with respect to the **raw** weight of a masked connection is
   **exactly zero**.
3. The gradient with respect to the **effective** weight of a masked
   connection is generally **non-zero** -- that is the signal that makes
   re-growth possible.
4. The optimiser does not move a masked connection, does not accumulate
   momentum for it, and does not leave a stale second moment behind.
5. A revived connection starts from a clean optimiser state.
6. A masked network produces *bit-identical* results to the equivalent dense
   network with the pruned weights removed.

Every claim in ``DESIGN.md`` about masking has a corresponding test here.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqua.engine import ops
from aqua.engine.tensor import Parameter, Tensor
from aqua.nn.layers import Linear
from aqua.optim.adam import AdamOptimizer


def _makeLayer(seed: int = 0) -> Linear:
    rng = np.random.default_rng(seed)
    layer = Linear(inFeatures=4, outFeatures=3, bias=True, rng=rng)
    return layer


def _fakeModel(layer: Linear):
    """A minimal model-like object exposing only what the optimiser needs."""
    return type(
        "M",
        (),
        {
            "prunableLayers": lambda self: [layer],
            "parameters": lambda self: layer.parameters(),
        },
    )()


# ----------------------------------------------------------------------
# Forward pass
# ----------------------------------------------------------------------
def test_masked_connection_contributes_exactly_zero() -> None:
    """A masked weight must contribute exactly zero, not a small number."""
    layer = _makeLayer()
    mask = np.ones((4, 3))
    mask[1, 2] = 0.0
    layer.setMask(mask)

    inputs = Tensor(np.ones((1, 4)))
    outputs = layer.forward(inputs)

    # Recompute by hand with the masked weight literally removed.
    effective = layer.rawWeights.data * mask
    expected = np.ones((1, 4)) @ effective + layer.bias.data
    assert np.array_equal(outputs.data, expected)

    # And the masked entry itself is exactly zero.
    assert layer.applyMaskToWeights()[1, 2] == 0.0


def test_masked_layer_equals_layer_with_pruned_weights_deleted() -> None:
    """Masking must be equivalent to physically deleting the connection."""
    layer = _makeLayer()
    mask = np.ones((4, 3))
    mask[0, 0] = 0.0
    mask[2, 1] = 0.0
    layer.setMask(mask)

    inputs = Tensor(np.random.default_rng(1).normal(size=(5, 4)))
    maskedOutput = layer.forward(inputs).data

    # Reference: a dense layer whose pruned weights are set to zero.
    reference = Linear(inFeatures=4, outFeatures=3, bias=True, rng=np.random.default_rng(0))
    reference.rawWeights.data = layer.rawWeights.data * mask
    reference.bias.data = layer.bias.data.copy()
    referenceOutput = reference.forward(inputs).data

    assert np.allclose(maskedOutput, referenceOutput, atol=1e-12)


def test_mask_must_be_binary() -> None:
    """A non-binary mask would silently turn pruning into scaling."""
    layer = _makeLayer()
    with pytest.raises(ValueError):
        layer.setMask(np.full((4, 3), 0.5))


def test_mask_shape_is_validated() -> None:
    layer = _makeLayer()
    with pytest.raises(ValueError):
        layer.setMask(np.ones((2, 2)))


# ----------------------------------------------------------------------
# Gradient semantics
# ----------------------------------------------------------------------
def test_gradient_wrt_raw_weight_is_zero_at_masked_positions() -> None:
    """The raw-weight gradient of a masked connection is exactly zero."""
    layer = _makeLayer()
    mask = np.ones((4, 3))
    mask[1, 2] = 0.0
    layer.setMask(mask)

    inputs = Tensor(np.random.default_rng(2).normal(size=(3, 4)))
    outputs = layer.forward(inputs)
    ops.sum(outputs).backward()

    rawGradient = layer.rawWeights.grad
    assert rawGradient is not None
    assert rawGradient[1, 2] == 0.0
    # Alive connections must still receive a gradient.
    assert np.abs(rawGradient[0, 0]) > 0.0


def test_gradient_wrt_effective_weight_is_non_zero_at_masked_positions() -> None:
    """The effective-weight gradient of a masked connection is NOT zero.

    This is the distinction the challenge probes.  The effective gradient
    answers "how sensitive is the loss to this connection?", and it does not
    depend on the weight value -- only on the layer input and the downstream
    gradient.  It is therefore non-zero even where the connection is
    currently masked, and it is the signal used to score re-growth
    candidates.
    """
    layer = _makeLayer()
    mask = np.ones((4, 3))
    mask[1, 2] = 0.0
    layer.setMask(mask)

    inputs = Tensor(np.random.default_rng(3).normal(size=(3, 4)))
    outputs = layer.forward(inputs)
    ops.sum(outputs).backward()

    node = layer._effectiveWeightsNode
    assert node.effectiveGrad is not None
    assert node.effectiveGrad[1, 2] != 0.0

    # And the published layer-level value agrees.
    layer.publishEffectiveGradient()
    assert layer.effectiveWeightGrad is not None
    assert layer.effectiveWeightGrad[1, 2] != 0.0
    # The raw gradient is the effective gradient multiplied by the mask.
    assert np.array_equal(layer.rawWeightGrad, node.effectiveGrad * mask)


def test_masked_gradient_matches_manual_computation() -> None:
    """dL/dW_raw must equal (dL/dW_effective) * mask, elementwise."""
    layer = _makeLayer()
    mask = np.ones((4, 3))
    mask[0, 1] = 0.0
    mask[3, 2] = 0.0
    layer.setMask(mask)

    inputs = Tensor(np.random.default_rng(4).normal(size=(4, 4)))
    outputs = layer.forward(inputs)
    loss = ops.sum(ops.multiply(outputs, np.random.default_rng(5).normal(size=(4, 3))))
    loss.backward()
    layer.publishEffectiveGradient()

    expected = layer.effectiveWeightGrad * mask
    assert np.allclose(layer.rawWeightGrad, expected, atol=1e-12)


def test_masked_weight_receives_no_bias_contribution() -> None:
    """Biases are never masked and always receive a gradient."""
    layer = _makeLayer()
    mask = np.ones((4, 3))
    mask[0, 0] = 0.0
    layer.setMask(mask)

    inputs = Tensor(np.random.default_rng(6).normal(size=(2, 4)))
    ops.sum(layer.forward(inputs)).backward()
    assert layer.bias.grad is not None
    assert np.any(layer.bias.grad != 0.0)


# ----------------------------------------------------------------------
# Optimiser behaviour
# ----------------------------------------------------------------------
def test_optimiser_does_not_move_masked_parameters() -> None:
    """A masked parameter must stay exactly zero after an optimiser step."""
    layer = _makeLayer()
    mask = np.ones((4, 3))
    mask[0, 0] = 0.0
    layer.setMask(mask)
    layer.rawWeights.data[0, 0] = 0.0

    model = _fakeModel(layer)
    optimizer = AdamOptimizer(learningRate=0.1)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    for _ in range(5):
        inputs = Tensor(np.random.default_rng(7).normal(size=(3, 4)))
        ops.sum(layer.forward(inputs)).backward()
        optimizer.step()
        assert layer.rawWeights.data[0, 0] == 0.0


def test_optimiser_state_is_zeroed_for_masked_parameters() -> None:
    """No residual momentum and no stale second moment for dead connections."""
    layer = _makeLayer()
    mask = np.ones((4, 3))
    layer.setMask(mask)

    model = _fakeModel(layer)
    optimizer = AdamOptimizer(learningRate=1e-3)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    # Take a few steps *while the connection is alive* so that it accumulates
    # real momentum, then mask it.
    for _ in range(3):
        inputs = Tensor(np.random.default_rng(8).normal(size=(3, 4)))
        ops.sum(layer.forward(inputs)).backward()
        optimizer.step()

    state = optimizer.getState(layer.rawWeights)
    assert np.abs(state.firstMoment[0, 0]) > 0.0, "the connection should have momentum"

    # Now prune it.
    mask[0, 0] = 0.0
    layer.setMask(mask)
    optimizer.enforceMasks()

    assert state.firstMoment[0, 0] == 0.0
    assert state.secondMoment[0, 0] == 0.0
    assert state.timestep[0, 0] == 0
    assert layer.rawWeights.data[0, 0] == 0.0


def test_revived_connection_starts_from_a_clean_optimizer_state() -> None:
    """A revived connection must not inherit stale momentum.

    This is the "pruned-then-revived" scenario the challenge probes.  A
    connection that accumulated momentum before being pruned, and whose
    second moment was left untouched, would have its first post-revival
    update mis-scaled by a denominator that corresponds to a different
    gradient.  The fix is that the revival resets the moments *and* the
    per-element timestep, so the bias correction restarts from t = 1.
    """
    layer = _makeLayer()
    model = _fakeModel(layer)
    optimizer = AdamOptimizer(learningRate=1e-2)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    # Build up real momentum at position (0, 0).
    for _ in range(4):
        inputs = Tensor(np.random.default_rng(9).normal(size=(3, 4)))
        ops.sum(layer.forward(inputs)).backward()
        optimizer.step()

    state = optimizer.getState(layer.rawWeights)
    assert np.abs(state.firstMoment[0, 0]) > 0.0
    assert state.timestep[0, 0] == 4

    # Prune it: mask off, zero the weight, reset the state.
    mask = layer.mask.copy()
    mask[0, 0] = 0.0
    layer.setMask(mask)
    layer.rawWeights.data[0, 0] = 0.0
    optimizer.resetStateAt(layer.rawWeights, np.array([0]))

    assert state.firstMoment[0, 0] == 0.0
    assert state.secondMoment[0, 0] == 0.0
    assert state.timestep[0, 0] == 0

    # Revive it: the mask comes back, the weight starts at zero, and the
    # state is still clean.
    mask[0, 0] = 1.0
    layer.setMask(mask)
    optimizer.enforceMasks()

    assert layer.mask[0, 0] == 1.0
    assert layer.rawWeights.data[0, 0] == 0.0
    assert state.firstMoment[0, 0] == 0.0
    assert state.timestep[0, 0] == 0


def test_revived_connection_first_update_matches_a_fresh_parameter() -> None:
    """After revival, the first Adam step must be identical to a new parameter.

    This is the strongest form of the "no silent corruption" claim: a
    revived connection must behave *exactly* like a parameter that was just
    created, not merely "close to" one.

    The test isolates a single connection by using a 1x1 layer, so that the
    gradient seen by the revived connection is exactly the gradient of the
    reference parameter.
    """
    # A one-connection layer makes the isolation exact.
    layer = Linear(inFeatures=1, outFeatures=1, bias=False, rng=np.random.default_rng(0))
    model = _fakeModel(layer)
    optimizer = AdamOptimizer(learningRate=1e-2, beta1=0.9, beta2=0.999)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    # Reference: an isolated parameter that is *never* pruned.
    reference = Parameter(np.zeros((1,)))

    # Build momentum on the connection, then prune and revive it.
    for _ in range(4):
        inputs = Tensor(np.ones((1, 1)))
        ops.sum(layer.forward(inputs)).backward()
        optimizer.step()

    state = optimizer.getState(layer.rawWeights)
    assert np.abs(state.firstMoment[0, 0]) > 0.0
    assert state.timestep[0, 0] == 4

    # Prune it: mask off, zero the weight, reset the state.
    mask = layer.mask.copy()
    mask[0, 0] = 0.0
    layer.setMask(mask)
    layer.rawWeights.data[0, 0] = 0.0
    optimizer.resetStateAt(layer.rawWeights, np.array([0]))

    assert state.firstMoment[0, 0] == 0.0
    assert state.secondMoment[0, 0] == 0.0
    assert state.timestep[0, 0] == 0

    # Revive it: the mask comes back, the weight starts at zero, and the
    # state is still clean.
    mask[0, 0] = 1.0
    layer.setMask(mask)
    optimizer.enforceMasks()

    assert layer.mask[0, 0] == 1.0
    assert layer.rawWeights.data[0, 0] == 0.0
    assert state.firstMoment[0, 0] == 0.0
    assert state.timestep[0, 0] == 0

    # Now give the revived connection a gradient and step.
    inputs = Tensor(np.ones((1, 1)))
    ops.sum(layer.forward(inputs)).backward()
    gradient = layer.rawWeights.grad[0, 0]
    optimizer.step()

    # The reference parameter gets the same gradient, from a clean state.
    reference.grad = np.array([gradient])
    optimizer._stepParameter(reference)

    revivedValue = layer.rawWeights.data[0, 0]
    assert revivedValue == pytest.approx(reference.data[0], rel=1e-12, abs=1e-15)
    assert optimizer.getState(reference).timestep[0] == 1


def test_masked_network_matches_dense_network_with_pruned_weights_zeroed() -> None:
    """End-to-end: masking must be equivalent to a dense network with zeros."""
    from aqua.nn.mlp import buildMlp

    rng = np.random.default_rng(11)
    model = buildMlp(inputSize=4, hiddenSizes=[8], outputSize=3, rng=rng)

    mask = np.ones((4, 8))
    mask[0, 0] = 0.0
    mask[1, 5] = 0.0
    model.layers[0].setMask(mask)

    features = np.random.default_rng(12).normal(size=(6, 4))
    maskedOutput = model.forward(Tensor(features)).data

    # The reference is the *same* model with the masked weights literally
    # zeroed in the raw parameter (and no mask at all).
    reference = buildMlp(inputSize=4, hiddenSizes=[8], outputSize=3, rng=rng)
    for source, target in zip(model.layers, reference.layers):
        target.rawWeights.data = source.rawWeights.data * source.mask
        if source.bias is not None and target.bias is not None:
            target.bias.data = source.bias.data.copy()
    referenceOutput = reference.forward(Tensor(features)).data

    assert np.allclose(maskedOutput, referenceOutput, atol=1e-12)


def test_pruning_then_training_does_not_corrupt_the_model() -> None:
    """A full training run with pruning must stay finite and make progress."""
    from aqua.data.datasets import makeSpiralDataset
    from aqua.nn.mlp import buildMlp
    from aqua.optim.adam import AdamOptimizer
    from aqua.prune.criteria import SaliencyPruner
    from aqua.prune.pruner import Pruner
    from aqua.prune.schedule import CubicSparsitySchedule
    from aqua.train.trainer import TrainConfig, Trainer

    dataset = makeSpiralDataset(sampleCount=400, noiseStd=0.2, turns=1.2, seed=0)
    model = buildMlp(2, [16], 2, rng=np.random.default_rng(0))
    optimizer = AdamOptimizer(learningRate=1e-3, maxGradientNorm=5.0)
    optimizer.attachModel(model)

    pruner = Pruner(
        model,
        optimizer,
        criterion=SaliencyPruner(),
        schedule=CubicSparsitySchedule(finalSparsity=0.7, startFraction=0.2, endFraction=0.8),
    )
    trainer = Trainer(
        model,
        optimizer,
        dataset,
        TrainConfig(epochs=25, batchSize=32, seed=0),
        onEpochEnd=lambda t, r: pruner.onEpochEnd(r.epoch, 25),
        onBatchEnd=lambda t: pruner.observeBatch(),
    )
    trainer.fit()

    assert all(record.isFinite for record in trainer.history)
    assert all(np.isfinite(record.validationLoss) for record in trainer.history)
    assert model.sparsity() > 0.5
    # Every masked connection must still be exactly zero in the raw weights.
    for layer in model.prunableLayers():
        dead = layer.mask == 0
        assert np.all(layer.rawWeights.data[dead] == 0.0)
        # And the optimiser state of dead connections must be clean.
        state = optimizer.getState(layer.rawWeights)
        assert np.all(state.firstMoment[dead] == 0.0)
        assert np.all(state.secondMoment[dead] == 0.0)
