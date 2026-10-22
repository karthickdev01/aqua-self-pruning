"""Tests for the Adam optimiser."""

from __future__ import annotations

import numpy as np
import pytest

from aqua.engine.tensor import Parameter
from aqua.optim.adam import AdamOptimizer, AdamState


class _FakeModel:
    """A minimal stand-in for a model, used to unit-test the optimiser."""

    def __init__(self, parameters: list[Parameter]) -> None:
        self._parameters = parameters

    def parameters(self) -> list[Parameter]:
        return self._parameters

    def prunableLayers(self) -> list:
        return []


def test_adam_first_step_matches_the_closed_form() -> None:
    """The first update must equal -lr * g / (sqrt(g^2) + eps)."""
    parameter = Parameter(np.array([0.0]))
    model = _FakeModel([parameter])
    optimizer = AdamOptimizer(learningRate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    gradient = 0.5
    parameter.grad = np.array([gradient])
    optimizer.step()

    expected = -0.1 * gradient / (np.sqrt(gradient**2) + 1e-8)
    assert parameter.data[0] == pytest.approx(expected, rel=1e-10)


def test_adam_bias_correction_is_applied() -> None:
    """Without bias correction the first step would be identically zero."""
    parameter = Parameter(np.array([0.0]))
    model = _FakeModel([parameter])
    optimizer = AdamOptimizer(learningRate=0.1, beta1=0.5, beta2=0.5, epsilon=1e-8)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    parameter.grad = np.array([1.0])
    optimizer.step()

    # m1 = 0.5 * 1 = 0.5 ; correction 1/(1-0.5) = 2  =>  m_hat = 1
    # v1 = 0.5 * 1 = 0.5 ; correction 1/(1-0.5) = 2  =>  v_hat = 1
    # step = -lr * 1 / (1 + eps)
    expected = -0.1 / (1.0 + 1e-8)
    assert parameter.data[0] == pytest.approx(expected, rel=1e-9)
    assert abs(parameter.data[0]) > 1e-6, "the update must not vanish"


def test_adam_second_moment_uses_the_squared_gradient() -> None:
    parameter = Parameter(np.array([0.0, 0.0]))
    model = _FakeModel([parameter])
    optimizer = AdamOptimizer(learningRate=0.1, beta1=0.0, beta2=0.0, epsilon=0.0)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    parameter.grad = np.array([3.0, -4.0])
    optimizer.step()

    # beta1 = beta2 = 0 => m = g, v = g^2, corrections are 1.
    # step = -lr * g / |g| = -lr * sign(g)
    assert parameter.data[0] == pytest.approx(-0.1, rel=1e-12)
    assert parameter.data[1] == pytest.approx(0.1, rel=1e-12)


def test_adam_state_tracks_first_and_second_moments() -> None:
    parameter = Parameter(np.zeros((2,)))
    model = _FakeModel([parameter])
    optimizer = AdamOptimizer(learningRate=0.1, beta1=0.9, beta2=0.999)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    parameter.grad = np.array([1.0, 2.0])
    optimizer.step()
    state = optimizer.getState(parameter)
    assert np.allclose(state.firstMoment, np.array([0.1, 0.2]))
    assert np.allclose(state.secondMoment, np.array([0.001, 0.004]))
    assert np.all(state.timestep == 1)

    parameter.grad = np.array([1.0, 2.0])
    optimizer.step()
    state = optimizer.getState(parameter)
    assert np.allclose(state.firstMoment, np.array([0.19, 0.38]))
    assert np.all(state.timestep == 2)


def test_adam_per_element_timestep_is_independent() -> None:
    """Different elements must have independent bias corrections."""
    parameter = Parameter(np.zeros((2,)))
    model = _FakeModel([parameter])
    optimizer = AdamOptimizer(learningRate=0.1)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    state = optimizer.getState(parameter)
    state.timestep = np.array([5, 1])

    parameter.grad = np.array([1.0, 1.0])
    optimizer.step()

    state = optimizer.getState(parameter)
    assert state.timestep[0] == 6
    assert state.timestep[1] == 2


def test_adam_skips_parameters_without_a_gradient() -> None:
    parameter = Parameter(np.array([1.0]))
    model = _FakeModel([parameter])
    optimizer = AdamOptimizer(learningRate=0.1)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    optimizer.step()
    assert parameter.data[0] == 1.0


def test_adam_global_norm_clipping() -> None:
    """Clipping must scale the *global* gradient norm to the threshold."""
    parameter = Parameter(np.zeros((2,)))
    model = _FakeModel([parameter])
    optimizer = AdamOptimizer(learningRate=0.1, maxGradientNorm=1.0)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    parameter.grad = np.array([3.0, 4.0])  # norm 5
    optimizer.step()

    assert optimizer.lastGradientNorm == pytest.approx(5.0)
    assert optimizer.lastClipScale == pytest.approx(0.2, rel=1e-9)


def test_adam_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError):
        AdamOptimizer(learningRate=0.0)
    with pytest.raises(ValueError):
        AdamOptimizer(learningRate=-1e-3)
    with pytest.raises(ValueError):
        AdamOptimizer(beta1=1.0)
    with pytest.raises(ValueError):
        AdamOptimizer(beta2=1.5)
    with pytest.raises(ValueError):
        AdamOptimizer(epsilon=-1e-8)


def test_step_without_a_model_raises() -> None:
    optimizer = AdamOptimizer()
    with pytest.raises(RuntimeError):
        optimizer.step()


def test_state_snapshot_and_restore() -> None:
    parameter = Parameter(np.zeros((3,)))
    model = _FakeModel([parameter])
    optimizer = AdamOptimizer(learningRate=0.1)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    parameter.grad = np.array([1.0, 2.0, 3.0])
    optimizer.step()
    snapshot = optimizer.snapshotState(parameter)

    parameter.grad = np.array([0.0, 0.0, 0.0])
    optimizer.step()

    optimizer.restoreStateAt(parameter, np.array([0]), snapshot)
    state = optimizer.getState(parameter)
    assert state.firstMoment[0] == pytest.approx(snapshot.firstMoment[0])
    assert state.timestep[0] == snapshot.timestep[0]


def test_restore_state_without_snapshot_is_a_fresh_start() -> None:
    parameter = Parameter(np.zeros((2,)))
    model = _FakeModel([parameter])
    optimizer = AdamOptimizer(learningRate=0.1)
    optimizer.attachModel(model)  # type: ignore[arg-type]

    parameter.grad = np.array([1.0, 1.0])
    optimizer.step()
    optimizer.resetStateAt(parameter, np.array([0, 1]))

    state = optimizer.getState(parameter)
    assert np.all(state.firstMoment == 0.0)
    assert np.all(state.secondMoment == 0.0)
    assert np.all(state.timestep == 0)
