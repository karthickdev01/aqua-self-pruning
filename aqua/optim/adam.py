"""Adam optimiser implemented from scratch, with mask-aware state semantics.

Why a hand-written Adam
-----------------------
The whole point of the challenge is that the optimiser has to interact
correctly with a *structurally changing* model.  A library optimiser hides
its state behind a flat ``list[np.ndarray]`` and gives you no way to ask
"what is the momentum of this particular connection, and what should happen
to it when it is revived?".  Here the state is explicit and addressable.

The update rule
---------------
For parameter :math:`\\theta` with gradient :math:`g_t`:

.. math::

    m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t
    v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2
    \\hat{m}_t = m_t / (1 - \\beta_1^t)
    \\hat{v}_t = v_t / (1 - \\beta_2^t)
    \\theta_t = \\theta_{t-1} - \\eta \\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon)

The bias corrections divide by :math:`1 - \\beta^t`, which tends to zero as
:math:`t \\to \\infty`; at :math:`t = 0` they are exactly :math:`0` and the
first update would be identically zero without the correction.  This is why
the timestep is tracked **per parameter element** here: a connection that is
revived after being pruned must restart its bias correction from :math:`t=0`,
otherwise its first update is mis-scaled.

Masked parameters
-----------------
The optimiser is attached to a model and reads the structural masks.  After
every step it enforces two invariants:

1. ``rawWeight[mask == 0] == 0``  -- a dead connection cannot drift away
   from zero, so if it is ever revived it starts from a clean slate.
2. ``m[mask == 0] == 0`` and ``v[mask == 0] == 0``  -- a dead connection
   carries no residual momentum and no stale second moment.

Those invariants are *belt and braces* on top of the mathematical fact that
the gradient with respect to a masked raw weight is exactly zero.  Without
them, Adam's exponentially decayed moments would still nudge a dead
connection for tens of thousands of steps, and the first step after a
re-growth would be scaled by a stale ``v`` -- the classic "revived connection
silently corrupts training" bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from aqua.engine.tensor import Parameter
from aqua.nn.layers import DenseNetwork

__all__ = [
    "AdamState",
    "AdamOptimizer",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BETA1",
    "DEFAULT_BETA2",
    "DEFAULT_EPSILON",
]

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPSILON = 1e-8


@dataclass
class AdamState:
    """Per-parameter Adam moments.

    ``timestep`` is an integer array with the same shape as the parameter so
    that individual connections can be reset independently.
    """

    firstMoment: np.ndarray
    secondMoment: np.ndarray
    timestep: np.ndarray

    @staticmethod
    def create(shape: tuple[int, ...]) -> "AdamState":
        """Allocate a zeroed state for a parameter of shape ``shape``."""
        return AdamState(
            firstMoment=np.zeros(shape, dtype=np.float64),
            secondMoment=np.zeros(shape, dtype=np.float64),
            timestep=np.zeros(shape, dtype=np.int64),
        )

    def copy(self) -> "AdamState":
        """Return an independent copy of this state."""
        return AdamState(
            firstMoment=self.firstMoment.copy(),
            secondMoment=self.secondMoment.copy(),
            timestep=self.timestep.copy(),
        )


class AdamOptimizer:
    """Adam optimiser with per-element state and mask enforcement.

    Parameters
    ----------
    learningRate:
        Step size :math:`\\eta`.
    beta1:
        Exponential decay rate of the first moment.
    beta2:
        Exponential decay rate of the second moment.
    epsilon:
        Numerical stabiliser added to the denominator.
    maxGradientNorm:
        Optional global-norm gradient clipping threshold.  ``None`` disables
        clipping.  Clipping keeps the first few steps of a randomly
        initialised network from producing a huge first update.
    """

    def __init__(
        self,
        learningRate: float = DEFAULT_LEARNING_RATE,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
        maxGradientNorm: Optional[float] = None,
    ) -> None:
        if learningRate <= 0:
            raise ValueError("learningRate must be positive.")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 and beta2 must lie in [0, 1).")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")

        self.learningRate = float(learningRate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.maxGradientNorm = maxGradientNorm

        self.state: Dict[int, AdamState] = {}
        self._stateKeys: Dict[int, Parameter] = {}
        self._model: Optional[DenseNetwork] = None
        self.stepCount: int = 0
        self.lastGradientNorm: float = 0.0
        self.lastClipScale: float = 1.0
        self.lastGradientNorm: float = 0.0
        self.lastClipScale: float = 1.0
        self.lastGradientNorm: float = 0.0
        self.lastClipScale: float = 1.0

    # ------------------------------------------------------------------
    # Model attachment
    # ------------------------------------------------------------------
    def attachModel(self, model: DenseNetwork) -> None:
        """Register the model whose masks and parameters this optimiser owns."""
        self._model = model
        for parameter in model.parameters():
            self._register(parameter)

    def _register(self, parameter: Parameter) -> None:
        key = id(parameter)
        if key not in self.state:
            self.state[key] = AdamState.create(parameter.shape)
            self._stateKeys[key] = parameter

    def getState(self, parameter: Parameter) -> AdamState:
        """Return (creating if needed) the Adam state of ``parameter``."""
        self._register(parameter)
        return self.state[id(parameter)]

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------
    def _globalGradientNorm(self, parameters: Sequence[Parameter]) -> float:
        """Return the global L2 norm across every parameter gradient."""
        totalSquared = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            totalSquared += float(np.sum(parameter.grad * parameter.grad))
        return float(np.sqrt(totalSquared))

    def _clipScaleFactor(self, parameters: Sequence[Parameter]) -> float:
        """Return the multiplicative factor that clips the global gradient norm.

        Returns ``1.0`` when clipping is disabled or the norm is already below
        the threshold.  Clipping is applied *globally* (across all
        parameters) rather than per tensor, which is the standard definition
        and keeps the effective step size independent of how many parameters
        the model happens to have.
        """
        if self.maxGradientNorm is None:
            return 1.0
        norm = self._globalGradientNorm(parameters)
        if norm <= self.maxGradientNorm or norm == 0.0:
            return 1.0
        return self.maxGradientNorm / (norm + 1e-12)

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Apply one Adam update to every registered parameter."""
        if self._model is None:
            raise RuntimeError("attachModel() must be called before step().")

        parameters = self._model.parameters()
        scale = self._clipScaleFactor(parameters)
        self.lastGradientNorm = self._globalGradientNorm(parameters)
        self.lastClipScale = scale

        for parameter in parameters:
            self._stepParameter(parameter, scale)

        self.stepCount += 1
        self.enforceMasks()

    def _stepParameter(self, parameter: Parameter, scale: float = 1.0) -> None:
        """Apply the Adam update to a single parameter.

        ``scale`` is the global-norm clipping factor computed in :meth:`step`.
        """
        if parameter.grad is None:
            return
        state = self.getState(parameter)
        gradient = parameter.grad * scale

        # First moment: m <- beta1 * m + (1 - beta1) * g
        state.firstMoment = self.beta1 * state.firstMoment + (1.0 - self.beta1) * gradient
        # Second moment: v <- beta2 * v + (1 - beta2) * g^2
        state.secondMoment = (
            self.beta2 * state.secondMoment + (1.0 - self.beta2) * (gradient * gradient)
        )
        # Per-element timestep so revived connections can restart cleanly.
        state.timestep = state.timestep + 1

        biasCorrection1 = 1.0 - np.power(self.beta1, state.timestep)
        biasCorrection2 = 1.0 - np.power(self.beta2, state.timestep)

        correctedFirst = state.firstMoment / biasCorrection1
        correctedSecond = state.secondMoment / biasCorrection2

        denominator = np.sqrt(correctedSecond) + self.epsilon
        parameter.data = parameter.data - self.learningRate * correctedFirst / denominator

    # ------------------------------------------------------------------
    # Mask enforcement
    # ------------------------------------------------------------------
    def enforceMasks(self) -> None:
        """Enforce the mask invariants on every prunable parameter.

        * dead connections have exactly zero raw weight,
        * dead connections have exactly zero first and second moment,
        * dead connections keep their timestep frozen (so that a revived
          connection restarts bias correction from zero).
        """
        if self._model is None:
            return
        for layer in self._model.prunableLayers():
            dead = layer.mask == 0
            if not np.any(dead):
                continue
            parameter = layer.rawWeights
            state = self.getState(parameter)

            parameter.data[dead] = 0.0
            state.firstMoment[dead] = 0.0
            state.secondMoment[dead] = 0.0
            state.timestep[dead] = 0

    # ------------------------------------------------------------------
    # State surgery for pruning and re-growth
    # ------------------------------------------------------------------
    def resetStateAt(self, parameter: Parameter, indices: np.ndarray) -> None:
        """Reset the Adam state of ``parameter`` at the given flat ``indices``.

        Used when connections are pruned: their momentum and second moment
        are discarded so that a later re-growth starts from a clean state.
        """
        if indices.size == 0:
            return
        state = self.getState(parameter)
        flatFirst = state.firstMoment.reshape(-1)
        flatSecond = state.secondMoment.reshape(-1)
        flatTimestep = state.timestep.reshape(-1)
        flatFirst[indices] = 0.0
        flatSecond[indices] = 0.0
        flatTimestep[indices] = 0

    def restoreStateAt(
        self,
        parameter: Parameter,
        indices: np.ndarray,
        snapshot: Optional[AdamState] = None,
    ) -> None:
        """Restore Adam state at ``indices``, optionally from a ``snapshot``.

        Two policies are supported:

        ``snapshot=None`` (default)
            *Fresh start.*  The moments and the timestep are zeroed, so the
            revived connection behaves exactly like a brand new parameter.
            This is the policy used by :class:`aqua.prune.regrowth.Regrowth`
            and is the safe default.

        ``snapshot`` provided
            *Warm restore.*  The moments captured at prune time are put back,
            so the connection resumes with the momentum it had.  This is
            useful when a connection is temporarily masked (for example while
            its layer is being restructured) and must resume exactly where it
            left off.
        """
        if indices.size == 0:
            return
        state = self.getState(parameter)
        flatFirst = state.firstMoment.reshape(-1)
        flatSecond = state.secondMoment.reshape(-1)
        flatTimestep = state.timestep.reshape(-1)

        if snapshot is None:
            flatFirst[indices] = 0.0
            flatSecond[indices] = 0.0
            flatTimestep[indices] = 0
            return

        sourceFirst = snapshot.firstMoment.reshape(-1)[indices]
        sourceSecond = snapshot.secondMoment.reshape(-1)[indices]
        sourceTimestep = snapshot.timestep.reshape(-1)[indices]
        flatFirst[indices] = sourceFirst
        flatSecond[indices] = sourceSecond
        flatTimestep[indices] = sourceTimestep

    def snapshotState(self, parameter: Parameter) -> AdamState:
        """Return an independent copy of the Adam state of ``parameter``."""
        return self.getState(parameter).copy()

    def stateNorm(self, parameter: Parameter) -> float:
        """Return the L2 norm of the first moment (useful for diagnostics)."""
        state = self.getState(parameter)
        return float(np.linalg.norm(state.firstMoment))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"AdamOptimizer(lr={self.learningRate}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.epsilon})"
        )
