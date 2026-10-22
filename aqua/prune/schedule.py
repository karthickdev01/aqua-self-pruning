"""Pruning schedules.

Cubic sparsity ramp
--------------------
The canonical iterative-magnitude-pruning schedule (Luo et al., 2019) ramps
the sparsity with a cubic polynomial in the *fraction of training completed*:

.. math::

    s(t) = s_{\\text{final}}
           + (s_{\\text{initial}} - s_{\\text{final}})
             \\left(1 - \\left(\\frac{t - t_0}{t_1 - t_0}\\right)^3\\right)^3

where :math:`t` is the training step, :math:`t_0` is the step at which
pruning starts and :math:`t_1` the step at which the final sparsity is
reached.  The ramp is *slow at first* (the cubic of a cubic makes the
derivative vanish at :math:`t_0`) and *fast in the middle*, which gives the
network a long period of low sparsity to recover from each pruning shock
before the next one arrives.

Why a gradual ramp beats one-shot pruning
------------------------------------------
One-shot pruning removes everything at the end of training.  The network
then has no time to adapt: the loss jumps and the final accuracy reflects a
model that has never been trained with the structure it will be deployed
with.  A gradual ramp turns the single shock into many small shocks, each
followed by recovery.  The trade-off is that the *intermediate* models are
trained with less capacity than the final one, so the final model has
effectively been trained under a moving sparsity constraint; this is
mildly harmful when the budget is generous and clearly beneficial when the
budget is tight.

Parameterisation
----------------
The schedule is expressed in terms of *fractions of total training* rather
than absolute epochs, so the same schedule works for any epoch count:

``startFraction``
    Fraction of training after which pruning begins (0.2 = after 20% of the
    epochs).
``endFraction``
    Fraction of training at which the final sparsity is reached (0.8).
``initialSparsity``
    Sparsity at ``startFraction`` (0.0 by default).
``finalSparsity``
    The target sparsity (e.g. 0.9).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PruningSchedule",
    "CubicSparsitySchedule",
    "ConstantSparsitySchedule",
    "buildSchedule",
]


@dataclass
class PruningSchedule:
    """Base class for pruning schedules."""

    def targetSparsity(self, progress: float) -> float:
        """Return the target sparsity at training progress ``progress`` in [0, 1]."""
        raise NotImplementedError

    def shouldPrune(self, progress: float) -> bool:
        """Whether a pruning event should happen at this progress."""
        raise NotImplementedError

    def describe(self) -> str:
        """Return a one-line description of the schedule."""
        raise NotImplementedError


class CubicSparsitySchedule(PruningSchedule):
    """The cubic sparsity ramp described in the module docstring.

    Parameters
    ----------
    finalSparsity:
        The sparsity to reach at ``endFraction`` (e.g. 0.9 for 90%).
    startFraction:
        Fraction of training at which pruning starts.
    endFraction:
        Fraction of training at which the final sparsity is reached.
    initialSparsity:
        Sparsity at ``startFraction``.  Zero by default, i.e. the network
        starts dense.
    pruneFrequency:
        Kept for interface compatibility.  The budget is recomputed from the
        schedule at every event, so the frequency controls the *granularity*
        of the ramp, not the final target.
    """

    def __init__(
        self,
        finalSparsity: float = 0.9,
        startFraction: float = 0.2,
        endFraction: float = 0.8,
        initialSparsity: float = 0.0,
        pruneFrequency: int = 1,
    ) -> None:
        if not 0.0 <= finalSparsity < 1.0:
            raise ValueError("finalSparsity must lie in [0, 1).")
        if not 0.0 <= initialSparsity <= finalSparsity:
            raise ValueError("initialSparsity must lie in [0, finalSparsity].")
        if not 0.0 <= startFraction < endFraction <= 1.0:
            raise ValueError("Require 0 <= startFraction < endFraction <= 1.")
        if pruneFrequency < 1:
            raise ValueError("pruneFrequency must be at least 1.")

        self.finalSparsity = float(finalSparsity)
        self.startFraction = float(startFraction)
        self.endFraction = float(endFraction)
        self.initialSparsity = float(initialSparsity)
        self.pruneFrequency = int(pruneFrequency)

    def targetSparsity(self, progress: float) -> float:
        """Return the target sparsity at training progress ``progress``."""
        if progress <= self.startFraction:
            return self.initialSparsity
        if progress >= self.endFraction:
            return self.finalSparsity
        local = (progress - self.startFraction) / (self.endFraction - self.startFraction)
        ramp = (1.0 - local**3) ** 3
        return self.finalSparsity + (self.initialSparsity - self.finalSparsity) * ramp

    def shouldPrune(self, progress: float) -> bool:
        """Whether pruning is *active* at this progress.

        The window is closed on both ends: ``startFraction`` and
        ``endFraction`` are both inside the window, so the final sparsity
        is applied exactly at ``endFraction`` and not one step later.
        """
        return self.startFraction <= progress <= self.endFraction

    def describe(self) -> str:
        """Return a one-line description of the schedule."""
        return (
            f"cubic ramp: sparsity {self.initialSparsity:.3f} -> {self.finalSparsity:.3f} "
            f"over training fraction [{self.startFraction:.2f}, {self.endFraction:.2f}], "
            f"prune every {self.pruneFrequency} epoch(s)"
        )


class ConstantSparsitySchedule(PruningSchedule):
    """Prunes to ``finalSparsity`` immediately and holds it.

    This is the one-shot-pruning baseline used to justify the gradual ramp.
    It prunes exactly once, at the very first progress point it is asked
    about, and never again.
    """

    def __init__(self, finalSparsity: float = 0.9, startFraction: float = 0.0) -> None:
        if not 0.0 <= finalSparsity < 1.0:
            raise ValueError("finalSparsity must lie in [0, 1).")
        self.finalSparsity = float(finalSparsity)
        self.startFraction = float(startFraction)
        self._fired = False

    def targetSparsity(self, progress: float) -> float:
        """Return the target sparsity at training progress ``progress``."""
        if progress < self.startFraction:
            return 0.0
        return self.finalSparsity

    def shouldPrune(self, progress: float) -> bool:
        """Prune exactly once, as soon as ``startFraction`` is reached."""
        if progress < self.startFraction or self._fired:
            return False
        self._fired = True
        return True

    def describe(self) -> str:
        """Return a one-line description of the schedule."""
        return f"constant (one-shot): sparsity {self.finalSparsity:.3f} from the start"


def buildSchedule(
    kind: str = "cubic",
    finalSparsity: float = 0.9,
    **kwargs: object,
) -> PruningSchedule:
    """Factory that creates a schedule from its string name."""
    key = kind.lower()
    if key == "cubic":
        return CubicSparsitySchedule(finalSparsity=finalSparsity, **kwargs)  # type: ignore[arg-type]
    if key in {"constant", "oneshot", "one_shot"}:
        return ConstantSparsitySchedule(finalSparsity=finalSparsity, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown schedule '{kind}'. Known: cubic, constant.")
