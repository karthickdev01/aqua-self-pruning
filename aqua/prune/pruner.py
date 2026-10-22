"""The self-pruning controller.

The pruner owns the *policy*; the allocator owns the *mechanics*; the
criteria own the *scores*.  Keeping these three apart is what makes the
system testable: the allocation can be unit-tested without a training loop,
and the criterion can be unit-tested without a model.

What happens at a pruning event
-------------------------------
1. The importance score of every prunable layer is computed (consistently
   across layers -- same criterion, same data source).
2. The global allocator removes the least important connections until the
   schedule's sparsity budget is met.
3. The masks are written back to the layers.
4. The **optimiser state of the removed connections is reset**, so no stale
   momentum survives into a possible future re-growth.
5. The raw weights of the removed connections are zeroed, so the effective
   forward pass contributes exactly zero.
6. The achieved sparsity and active parameter count are recorded.

Biases are never pruned
-----------------------
A bias is not a connection between two units; it is a constant offset
applied to a whole unit.  Removing a bias does not save a matrix multiply,
it just shifts the unit's operating point -- which is not something a
saliency score can reason about meaningfully, because the "gradient of the
loss with respect to a bias" is a sum over the whole batch and says nothing
about whether that unit is useful.  Biases are therefore excluded from the
pruning budget, and the reported sparsity is computed over *prunable
connections only*.  This is stated explicitly in the README so that the
sparsity numbers cannot be misread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from aqua.nn.layers import DenseNetwork
from aqua.optim.adam import AdamOptimizer
from aqua.prune.allocator import AllocationResult, MaskAllocator
from aqua.prune.criteria import SaliencyPruner
from aqua.prune.schedule import PruningSchedule

__all__ = ["Pruner", "PruningEvent"]


@dataclass
class PruningEvent:
    """A record of one pruning event."""

    epoch: int
    progress: float
    targetSparsity: float
    achievedSparsity: float
    prunedCount: int
    totalCount: int
    activeParameters: int
    revivedCount: int = 0
    extras: Dict[str, float] = field(default_factory=dict)

    def asDict(self) -> Dict[str, object]:
        """Return the event as a flat dictionary."""
        payload: Dict[str, object] = {
            "epoch": self.epoch,
            "progress": self.progress,
            "targetSparsity": self.targetSparsity,
            "achievedSparsity": self.achievedSparsity,
            "prunedCount": self.prunedCount,
            "totalCount": self.totalCount,
            "activeParameters": self.activeParameters,
            "revivedCount": self.revivedCount,
        }
        payload.update(self.extras)
        return payload


class Pruner:
    """Coordinates criteria, schedule, allocation and optimiser state.

    Parameters
    ----------
    model:
        The network whose weights are pruned.
    optimizer:
        The optimiser whose state must be kept consistent with the masks.
    criterion:
        The importance criterion.  Defaults to a batch-accumulating
        :class:`~aqua.prune.criteria.SaliencyPruner`.
    schedule:
        The sparsity schedule.  Defaults to a cubic ramp to 90%.
    """

    def __init__(
        self,
        model: DenseNetwork,
        optimizer: AdamOptimizer,
        criterion: Optional[object] = None,
        schedule: Optional[PruningSchedule] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else SaliencyPruner()
        self.schedule = schedule
        self.allocator = MaskAllocator(model)
        self.events: List[PruningEvent] = []
        self.lastAllocation: Optional[AllocationResult] = None

    # ------------------------------------------------------------------
    # Score collection
    # ------------------------------------------------------------------
    def collectScores(self) -> Dict[int, np.ndarray]:
        """Return the per-connection importance score of every prunable layer."""
        scores: Dict[int, np.ndarray] = {}
        for layer in self.model.prunableLayers():
            scores[id(layer)] = np.asarray(self.criterion.score(layer), dtype=np.float64)
        return scores

    def observeBatch(self) -> None:
        """Give the criterion a chance to accumulate the latest mini-batch.

        The trainer calls this after every mini-batch.  Criteria that do not
        accumulate simply ignore it.

        This is essential for the gradient-based criteria: the gradient of a
        single mini-batch is a noisy estimate of the gradient of the loss on
        the whole dataset, and a ranking built from a noisy gradient is
        measurably *worse* than a magnitude ranking.  Averaging the score
        over the mini-batches of an epoch removes most of that noise.
        """
        accumulate = getattr(self.criterion, "accumulate", None)
        if accumulate is None:
            return
        for layer in self.model.prunableLayers():
            accumulate(layer)

    def resetAccumulation(self) -> None:
        """Reset any batch accumulation in the criterion."""
        reset = getattr(self.criterion, "reset", None)
        if reset is not None:
            reset()

    # ------------------------------------------------------------------
    # The pruning event
    # ------------------------------------------------------------------
    def pruneTo(self, targetSparsity: float) -> AllocationResult:
        """Prune the network until it reaches ``targetSparsity``.

        This method is idempotent: calling it again with the same target
        prunes nothing, because the allocator clamps the request to the
        achievable range.
        """
        scores = self.collectScores()
        allocation = self.allocator.allocate(scores, targetSparsity)
        self.lastAllocation = allocation

        if allocation.prunedCount > 0:
            self._syncOptimiserState()
            self._zeroPrunedWeights()

        return allocation

    def _syncOptimiserState(self) -> None:
        """Reset the optimiser state of every connection that was removed.

        This is the step that most naive implementations forget.  Without it,
        a connection that was pruned while carrying a large Adam first moment
        would have that momentum "reappear" if it were ever revived, and the
        first update after revival would be mis-scaled.
        """
        for layer in self.model.prunableLayers():
            deadIndices = np.flatnonzero((layer.mask == 0).reshape(-1))
            if deadIndices.size == 0:
                continue
            self.optimizer.resetStateAt(layer.rawWeights, deadIndices)

    def _zeroPrunedWeights(self) -> None:
        """Force the raw weight of every removed connection to exactly zero."""
        for layer in self.model.prunableLayers():
            dead = layer.mask == 0
            if not np.any(dead):
                continue
            layer.rawWeights.data[dead] = 0.0

    # ------------------------------------------------------------------
    # Epoch hook
    # ------------------------------------------------------------------
    def onEpochEnd(self, epoch: int, totalEpochs: int) -> Optional[PruningEvent]:
        """Run a pruning event if the schedule calls for one.

        Returns the :class:`PruningEvent` that was executed, or ``None`` if no
        pruning happened at this epoch.
        """
        if self.schedule is None:
            return None

        progress = epoch / max(1, totalEpochs)
        if not self.schedule.shouldPrune(progress):
            return None

        target = self.schedule.targetSparsity(progress)
        allocation = self.pruneTo(target)
        self.resetAccumulation()

        event = PruningEvent(
            epoch=epoch,
            progress=progress,
            targetSparsity=target,
            achievedSparsity=allocation.newSparsity,
            prunedCount=allocation.prunedCount,
            totalCount=allocation.totalCount,
            activeParameters=self.model.activeParameterCount(),
        )
        self.events.append(event)
        return event

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        """Return a summary of the pruning activity."""
        return {
            "achievedSparsity": self.model.sparsity(),
            "activeParameters": float(self.model.activeParameterCount()),
            "totalParameters": float(self.model.totalParameterCount()),
            "eventCount": float(len(self.events)),
        }
