"""Global mask allocation.

The pruner must decide *which* connections to remove, and the decision has to
be made **globally across the whole network**, not layer by layer.  A
layer-local budget would, for example, force the first layer to keep 90% of
its connections even when the second layer needs 90% of the network's
budget more urgently.

The allocator therefore:

1. collects the per-connection importance score of every prunable layer,
2. pools all scores into one flat vector,
3. finds the threshold that removes exactly the requested number of
   connections,
4. writes the new masks back, layer by layer.

Determinism
-----------
Ties are broken by a **stable sort on the pooled score**, where the pooled
index order is deterministic (layers in model order, then row-major position
within a layer).  NumPy's ``argsort`` with ``kind="stable"`` guarantees that
equal scores are ordered by their original index, so the same seed always
produces exactly the same mask.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from aqua.nn.layers import DenseNetwork, Linear

__all__ = ["MaskAllocator", "AllocationResult"]


class AllocationResult:
    """The outcome of a mask allocation."""

    def __init__(
        self,
        prunedCount: int,
        totalCount: int,
        previousSparsity: float,
        newSparsity: float,
    ) -> None:
        self.prunedCount = prunedCount
        self.totalCount = totalCount
        self.previousSparsity = previousSparsity
        self.newSparsity = newSparsity

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"AllocationResult(pruned={self.prunedCount}, total={self.totalCount}, "
            f"sparsity {self.previousSparsity:.4f} -> {self.newSparsity:.4f})"
        )


class MaskAllocator:
    """Allocates a global sparsity budget across the layers of a network."""

    def __init__(self, model: DenseNetwork) -> None:
        self.model = model

    def prunableLayers(self) -> List[Linear]:
        """Return the layers whose weights are eligible for pruning."""
        return self.model.prunableLayers()

    def totalConnections(self) -> int:
        """Total number of prunable connections across the network."""
        return int(sum(layer.mask.size for layer in self.prunableLayers()))

    def activeConnections(self) -> int:
        """Number of connections that are currently alive."""
        return int(
            sum(int(np.count_nonzero(layer.mask)) for layer in self.prunableLayers())
        )

    def allocate(
        self,
        scores: Dict[int, np.ndarray],
        targetSparsity: float,
    ) -> AllocationResult:
        """Allocate ``targetSparsity`` across the network using ``scores``.

        Parameters
        ----------
        scores:
            Mapping from ``id(layer)`` to a per-connection importance score
            of the same shape as the layer's mask.
        targetSparsity:
            The desired fraction of *prunable* connections to remove.  The
            value is clamped to the achievable range: a network can never be
            sparser than 100% of its connections removed, and the requested
            sparsity can never be *less* than the current one (pruning is
            monotone -- connections are removed, never un-removed by the
            allocator; re-growth is a separate, explicit mechanism).

        Returns
        -------
        AllocationResult
            How many connections were removed and the resulting sparsity.
        """
        layers = self.prunableLayers()
        total = self.totalConnections()
        if total == 0:
            return AllocationResult(0, 0, 0.0, 0.0)

        currentActive = self.activeConnections()
        currentSparsity = 1.0 - (currentActive / total)

        desired = float(np.clip(targetSparsity, currentSparsity, 1.0))
        targetActive = int(round(total * (1.0 - desired)))
        removalsNeeded = currentActive - targetActive
        if removalsNeeded <= 0:
            return AllocationResult(0, total, currentSparsity, currentSparsity)

        # Pool the scores of all *alive* connections into one flat vector.
        pooledScores: List[np.ndarray] = []
        pooledLayerIndex: List[np.ndarray] = []
        pooledPosition: List[np.ndarray] = []
        for layerIndex, layer in enumerate(layers):
            score = scores.get(id(layer))
            if score is None:
                score = np.zeros(layer.mask.shape, dtype=np.float64)
            alive = layer.mask > 0
            pooledScores.append(score[alive])
            pooledLayerIndex.append(
                np.full(int(np.count_nonzero(alive)), layerIndex, dtype=np.int64)
            )
            pooledPosition.append(np.flatnonzero(alive.reshape(-1)))

        if not pooledScores:
            return AllocationResult(0, total, currentSparsity, currentSparsity)

        flatScores = np.concatenate(pooledScores)
        flatLayerIndex = np.concatenate(pooledLayerIndex)
        flatPosition = np.concatenate(pooledPosition)

        # Deterministic ordering: ascending score, ties broken by the pooled
        # index (stable sort guarantees this).
        order = np.argsort(flatScores, kind="stable")
        victims = order[:removalsNeeded]

        victimLayerIndex = flatLayerIndex[victims]
        victimPosition = flatPosition[victims]

        for layerIndex, layer in enumerate(layers):
            selected = victimPosition[victimLayerIndex == layerIndex]
            if selected.size == 0:
                continue
            flatMask = layer.mask.reshape(-1)
            flatMask[selected] = 0.0
            layer.mask = flatMask.reshape(layer.mask.shape)

        newActive = self.activeConnections()
        newSparsity = 1.0 - (newActive / total)
        return AllocationResult(
            prunedCount=int(victims.size),
            totalCount=total,
            previousSparsity=currentSparsity,
            newSparsity=newSparsity,
        )
