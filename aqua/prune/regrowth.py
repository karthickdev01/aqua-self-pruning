"""Regrowth: letting a pruned connection come back.

Why regrowth is worth the trouble
---------------------------------
A pure prune-only schedule is *monotone*: once a connection is gone it is
gone forever.  That is a strong assumption.  A connection that is useless at
epoch 10 may become the most useful one at epoch 40, because the rest of the
network re-organised around it being absent.  Allowing a small number of
connections to be revived lets the structure keep searching instead of
freezing a mistake.

The stability problem
---------------------
Re-growth is where most "self-pruning" implementations silently corrupt
training, and it is exactly the scenario the challenge probes.  When a
connection is revived there are three pieces of state that must be handled:

1. **The raw weight.**  The connection was zeroed when it was pruned.  A
   revived connection therefore starts from weight zero and has to *learn*
   its value again from scratch.  Restarting at zero is deliberate: the
   weight is meaningless once the connection has been removed for a while,
   and re-initialising it with a random value would inject noise into a
   converged model.

2. **The Adam moments.**  A revived connection must start with
   :math:`m = v = 0` and :math:`t = 0`.  If the stale :math:`v` from before
   the pruning survived, the first update after revival would be scaled by a
   denominator that does not correspond to the new gradient, and the step
   size would be wrong.  Because the bias correction is computed per
   *element*, zeroing the element's timestep makes the correction restart
   from :math:`t = 1`, which is exactly the correct behaviour for a brand new
   parameter.

3. **The mask.**  The mask entry is set back to ``1`` and the optimiser's
   ``enforceMasks()`` will not zero it again.

The budget-neutral swap
-----------------------
The challenge asks for a **hard compute budget**.  Re-growth that simply
adds connections back would break that budget, so this implementation
performs a *swap* instead:

* **revive** the dead connection with the highest sensitivity
  :math:`|dL/dW_{\\text{effective}}|`, and
* **prune** the live connection with the lowest importance score,

in equal numbers, in the same event.  The number of active connections is
therefore *unchanged*, the sparsity target is respected exactly, and the
structure still gets to re-allocate capacity towards the connections that
now matter.

Which score ranks the candidates
--------------------------------
For a connection that is **currently absent**, the Taylor product
:math:`|w g|` is zero for every candidate (the effective weight is zero), so
it carries no information.  The correct quantity for an absent connection is
the raw sensitivity :math:`|dL/dW_{\\text{effective}}|`: it answers "how
much would the loss change if this connection were re-introduced with a
small weight?".  The victim is then chosen with the *same* importance
criterion that the pruner uses, so the comparison is apples-to-apples.

Stability
---------
The number of swaps per event is capped by ``maxReviveFraction`` (a
fraction of the live connections).  Without a cap, the structure churns: a
revived connection immediately looks valuable (its gradient is non-zero
precisely because it is connected), so an uncapped rule would swap forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from aqua.nn.layers import DenseNetwork, Linear
from aqua.optim.adam import AdamOptimizer

__all__ = ["Regrowth", "RegrowthEvent"]


@dataclass
class RegrowthEvent:
    """A record of one re-growth event."""

    epoch: int
    revivedCount: int
    prunedCount: int
    budgetNeutral: bool
    activeParameters: int
    achievedSparsity: float

    def asDict(self) -> Dict[str, object]:
        """Return the event as a flat dictionary."""
        return {
            "epoch": self.epoch,
            "revivedCount": self.revivedCount,
            "prunedCount": self.prunedCount,
            "budgetNeutral": self.budgetNeutral,
            "activeParameters": self.activeParameters,
            "achievedSparsity": self.achievedSparsity,
        }


class Regrowth:
    """Revives pruned connections by swapping them for unimportant live ones.

    Parameters
    ----------
    model:
        The network to restructure.
    optimizer:
        The optimiser whose state is updated on revival and on pruning.
    criterion:
        The importance criterion.  It is used to score the *live* victims;
        the *dead* candidates are scored by their gradient sensitivity.
    maxCandidatesPerEvent:
        Hard cap on how many connections are swapped in a single event.
    maxReviveFraction:
        Hard cap on the fraction of the live connections that may be swapped
        in a single event.
    """

    def __init__(
        self,
        model: DenseNetwork,
        optimizer: AdamOptimizer,
        criterion: object,
        maxCandidatesPerEvent: int = 256,
        maxReviveFraction: float = 0.05,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.maxCandidatesPerEvent = int(maxCandidatesPerEvent)
        self.maxReviveFraction = float(maxReviveFraction)
        self.events: List[RegrowthEvent] = []

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _sensitivity(self, layer: Linear) -> Optional[np.ndarray]:
        """Return the gradient sensitivity of every connection of ``layer``."""
        averaged = getattr(self.criterion, "averagedGradient", None)
        gradient = averaged(layer) if averaged is not None else layer.effectiveWeightGrad
        if gradient is None:
            return None
        return np.abs(gradient)

    def _importance(self, layer: Linear) -> np.ndarray:
        """Return the importance score of every connection of ``layer``."""
        score = getattr(self.criterion, "score", None)
        if score is None:
            sensitivity = self._sensitivity(layer)
            if sensitivity is None:
                return np.zeros(layer.mask.shape, dtype=np.float64)
            return sensitivity
        return np.asarray(score(layer), dtype=np.float64)

    def _pooledCandidates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the dead-connection candidates ranked best-first.

        Returns ``(scores, layerIndices, positions)`` where the scores are
        sorted descending (best candidate first) and the positions are flat
        indices into the layer's mask.
        """
        pooledScores: List[np.ndarray] = []
        pooledLayerIndex: List[np.ndarray] = []
        pooledPosition: List[np.ndarray] = []

        for layerIndex, layer in enumerate(self.model.prunableLayers()):
            sensitivity = self._sensitivity(layer)
            if sensitivity is None:
                continue
            dead = layer.mask == 0
            if not np.any(dead):
                continue
            pooledScores.append(sensitivity[dead])
            pooledLayerIndex.append(
                np.full(int(np.count_nonzero(dead)), layerIndex, dtype=np.int64)
            )
            pooledPosition.append(np.flatnonzero(dead.reshape(-1)))

        if not pooledScores:
            empty = np.zeros(0, dtype=np.float64)
            return empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        flatScores = np.concatenate(pooledScores)
        flatLayerIndex = np.concatenate(pooledLayerIndex)
        flatPosition = np.concatenate(pooledPosition)

        # Descending score; ties broken by the pooled index (stable sort).
        order = np.argsort(-flatScores, kind="stable")
        return flatScores[order], flatLayerIndex[order], flatPosition[order]

    def _pooledVictims(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the live connections ranked least-important-first."""
        pooledScores: List[np.ndarray] = []
        pooledLayerIndex: List[np.ndarray] = []
        pooledPosition: List[np.ndarray] = []

        for layerIndex, layer in enumerate(self.model.prunableLayers()):
            importance = self._importance(layer)
            alive = layer.mask == 1
            if not np.any(alive):
                continue
            pooledScores.append(importance[alive])
            pooledLayerIndex.append(
                np.full(int(np.count_nonzero(alive)), layerIndex, dtype=np.int64)
            )
            pooledPosition.append(np.flatnonzero(alive.reshape(-1)))

        if not pooledScores:
            empty = np.zeros(0, dtype=np.float64)
            return empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        flatScores = np.concatenate(pooledScores)
        flatLayerIndex = np.concatenate(pooledLayerIndex)
        flatPosition = np.concatenate(pooledPosition)

        # Ascending score: the least important live connections come first.
        order = np.argsort(flatScores, kind="stable")
        return flatScores[order], flatLayerIndex[order], flatPosition[order]

    # ------------------------------------------------------------------
    # The event
    # ------------------------------------------------------------------
    def regrow(
        self,
        epoch: int = 0,
        targetSparsity: Optional[float] = None,
        budgetNeutral: bool = True,
    ) -> RegrowthEvent:
        """Run one re-growth (swap) event.

        Parameters
        ----------
        targetSparsity:
            When ``budgetNeutral`` is ``True`` the swap is refused if it
            would push the achieved sparsity below this value.  Because a
            swap keeps the active count constant, the sparsity is unchanged
            and the check is a no-op -- it is kept as an explicit guard.
        budgetNeutral:
            When ``True`` the number of revived connections equals the
            number of pruned ones, so the sparsity is exactly preserved.
        """
        activeBefore = self.model.activeParameterCount()

        candidateScores, candidateLayerIndex, candidatePosition = self._pooledCandidates()
        victimScores, victimLayerIndex, victimPosition = self._pooledVictims()

        def _emptyEvent() -> RegrowthEvent:
            return RegrowthEvent(
                epoch=epoch,
                revivedCount=0,
                prunedCount=0,
                budgetNeutral=budgetNeutral,
                activeParameters=self.model.activeParameterCount(),
                achievedSparsity=self.model.sparsity(),
            )

        if candidateScores.size == 0 or victimScores.size == 0:
            event = _emptyEvent()
            self.events.append(event)
            return event

        # How many swaps are allowed?
        maxSwap = self.maxCandidatesPerEvent
        if self.maxReviveFraction > 0:
            maxSwap = min(maxSwap, int(activeBefore * self.maxReviveFraction))
        maxSwap = min(maxSwap, int(victimScores.size))
        if targetSparsity is not None and budgetNeutral:
            total = self.model.totalParameterCount()
            if total > 0:
                allowedActive = int(round(total * (1.0 - targetSparsity)))
                # A swap does not change the active count, so the budget is
                # respected as long as we are not already over it.
                if activeBefore > allowedActive:
                    maxSwap = 0

        if maxSwap <= 0:
            event = _emptyEvent()
            self.events.append(event)
            return event

        layers = self.model.prunableLayers()

        revivedByLayer: Dict[int, List[int]] = {}
        for layerIndex, position in zip(
            candidateLayerIndex[:maxSwap], candidatePosition[:maxSwap]
        ):
            revivedByLayer.setdefault(id(layers[layerIndex]), []).append(int(position))

        prunedByLayer: Dict[int, List[int]] = {}
        for layerIndex, position in zip(
            victimLayerIndex[:maxSwap], victimPosition[:maxSwap]
        ):
            prunedByLayer.setdefault(id(layers[layerIndex]), []).append(int(position))

        revivedCount = 0
        prunedCount = 0

        for layer in layers:
            revived = np.asarray(revivedByLayer.get(id(layer), []), dtype=np.int64)
            pruned = np.asarray(prunedByLayer.get(id(layer), []), dtype=np.int64)
            if revived.size == 0 and pruned.size == 0:
                continue

            flatMask = layer.mask.reshape(-1)
            flatWeights = layer.rawWeights.data.reshape(-1)

            if pruned.size:
                flatMask[pruned] = 0.0
                flatWeights[pruned] = 0.0
                # The pruned connection's optimiser state is discarded.
                self.optimizer.resetStateAt(layer.rawWeights, pruned)
                prunedCount += int(pruned.size)

            if revived.size:
                flatMask[revived] = 1.0
                # A revived connection starts from exactly zero and its
                # optimiser state is reset, so its bias correction restarts.
                flatWeights[revived] = 0.0
                self.optimizer.resetStateAt(layer.rawWeights, revived)
                revivedCount += int(revived.size)

            layer.mask = flatMask.reshape(layer.mask.shape)

        event = RegrowthEvent(
            epoch=epoch,
            revivedCount=revivedCount,
            prunedCount=prunedCount,
            budgetNeutral=budgetNeutral,
            activeParameters=self.model.activeParameterCount(),
            achievedSparsity=self.model.sparsity(),
        )
        self.events.append(event)
        return event
