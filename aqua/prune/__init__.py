"""Self-pruning machinery for the AQUA project.

* :mod:`aqua.prune.criteria` -- importance criteria (magnitude, saliency,
  signed Taylor, gradient-magnitude ablation) and the Taylor derivation
  behind them.
* :mod:`aqua.prune.schedule` -- the cubic sparsity ramp and a one-shot
  baseline.
* :mod:`aqua.prune.allocator` -- global budget allocation with deterministic
  tie-breaking.
* :mod:`aqua.prune.pruner` -- the controller that ties the three together
  and keeps the optimiser state consistent with the masks.
* :mod:`aqua.prune.regrowth` -- budget-neutral revival of pruned
  connections.
"""

from aqua.prune.allocator import AllocationResult, MaskAllocator
from aqua.prune.criteria import (
    GradientMagnitudePruner,
    ImportanceCriterion,
    MagnitudePruner,
    SaliencyPruner,
    SignedTaylorPruner,
    buildCriterion,
)
from aqua.prune.pruner import Pruner, PruningEvent
from aqua.prune.regrowth import Regrowth, RegrowthEvent
from aqua.prune.schedule import (
    ConstantSparsitySchedule,
    CubicSparsitySchedule,
    PruningSchedule,
    buildSchedule,
)

__all__ = [
    "AllocationResult",
    "ConstantSparsitySchedule",
    "CubicSparsitySchedule",
    "GradientMagnitudePruner",
    "ImportanceCriterion",
    "MagnitudePruner",
    "MaskAllocator",
    "Pruner",
    "PruningEvent",
    "PruningSchedule",
    "Regrowth",
    "RegrowthEvent",
    "SaliencyPruner",
    "SignedTaylorPruner",
    "buildCriterion",
    "buildSchedule",
]
