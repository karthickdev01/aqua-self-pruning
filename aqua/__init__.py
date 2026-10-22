"""AQUA -- a self-pruning neural network built from scratch in NumPy.

The package is split by responsibility:

``aqua.engine``
    Reverse-mode automatic differentiation (tensors, graph, ops, masking).
``aqua.nn``
    Layers, activations, initialisation, MLP assembly.
``aqua.optim``
    The Adam optimiser with mask-aware state semantics.
``aqua.prune``
    Importance criteria, pruning schedules, mask allocation, re-growth.
``aqua.data``
    Deterministic synthetic datasets.
``aqua.train``
    Training / evaluation loops, experiment driver, cost measurement.
``aqua.utils``
    Seeding, result serialisation, small helpers.
"""

__version__ = "1.0.0"
