"""Sparse-aware inference.

Three forward paths are implemented so that the difference between them can
be *measured* rather than asserted:

``denseForward``
    The ordinary dense matmul ``X @ W_effective``.  This is what a framework
    does.  It always performs ``batch * inFeatures * outFeatures`` multiply-
    accumulates, even if 90% of ``W_effective`` is zero.

``maskedDenseForward``
    A deliberately *honest but pointless* path: it multiplies the dense
    weight by the mask and then does a dense matmul.  It is included so the
    report can say explicitly that "dense forward with zeros" performs the
    **same** number of FLOPs as the dense path -- multiplying by zero is not
    a speedup, it is a rounding error at best and a correctness hazard at
    worst.

``sparseAwareForward``
    A genuine sparse path that never touches a masked connection.  For every
    output unit it gathers the *active* input indices of that unit and
    performs a matmul over only those columns of ``X``.  The number of
    multiply-accumulates is exactly the number of active connections, so the
    arithmetic work genuinely scales with sparsity.

The sparse path is written for clarity first and optimised afterwards (or
not at all).  Section "Sparse vs dense: the honest story" in ``DESIGN.md``
explains why the measured wall-clock speedup is much smaller than the
theoretical FLOP reduction, and the report prints both numbers side by side
so that nobody has to guess which one is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from aqua.nn.layers import DenseNetwork, Linear

__all__ = [
    "SparseLayerStructure",
    "buildSparseStructure",
    "denseForward",
    "maskedDenseForward",
    "sparseAwareForward",
    "inferenceAccuracy",
]


@dataclass
class SparseLayerStructure:
    """A compressed (CSC-style) view of one layer's mask and weights.

    For every output unit ``j`` we store the indices of the *active* input
    units feeding it and the corresponding weight values.  This is the
    classic compressed-sparse-column layout used by CSR/CSC sparse matrix
    libraries, written out in plain NumPy.
    """

    inputIndices: List[np.ndarray]
    weightValues: List[np.ndarray]
    inputCount: int
    outputCount: int

    @property
    def nonZeroCount(self) -> int:
        """Number of active (non-zero) connections."""
        return int(sum(values.size for values in self.weightValues))

    @property
    def denseCount(self) -> int:
        """Number of connections a dense layer would have."""
        return int(self.inputCount * self.outputCount)

    @property
    def density(self) -> float:
        """Fraction of connections that are active."""
        if self.denseCount == 0:
            return 0.0
        return self.nonZeroCount / self.denseCount


def buildSparseStructure(layer: Linear) -> SparseLayerStructure:
    """Build the compressed structure of ``layer`` from its current mask.

    The structure is rebuilt whenever the mask changes; the pruner changes
    the mask rarely compared to how often inference runs, so this is the
    right place to spend the work.
    """
    weights = layer.applyMaskToWeights()
    inputIndices: List[np.ndarray] = []
    weightValues: List[np.ndarray] = []
    for outputIndex in range(layer.outFeatures):
        column = weights[:, outputIndex]
        active = np.flatnonzero(column != 0.0)
        inputIndices.append(active)
        weightValues.append(column[active])
    return SparseLayerStructure(
        inputIndices=inputIndices,
        weightValues=weightValues,
        inputCount=layer.inFeatures,
        outputCount=layer.outFeatures,
    )


def denseForward(model: DenseNetwork, features: np.ndarray) -> np.ndarray:
    """Standard dense forward pass (the baseline)."""
    output = features
    for layer in model.layers:
        effective = layer.applyMaskToWeights()
        output = output @ effective
        if layer.bias is not None:
            output = output + layer.bias.data
        output = _applyActivationDense(output, layer)
    return output


def maskedDenseForward(model: DenseNetwork, features: np.ndarray) -> np.ndarray:
    """Forward pass that multiplies the dense weight by the mask.

    This performs the *same* arithmetic as :func:`denseForward`.  It exists
    to make the point explicit: "we pruned the weights" is not a cost
    reduction if the runtime still multiplies every weight by zero.
    """
    output = features
    for layer in model.layers:
        masked = layer.rawWeights.data * layer.mask
        output = output @ masked
        if layer.bias is not None:
            output = output + layer.bias.data
        output = _applyActivationDense(output, layer)
    return output


def sparseAwareForward(
    model: DenseNetwork,
    features: np.ndarray,
    structures: Optional[Sequence[SparseLayerStructure]] = None,
) -> np.ndarray:
    """Sparse-aware forward pass that skips masked connections entirely.

    For every output unit the layer gathers only the active input features
    and performs a matmul over that reduced slice.  A masked connection
    never appears in any gathered slice, so it is never multiplied.

    Parameters
    ----------
    model:
        The network to run.
    features:
        Input batch of shape ``(batch, inFeatures)``.
    structures:
        Pre-built sparse structures.  When ``None`` they are built on the
        fly, which would dominate the measurement; callers that time this
        path must build them once and pass them in.
    """
    if structures is None:
        structures = [buildSparseStructure(layer) for layer in model.layers]

    output = features
    for layer, structure in zip(model.layers, structures):
        batchSize = output.shape[0]
        layerOutput = np.zeros((batchSize, layer.outFeatures), dtype=np.float64)
        for outputIndex in range(structure.outputCount):
            activeInputs = structure.inputIndices[outputIndex]
            if activeInputs.size == 0:
                continue
            weights = structure.weightValues[outputIndex]
            layerOutput[:, outputIndex] = output[:, activeInputs] @ weights
        if layer.bias is not None:
            layerOutput = layerOutput + layer.bias.data
        output = _applyActivationDense(layerOutput, layer)
    return output


def _applyActivationDense(values: np.ndarray, layer: Linear) -> np.ndarray:
    """Apply ``layer``'s activation to a raw NumPy array."""
    name = layer.activation.name
    if name == "identity":
        return values
    if name == "relu":
        return np.maximum(values, 0.0)
    if name == "leakyRelu":
        slope = getattr(layer.activation, "negativeSlope", 0.01)
        return np.where(values > 0.0, values, slope * values)
    if name == "tanh":
        return np.tanh(values)
    if name == "sigmoid":
        result = np.empty_like(values, dtype=np.float64)
        positive = values >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponential = np.exp(values[~positive])
        result[~positive] = exponential / (1.0 + exponential)
        return result
    raise ValueError(f"Unsupported activation '{name}' in the dense inference path.")


def inferenceAccuracy(
    model: DenseNetwork,
    features: np.ndarray,
    labels: np.ndarray,
    forwardFn: Callable[[DenseNetwork, np.ndarray], np.ndarray] = denseForward,
) -> float:
    """Accuracy of a NumPy inference path (no autodiff involved)."""
    predictions = np.argmax(forwardFn(model, features), axis=1)
    if labels.size == 0:
        return 0.0
    return float(np.mean(predictions == labels))
