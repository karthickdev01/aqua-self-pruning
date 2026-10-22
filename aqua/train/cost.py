"""Cost measurement: operation counts and wall-clock latency.

What is measured
----------------
**Operation counts (exact, analytic).**

* ``denseMacs`` -- the multiply-accumulate count of a dense forward pass:
  ``sum over layers of batch * inFeatures * outFeatures``.
* ``activeMacs`` -- the multiply-accumulate count of the sparse-aware path:
  ``sum over layers of batch * (number of active connections)``.  This is
  the count of multiply-accumulates that the sparse path *actually performs*,
  because a masked connection is never gathered and therefore never
  multiplied.
* ``denseTimesZeroMacs`` -- the count of the "dense forward that multiplies
  by zero".  It equals ``denseMacs`` exactly.  Reporting it makes the
  central point of the challenge impossible to miss: multiplying by zero is
  not a saving.

**Wall-clock latency (measured).**

Latency is measured with :func:`time.perf_counter` over many repetitions
after a warm-up, and the *median* is reported (the median is far more
robust to scheduler noise than the mean).  Several batch sizes are
reported, because the overhead profile of a sparse path depends strongly on
the batch size: at batch size 1 the Python-level per-unit loop dominates and
the sparse path is *slower* than the dense path; at large batch sizes the
dense path starts to be memory-bandwidth bound.

**Honesty rules enforced by this module.**

* A speedup is only reported when the measured latency actually improves.
* The theoretical FLOP reduction and the measured wall-clock change are
  reported in *separate* fields so they can never be conflated.
* The measurement returns the raw samples, so the spread can be inspected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from aqua.data.datasets import Dataset
from aqua.nn.layers import DenseNetwork
from aqua.train.sparse import (
    SparseLayerStructure,
    buildSparseStructure,
    denseForward,
    inferenceAccuracy,
    maskedDenseForward,
    sparseAwareForward,
)

__all__ = ["LatencyResult", "measureLatency", "measureCost", "DEFAULT_BATCH_SIZES"]

DEFAULT_BATCH_SIZES = (1, 32, 256)


@dataclass
class LatencyResult:
    """A latency measurement."""

    label: str
    batchSize: int
    medianSeconds: float
    meanSeconds: float
    minimumSeconds: float
    samples: List[float] = field(default_factory=list)

    def asDict(self) -> Dict[str, Any]:
        """Return the measurement as a flat dictionary."""
        return {
            "label": self.label,
            "batchSize": self.batchSize,
            "medianSeconds": self.medianSeconds,
            "meanSeconds": self.meanSeconds,
            "minimumSeconds": self.minimumSeconds,
        }


def measureLatency(
    forwardFn: Callable[[], Any],
    repetitions: int = 200,
    warmup: int = 20,
) -> LatencyResult:
    """Measure the wall-clock latency of ``forwardFn``.

    The median is used as the headline number because a single scheduling
    hiccup can double a mean over a small number of repetitions.
    """
    for _ in range(warmup):
        forwardFn()

    samples: List[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        forwardFn()
        samples.append(time.perf_counter() - start)

    array = np.asarray(samples, dtype=np.float64)
    return LatencyResult(
        label="",
        batchSize=0,
        medianSeconds=float(np.median(array)),
        meanSeconds=float(np.mean(array)),
        minimumSeconds=float(np.min(array)),
        samples=[float(value) for value in samples],
    )


def _buildBatch(validationFeatures: np.ndarray, batchSize: int) -> np.ndarray:
    """Return a batch of exactly ``batchSize`` rows from the validation set."""
    if validationFeatures.shape[0] >= batchSize:
        return validationFeatures[:batchSize]
    repeats = int(np.ceil(batchSize / max(1, validationFeatures.shape[0])))
    return np.tile(validationFeatures, (repeats, 1))[:batchSize]


def measureCost(
    model: DenseNetwork,
    dataset: Dataset,
    batchSizes: Tuple[int, ...] = DEFAULT_BATCH_SIZES,
    repetitions: int = 200,
) -> Dict[str, Any]:
    """Measure operation counts and latency for a trained model.

    Returns a dictionary with

    * the exact parameter counts,
    * the exact multiply-accumulate counts for the dense, dense-times-zero
      and sparse-aware paths,
    * the measured latency for each path at each batch size,
    * the accuracy of the sparse-aware path (it must match the dense path
      to within floating point tolerance -- this is checked by the test
      suite, and reported here as ``sparseMatchesDense``).
    """
    totalConnections = model.totalParameterCount()
    activeConnections = model.activeParameterCount()
    achievedSparsity = model.sparsity()

    structures: List[SparseLayerStructure] = [
        buildSparseStructure(layer) for layer in model.layers
    ]

    validationFeatures = dataset.validationFeatures
    validationLabels = dataset.validationLabels

    denseOutputs = denseForward(model, validationFeatures)
    sparseOutputs = sparseAwareForward(model, validationFeatures, structures)
    maskedOutputs = maskedDenseForward(model, validationFeatures)
    maxDifference = (
        float(np.max(np.abs(denseOutputs - sparseOutputs))) if denseOutputs.size else 0.0
    )
    maskedDifference = (
        float(np.max(np.abs(denseOutputs - maskedOutputs))) if denseOutputs.size else 0.0
    )

    latencyByPath: Dict[str, Dict[int, Dict[str, float]]] = {
        "dense": {},
        "denseTimesZero": {},
        "sparseAware": {},
    }

    for batchSize in batchSizes:
        batch = _buildBatch(validationFeatures, batchSize)

        denseMeasurement = measureLatency(
            lambda: denseForward(model, batch), repetitions=repetitions
        )
        denseMeasurement.label = "dense"
        denseMeasurement.batchSize = batchSize

        maskedMeasurement = measureLatency(
            lambda: maskedDenseForward(model, batch), repetitions=repetitions
        )
        maskedMeasurement.label = "denseTimesZero"
        maskedMeasurement.batchSize = batchSize

        sparseMeasurement = measureLatency(
            lambda: sparseAwareForward(model, batch, structures), repetitions=repetitions
        )
        sparseMeasurement.label = "sparseAware"
        sparseMeasurement.batchSize = batchSize

        latencyByPath["dense"][batchSize] = denseMeasurement.asDict()
        latencyByPath["denseTimesZero"][batchSize] = maskedMeasurement.asDict()
        latencyByPath["sparseAware"][batchSize] = sparseMeasurement.asDict()

    # Multiply-accumulate counts for the validation batch actually used.
    evaluationBatch = validationFeatures.shape[0]
    denseMacs = int(evaluationBatch * totalConnections)
    activeMacs = int(evaluationBatch * activeConnections)

    return {
        "totalConnections": int(totalConnections),
        "activeConnections": int(activeConnections),
        "achievedSparsity": float(achievedSparsity),
        "biasParameters": int(
            sum(layer.bias.size for layer in model.layers if layer.bias is not None)
        ),
        "totalScalarParameters": int(model.parameterCount()),
        "denseMacs": denseMacs,
        "denseTimesZeroMacs": denseMacs,
        "activeMacs": activeMacs,
        "theoreticalMacReduction": float(1.0 - (activeMacs / denseMacs)) if denseMacs else 0.0,
        "denseMacsPerSample": int(totalConnections),
        "activeMacsPerSample": int(activeConnections),
        "evaluationBatchSize": int(evaluationBatch),
        "sparseMatchesDense": bool(maxDifference < 1e-9),
        "sparseMaxAbsDifference": maxDifference,
        "maskedDenseMaxAbsDifference": maskedDifference,
        "denseAccuracy": inferenceAccuracy(model, validationFeatures, validationLabels, denseForward),
        "sparseAccuracy": inferenceAccuracy(
            model,
            validationFeatures,
            validationLabels,
            lambda m, f: sparseAwareForward(m, f, structures),
        ),
        "latency": latencyByPath,
        "latencyRepetitions": int(repetitions),
    }
