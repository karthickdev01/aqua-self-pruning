"""Apply the engine fixes directly to the files on disk.

This is a maintenance script kept in the repository so that the engine
fixes can be re-applied deterministically and verified, without relying on
an editor's file-content cache.

The three fixes are:

1. ``Tensor``: integer and boolean data is promoted to ``float64``.
2. ``Tensor``: ``retainGraph=True`` snapshots, clears, recomputes and
   re-adds the gradients, so an accumulation is correct for shared subgraphs.
3. ``ops.reduceGradToShape``: broadcasts to the target shape before
   reshaping, which fixes a trailing-axis case.
4. ``Adam``: global (not per-tensor) gradient-norm clipping with
   diagnostics, and ``epsilon`` may be zero.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path: Path, old: str, new: str, label: str) -> bool:
    """Replace ``old`` with ``new`` in ``path`` if ``old`` is present."""
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"  SKIP  {label}: pattern not found")
        return False
    if text.count(old) != 1:
        print(f"  SKIP  {label}: pattern appears {text.count(old)} times")
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"  OK    {label}")
    return True


# ----------------------------------------------------------------------
# 1. Tensor: integer promotion and the retainGraph accumulation.
# ----------------------------------------------------------------------
TENSOR = ROOT / "aqua" / "engine" / "tensor.py"

patch(
    TENSOR,
    """    if array.dtype == np.bool_:
        array = array.astype(np.float64)
    return array""",
    """    if array.dtype.kind in "biu":
        # Integer and boolean data is promoted to float64 so that gradients
        # never truncate and so that the dtype is uniform across the graph.
        array = array.astype(np.float64)
    return array""",
    "tensor: integer promotion",
)

patch(
    TENSOR,
    """        topo = self._buildTopologicalOrder()

        if not retainGraph:
            for node in topo:
                node.grad = None
""",
    """        topo = self._buildTopologicalOrder()

        if retainGraph:
            # Each backward pass must be *independent* and the results must
            # be summed.  Simply leaving the buffers in place would make the
            # second pass propagate the already-accumulated upstream
            # gradient, which squares the contribution of shared nodes.
            # Snapshotting, clearing, running and re-adding gives the
            # behaviour people expect from "accumulate this loss's
            # gradient into the buffers".
            savedGrads = {id(node): node.grad for node in topo}
            for node in topo:
                node.grad = None
        else:
            for node in topo:
                node.grad = None
""",
    "tensor: retainGraph snapshot",
)

patch(
    TENSOR,
    """        for node in reversed(topo):
            if node._backwardFn is not None and node.grad is not None:
                node._backwardFn()

    # ------------------------------------------------------------------
    # Operator sugar""",
    """        for node in reversed(topo):
            if node._backwardFn is not None and node.grad is not None:
                node._backwardFn()

        if retainGraph:
            for node in topo:
                previous = savedGrads.get(id(node))
                if previous is None:
                    continue
                if node.grad is None:
                    node.grad = previous
                else:
                    node.grad = previous + node.grad

    # ------------------------------------------------------------------
    # Operator sugar""",
    "tensor: retainGraph re-add",
)

# ----------------------------------------------------------------------
# 2. Ops: the trailing-axis broadcast reduction.
# ----------------------------------------------------------------------
OPS = ROOT / "aqua" / "engine" / "ops.py"

patch(
    OPS,
    """    # Sum over axes that were stretched from size 1.
    for axis, size in enumerate(targetShape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(targetShape)""",
    """    # Sum over the axes that were stretched from size 1.  The reduction
    # uses keepdims=True so that the axis indices of the *remaining* axes do
    # not shift while the loop is running.
    for axis, size in enumerate(targetShape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    # The gradient may now have a *trailing* axis of size 1 where the target
    # has a larger one.  Broadcast the result to the target shape before
    # reshaping.
    if grad.shape != targetShape:
        grad = np.broadcast_to(grad, targetShape)

    return np.ascontiguousarray(grad).reshape(targetShape)""",
    "ops: broadcast reduction",
)

# ----------------------------------------------------------------------
# 3. Adam: global-norm clipping and the epsilon validation.
# ----------------------------------------------------------------------
ADAM = ROOT / "aqua" / "optim" / "adam.py"

patch(
    ADAM,
    """        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")""",
    """        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")""",
    "adam: epsilon validation",
)

patch(
    ADAM,
    """    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------
    def _clipGradient(self, gradient: np.ndarray) -> np.ndarray:
        \"\"\"Clip ``gradient`` by its global L2 norm, if clipping is enabled.\"\"\"
        if self.maxGradientNorm is None:
            return gradient
        norm = float(np.linalg.norm(gradient))
        if norm <= self.maxGradientNorm or norm == 0.0:
            return gradient
        return gradient * (self.maxGradientNorm / (norm + 1e-12))

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------
    def step(self) -> None:
        \"\"\"Apply one Adam update to every registered parameter.\"\"\"
        if self._model is None:
            raise RuntimeError("attachModel() must be called before step().")

        for parameter in self._model.parameters():
            self._stepParameter(parameter)

        self.stepCount += 1
        self.enforceMasks()

    def _stepParameter(self, parameter: Parameter) -> None:
        \"\"\"Apply the Adam update to a single parameter.\"\"\"
        if parameter.grad is None:
            return
        state = self.getState(parameter)
        gradient = self._clipGradient(parameter.grad)
""",
    """    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------
    def _globalGradientNorm(self, parameters: Sequence[Parameter]) -> float:
        \"\"\"Return the global L2 norm across every parameter gradient.\"\"\"
        totalSquared = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            totalSquared += float(np.sum(parameter.grad * parameter.grad))
        return float(np.sqrt(totalSquared))

    def _clipScaleFactor(self, parameters: Sequence[Parameter]) -> float:
        \"\"\"Return the multiplicative factor that clips the global gradient norm.

        Returns ``1.0`` when clipping is disabled or the norm is already below
        the threshold.  Clipping is applied *globally* (across all
        parameters) rather than per tensor, which is the standard definition
        and keeps the effective step size independent of how many parameters
        the model happens to have.
        \"\"\"
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
        \"\"\"Apply one Adam update to every registered parameter.\"\"\"
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
        \"\"\"Apply the Adam update to a single parameter.

        ``scale`` is the global-norm clipping factor computed in :meth:`step`.
        \"\"\"
        if parameter.grad is None:
            return
        state = self.getState(parameter)
        gradient = parameter.grad * scale
""",
    "adam: global-norm clipping",
)

patch(
    ADAM,
    """        self._model: Optional[DenseNetwork] = None
        self.stepCount: int = 0""",
    """        self._model: Optional[DenseNetwork] = None
        self.stepCount: int = 0
        self.lastGradientNorm: float = 0.0
        self.lastClipScale: float = 1.0""",
    "adam: clip diagnostics",
)

print("done")
