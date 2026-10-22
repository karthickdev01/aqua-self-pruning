"""Core :class:`Tensor` abstraction for the AQUA reverse-mode autodiff engine.

The tensor is a thin, NumPy-backed wrapper that carries three pieces of state:

``data``
    The forward value, a NumPy array.
``grad``
    The accumulated gradient of the scalar loss with respect to ``data``.
``parents`` / ``backwardFn``
    The edges of the computation graph plus the local rule used to push
    gradients from this node back to its parents.

The engine is deliberately small.  It is not a framework: there is no tape, no
dispatch table, and no operator overloading magic beyond the handful of
dunder methods that make the code in :mod:`aqua.nn` read like normal
mathematics.

Gradient lifecycle
------------------
Every node owns exactly one gradient buffer.  A fresh :meth:`Tensor.backward`
call clears the gradient buffers of every node reachable from the node the
backward pass starts at, *before* seeding the root gradient.  This makes
repeated backward calls idempotent with respect to stale intermediate values,
which matters a great deal in this project because the pruning machinery
captures gradient information between steps.

When a gradient must be accumulated across several backward passes (for
example when scoring candidate re-grown connections), pass
``retainGraph=True`` to :meth:`Tensor.backward`.  In that mode the gradient
buffers are *not* cleared, so gradients accumulate additively as usual.

Masked weights
--------------
The engine supports masking through :func:`aqua.engine.ops.applyMask`.  That
function implements ``W_effective = W_raw * mask`` and deliberately captures
the *unmasked* upstream gradient (``dLoss/dW_effective``) on the produced
node.  This distinction is the heart of the pruning correctness argument and
is documented in detail in ``DESIGN.md``.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np

__all__ = ["Tensor", "Parameter"]


def _asArray(data: object) -> np.ndarray:
    """Convert ``data`` to a float NumPy array without copying when possible."""
    if isinstance(data, Tensor):  # pragma: no cover - defensive
        raise TypeError("Cannot construct a Tensor directly from another Tensor; use the ops module.")
    array = np.asarray(data)
    if array.dtype == np.dtype(object):  # pragma: no cover - defensive
        raise TypeError("Tensor data must be numeric, got dtype=object.")
    if array.dtype.kind in "biu":
        # Integer and boolean data is promoted to float64 so that gradients
        # never truncate and so that the dtype is uniform across the graph.
        array = array.astype(np.float64)
    return array


class Tensor:
    """A node in the AQUA computation graph.

    Parameters
    ----------
    data:
        Anything NumPy can turn into an array.  Integers are promoted to
        ``float64`` so that gradients never truncate.
    requiresGrad:
        When ``True`` this tensor is a leaf that expects a gradient.
    name:
        Optional human readable label used in error messages.
    """

    __slots__ = (
        "data",
        "grad",
        "requiresGrad",
        "name",
        "_parents",
        "_backwardFn",
        "effectiveGrad",
        "_retainGraph",
    )

    def __init__(
        self,
        data: object,
        requiresGrad: bool = False,
        name: Optional[str] = None,
    ) -> None:
        self.data: np.ndarray = _asArray(data)
        self.grad: Optional[np.ndarray] = None
        self.requiresGrad: bool = bool(requiresGrad)
        self.name: Optional[str] = name

        self._parents: tuple[Tensor, ...] = ()
        self._backwardFn: Optional[Callable[[], None]] = None

        # dLoss/dW_effective captured by `applyMask`.  This stays populated even
        # for connections that are currently masked out, which is precisely what
        # makes re-growth scoring possible.
        self.effectiveGrad: Optional[np.ndarray] = None

        # When True the graph keeps its gradient buffers between backward calls.
        self._retainGraph: bool = False

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying NumPy array."""
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return int(self.data.ndim)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(self.data.size)

    def item(self) -> float:
        """Return the value of a single-element tensor as a Python float."""
        if self.data.size != 1:
            raise ValueError("item() is only valid for tensors with exactly one element.")
        return float(self.data.reshape(-1)[0])

    def detach(self) -> np.ndarray:
        """Return the raw NumPy array (no copy)."""
        return self.data

    def numpy(self) -> np.ndarray:
        """Return the raw NumPy array (no copy)."""
        return self.data

    def zeroGrad(self, recursive: bool = True) -> None:
        """Set this tensor's gradient buffer to ``None``.

        Parameters
        ----------
        recursive:
            When ``True`` the whole subgraph feeding this tensor is cleared as
            well.  This is the default because leaving stale gradients in
            intermediate nodes is the classic source of "works on the first
            step, wrong on the second" bugs.
        """
        self.grad = None
        if recursive:
            for parent in self._parents:
                parent.zeroGrad(recursive=True)

    def _buildTopologicalOrder(self) -> list[Tensor]:
        """Return every node in the graph, leaves first.

        A node appears exactly once even when it is used by several consumers,
        which is what makes gradient accumulation work for shared subgraphs.
        """
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]

        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in visited:
                    stack.append((parent, False))

        return order

    def backward(
        self,
        gradient: Optional[np.ndarray] = None,
        retainGraph: bool = False,
    ) -> None:
        """Run the reverse-mode backward pass.

        Parameters
        ----------
        gradient:
            Seed gradient with respect to ``self``.  Defaults to ``ones`` when
            ``self`` is a scalar.
        retainGraph:
            When ``True`` gradient buffers are *not* cleared first, so this
            call accumulates on top of the previous one.
        """
        topo = self._buildTopologicalOrder()

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

        if gradient is None:
            if self.data.size != 1:
                raise ValueError(
                    "A gradient must be supplied when back-propagating from a "
                    f"non-scalar tensor (shape {self.shape})."
                )
            self.grad = np.ones_like(self.data)
        else:
            seed = np.asarray(gradient, dtype=self.data.dtype)
            if seed.shape != self.data.shape:
                raise ValueError(
                    f"Seed gradient shape {seed.shape} does not match tensor shape {self.shape}."
                )
            self.grad = seed.astype(self.data.dtype, copy=True)

        for node in reversed(topo):
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
    # Operator sugar
    # ------------------------------------------------------------------
    def __add__(self, other: object) -> "Tensor":
        from aqua.engine.ops import add

        return add(self, other)

    def __radd__(self, other: object) -> "Tensor":
        from aqua.engine.ops import add

        return add(other, self)

    def __sub__(self, other: object) -> "Tensor":
        from aqua.engine.ops import subtract

        return subtract(self, other)

    def __rsub__(self, other: object) -> "Tensor":
        from aqua.engine.ops import subtract

        return subtract(other, self)

    def __mul__(self, other: object) -> "Tensor":
        from aqua.engine.ops import multiply

        return multiply(self, other)

    def __rmul__(self, other: object) -> "Tensor":
        from aqua.engine.ops import multiply

        return multiply(other, self)

    def __truediv__(self, other: object) -> "Tensor":
        from aqua.engine.ops import divide

        return divide(self, other)

    def __rtruediv__(self, other: object) -> "Tensor":
        from aqua.engine.ops import divide

        return divide(other, self)

    def __neg__(self) -> "Tensor":
        from aqua.engine.ops import negate

        return negate(self)

    def __matmul__(self, other: object) -> "Tensor":
        from aqua.engine.ops import matmul

        return matmul(self, other)

    def __rmatmul__(self, other: object) -> "Tensor":
        from aqua.engine.ops import matmul

        return matmul(other, self)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        label = f" name={self.name!r}" if self.name else ""
        return f"Tensor(shape={self.shape}, requiresGrad={self.requiresGrad}{label})"


class Parameter(Tensor):
    """A :class:`Tensor` that is meant to be optimised.

    Parameters behave exactly like tensors; the subclass exists so that
    parameter discovery in :mod:`aqua.nn` is unambiguous and so that a stray
    intermediate can never be handed to the optimiser by accident.
    """

    __slots__ = ()

    def __init__(self, data: object, name: Optional[str] = None) -> None:
        super().__init__(data, requiresGrad=True, name=name)
