"""Tests for the graph mechanics: ordering, accumulation, shared subgraphs.

These are the properties of the reverse-mode engine that are easy to get
subtly wrong and that a gradient check alone would not catch (a gradient
check on a shared subgraph *does* catch it, but these tests localise the
failure and document the intended behaviour).
"""

from __future__ import annotations

import numpy as np
import pytest

from aqua.engine import ops
from aqua.engine.tensor import Parameter, Tensor


def test_shared_subgraph_accumulates_gradients() -> None:
    """A tensor used twice must receive the *sum* of both contributions."""
    value = Parameter(np.array([2.0, 3.0]))
    # y = value + value  =>  dL/dvalue = 2
    out = ops.add(value, value)
    ops.sum(out).backward()
    assert value.grad is not None
    assert np.allclose(value.grad, np.array([2.0, 2.0]))


def test_shared_subgraph_in_a_diamond() -> None:
    """A diamond graph must accumulate the gradient at the shared node."""
    base = Parameter(np.array([1.0, 2.0]))
    left = ops.multiply(base, 3.0)
    right = ops.multiply(base, 5.0)
    out = ops.add(left, right)
    ops.sum(out).backward()
    # dL/dbase = 3 + 5 = 8
    assert base.grad is not None
    assert np.allclose(base.grad, np.array([8.0, 8.0]))


def test_topological_order_is_leaves_first() -> None:
    """The backward order must respect the data dependencies."""
    a = Parameter(np.array([1.0]))
    b = ops.multiply(a, 2.0)
    c = ops.add(b, 1.0)
    order = c._buildTopologicalOrder()
    assert order.index(a) < order.index(b)
    assert order.index(b) < order.index(c)
    # Every node appears exactly once.
    assert len(order) == len({id(node) for node in order})


def test_repeated_backward_does_not_retain_stale_gradients() -> None:
    """A second backward pass must not double the gradient.

    The engine clears the gradient buffers of every node in the graph before
    seeding the root, so ``backward()`` is idempotent.  This is what keeps
    the pruning machinery from reading a gradient that has been accumulated
    over several steps by accident.
    """
    value = Parameter(np.array([3.0]))
    out = ops.multiply(value, 4.0)
    ops.sum(out).backward()
    first = value.grad.copy()
    ops.sum(out).backward()
    second = value.grad.copy()
    assert np.allclose(first, second)
    assert np.allclose(first, np.array([4.0]))


def test_retain_graph_accumulates_gradients() -> None:
    """``retainGraph=True`` must accumulate rather than replace."""
    value = Parameter(np.array([3.0]))
    out = ops.multiply(value, 4.0)
    ops.sum(out).backward(retainGraph=True)
    ops.sum(out).backward(retainGraph=True)
    assert value.grad is not None
    assert np.allclose(value.grad, np.array([8.0]))


def test_zero_grad_clears_the_whole_subgraph() -> None:
    value = Parameter(np.array([3.0]))
    out = ops.multiply(value, 4.0)
    ops.sum(out).backward()
    assert value.grad is not None
    out.zeroGrad()
    assert value.grad is None
    assert out.grad is None


def test_backward_from_non_scalar_requires_a_seed() -> None:
    value = Parameter(np.ones((2, 2)))
    out = ops.multiply(value, 2.0)
    with pytest.raises(ValueError):
        out.backward()


def test_backward_with_explicit_seed() -> None:
    value = Parameter(np.ones((2, 2)))
    out = ops.multiply(value, 2.0)
    out.backward(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert value.grad is not None
    assert np.allclose(value.grad, np.full((2, 2), 2.0))


def test_seed_shape_is_validated() -> None:
    value = Parameter(np.ones((2, 2)))
    out = ops.multiply(value, 2.0)
    with pytest.raises(ValueError):
        out.backward(np.ones((3, 3)))


def test_constants_do_not_require_gradients() -> None:
    """A constant operand must not be given a gradient buffer."""
    value = Parameter(np.array([1.0, 2.0]))
    out = ops.add(value, 5.0)
    ops.sum(out).backward()
    assert value.grad is not None
    assert np.allclose(value.grad, np.array([1.0, 1.0]))


def test_deep_chain_backward() -> None:
    """A long chain must backpropagate correctly."""
    value = Parameter(np.array([1.0]))
    current = value
    for _ in range(50):
        current = ops.multiply(current, 1.5)
    ops.sum(current).backward()
    assert value.grad is not None
    assert np.allclose(value.grad, np.array([1.5**50]))


def test_scalar_tensor_item() -> None:
    tensor = Tensor(np.array([3.5]))
    assert tensor.item() == 3.5
    with pytest.raises(ValueError):
        Tensor(np.ones((2,))).item()


def test_tensor_rejects_non_numeric_data() -> None:
    with pytest.raises(TypeError):
        Tensor(np.array([{"a": 1}], dtype=object))


def test_integer_data_is_promoted_to_float() -> None:
    tensor = Tensor(np.array([1, 2, 3], dtype=np.int64))
    assert tensor.data.dtype == np.float64
