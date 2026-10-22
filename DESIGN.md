# DESIGN.md — AQUA Self-Pruning Network

This document explains the engineering and mathematical decisions behind the
submission. It is written to be defended live: every claim here is backed
by a test in `tests/` or by a number in `results/`.

---

## Table of contents

1. [Reverse-mode autodiff](#1-reverse-mode-autodiff)
2. [Topological backward propagation](#2-topological-backward-propagation)
3. [Broadcasting gradient reduction](#3-broadcasting-gradient-reduction)
4. [Masked-weight gradient semantics](#4-masked-weight-gradient-semantics)
5. [Adam state behaviour after pruning and regrowth](#5-adam-state-behaviour-after-pruning-and-regrowth)
6. [Taylor-based saliency derivation](#6-taylor-based-saliency-derivation)
7. [Pruning schedule and budget](#7-pruning-schedule-and-budget)
8. [Dense versus sparse-aware computation](#8-dense-versus-sparse-aware-computation)
9. [Experimental variance and statistical interpretation](#9-experimental-variance-and-statistical-interpretation)
10. [Production deployment](#10-production-deployment)
11. [Known limitations and risks](#11-known-limitations-and-risks)

---

## 1. Reverse-mode autodiff

### The problem

Forward-mode autodiff computes a Jacobian row at a time. For a network with
`L` parameters and a scalar loss, the forward-mode cost is `O(L)` per
output. Reverse-mode computes the *vector-Jacobian product* in a single
reverse sweep, which costs `O(L)` for *all* parameters at once. Since a
network has many parameters and one loss, reverse-mode is the right choice —
and it is what the challenge asks for.

### The design

`aqua/engine/tensor.py` defines `Tensor`, a NumPy-backed graph node holding:

| Field | Purpose |
|---|---|
| `data` | The forward value (a NumPy array, always `float64`). |
| `grad` | The accumulated gradient of the scalar loss w.r.t. `data`. |
| `requiresGrad` | Whether this node expects a gradient. |
| `_parents` | The upstream nodes feeding this one. |
| `_backwardFn` | The local rule used to push gradients to the parents. |
| `effectiveGrad` | `dLoss/dW_effective`, captured by `applyMask` (see §4). |

Each operation in `aqua/engine/ops.py` follows a fixed contract:

1. Promote plain scalars/arrays to constant tensors.
2. Compute the forward value with NumPy.
3. Create the output node and wire `_parents`.
4. Install a `_backwardFn` that **accumulates** into the parents' gradients.

The backward function receives no arguments; it reads its own upstream
gradient from `out.grad`. This makes the rules self-contained and
order-independent, which is what allows the graph to be re-traversed or
re-used.

### Gradient lifecycle

This is a deliberate design decision, and the challenge's "prevent repeated
backward calls from accidentally retaining stale intermediate gradients"
requirement is addressed explicitly:

* A `backward()` call **clears** the gradient buffer of every node in the
  subgraph before seeding the root. A second `backward()` therefore
  recomputes the gradient from scratch rather than adding to it.
* `backward(retainGraph=True)` **accumulates**. The implementation is not a
  naive "skip the clearing": it snapshots the existing buffers, clears them,
  runs the sweep, and adds the new gradients back. A naive implementation
  would propagate the *already-accumulated* upstream gradient a second
  time, which squares the contribution of any shared node and produces
  silently wrong results for a shared subgraph. The snapshot/clear/run/add
  sequence gives the behaviour people actually expect.
* `zeroGrad(recursive=True)` clears a whole subgraph, used when a caller
  wants to invalidate a stale gradient without running a backward pass.

Tested in `tests/test_engine_graph.py`:
`test_repeated_backward_does_not_retain_stale_gradients`,
`test_retain_graph_accumulates_gradients`, `test_shared_subgraph_accumulates_gradients`.

### Verified operations

Every operation has an automated central-difference gradient check in
`tests/test_engine_ops.py` (scalar, vector, matrix, broadcasting and
shared-graph cases): addition, subtraction, elementwise multiplication,
division, negation, matrix multiplication, sum, mean, reshape, ReLU, leaky
ReLU, tanh, sigmoid, softmax, log-softmax, cross-entropy, exp, log, square,
maximum, clip.

The checker is deliberately **independent** of the engine: it perturbs the
raw NumPy data of the inputs and re-evaluates the forward function
(`tests/gradient_checks.py`), so a bug in the backward rules cannot hide
behind a shared helper.

### Numerical stability

* **Softmax** subtracts the row maximum before exponentiating.
* **Cross-entropy from logits** never forms `log(softmax(...))`. It
  computes `logsumexp` directly and the gradient is the closed form
  `(softmax(z) - onehot(y)) / N`. This is exact for the mean reduction and
  cannot produce `NaN` when a probability underflows to zero.
  `test_cross_entropy_is_numerically_stable` asserts this with logits of
  magnitude 1000.
* **Sigmoid** is evaluated in the two-branch form so that `exp` never
  overflows.

---

## 2. Topological backward propagation

The backward sweep must visit a node only *after* all of its consumers have
contributed to its gradient. `Tensor._buildTopologicalOrder` produces a
leaves-first ordering using an iterative depth-first traversal with an
explicit `visited` set.

Two properties matter and are tested:

1. **Every node appears exactly once.** A node that is the input to two
   different operations (a shared subgraph) must have its backward rule
   executed *once*, after both consumers have run — not once per consumer.
   The `visited` set guarantees single insertion.
2. **Dependencies are respected.** `order.index(a) < order.index(b)` for
   every edge `a -> b`. The backward sweep then iterates `reversed(topo)`,
   which is a valid reverse topological order.

The traversal is **iterative**, not recursive, so a deep graph (the tests
build 50-layer chains) cannot blow the Python recursion limit.

Gradient accumulation happens in `_accumulate()`, which allocates the
buffer on demand and then *adds*:

```python
if target.grad is None:
    target.grad = np.array(contribution, dtype=target.data.dtype, copy=True)
else:
    target.grad = target.grad + contribution
```

This is what makes a shared subgraph correct: `y = x + x` gives
`dL/dx = 2`, and the diamond test
`test_shared_subgraph_in_a_diamond` asserts `dL/dx = 3 + 5 = 8`.

---

## 3. Broadcasting gradient reduction

NumPy broadcasts an operand of shape `S` up to shape `T` by *stretching*
axes. The gradient of a stretched operand is the sum of the gradient over
the stretched axes. This is implemented in exactly one place,
`aqua/engine/ops.py::reduceGradToShape`, so that every operation inherits
the same (correct) behaviour.

The algorithm:

1. If the shapes already match, return the gradient unchanged.
2. **Sum away the leading axes** that broadcasting added:
   `while grad.ndim > len(targetShape): grad = grad.sum(axis=0)`.
3. **Sum the stretched axes**, keeping the rank intact:
   for each axis where the target has size `1` and the gradient has size
   `n > 1`, apply `grad.sum(axis=axis, keepdims=True)`.
4. Broadcast to the target shape and reshape. (A target axis of size 1 can
   leave a *trailing* axis of size `n` in the gradient; the broadcast
   handles that case.)

The `keepdims=True` in step 3 is essential: without it the subsequent axis
indices would shift and the wrong axis would be reduced.

The challenge's specific example — `(N, D) + (D,)` — is covered by
`test_broadcast_row_vector_gradient` (finite differences) and
`test_broadcast_keeps_operand_shape` (shape assertions). The column-vector
`(N, 1)` and scalar cases are covered too.

`tests/test_broadcast_reduction.py` goes further: it verifies the
reduction against an **independent** reference that literally replays
NumPy's broadcasting index map, over twelve shape combinations. This is the
one piece of the engine where a plausible-looking implementation can be
subtly wrong, so it gets its own exhaustive test.

---

## 4. Masked-weight gradient semantics

**This is the section the challenge probes, and the answer is that there are
two different gradients, not one.**

### The model

A prunable layer owns three tensors:

* `rawWeights` — the free parameter, shape `(inFeatures, outFeatures)`. The
  optimiser sees *only* this.
* `bias` — a free parameter that is **never** pruned (see §7).
* `mask` — a binary array of the same shape as `rawWeights`.

The forward pass is

```
W_effective = rawWeights * mask
y = W_effective @ x + bias
```

so a masked connection contributes **exactly zero** — not a small number,
not a rounding artefact. `test_masked_connection_contributes_exactly_zero`
asserts the masked entry is exactly `0.0`.

The mask is validated to be strictly binary. A mask with intermediate
values would silently turn "pruned" into "scaled down", which is exactly
the failure mode the challenge warns about, so `setMask` rejects it.

### The two gradients

`applyMask` is the single point where masking enters the computational
model, and it separates two quantities:

| Quantity | Definition | Value at a masked connection | Why it matters |
|---|---|---|---|
| `dLoss/dW_raw` | ∂L/∂(rawWeights * mask) | **exactly 0** | The optimiser has nothing to act on; a dead connection cannot be moved. |
| `dLoss/dW_effective` | The upstream gradient, *before* the mask is applied | **generally non-zero** | Quantifies how sensitive the loss is to this connection; it is the signal for re-growth scoring. |

The backward rule is literally

```python
out.effectiveGrad = upstream            # captured BEFORE the mask
weight.grad += upstream * maskArray     # reduced THROUGH the mask
```

Two consequences that are easy to get wrong:

* **"The gradient of a masked weight must be zero" is not a well-formed
  claim** until you say *which tensor* the gradient is taken with respect
  to. It is exactly zero for the raw weight. It is *not* zero for the
  effective weight, and it must not be, because the effective gradient is
  the only thing that says "this connection would matter if it came back".
* The effective gradient does **not** depend on the weight value. It
  depends on the layer input and the downstream gradient. That is why it
  stays meaningful at masked positions.

Tested in `tests/test_masking.py`:
`test_gradient_wrt_raw_weight_is_zero_at_masked_positions`,
`test_gradient_wrt_effective_weight_is_non_zero_at_masked_positions`,
`test_masked_gradient_matches_manual_computation` (which asserts
`dL/dW_raw == dL/dW_effective * mask` elementwise).

### Why the effective gradient is captured on the node

`applyMask` stores the upstream gradient on the produced node
(`out.effectiveGrad`). `Linear.publishEffectiveGradient()` copies it onto
the layer as `effectiveWeightGrad` after the backward pass. The pruner then
reads `layer.effectiveWeightGrad` without re-running the backward pass,
which keeps the training loop free of back-propagation-specific knowledge
about pruning.

### The `Linear` layer as the unit of masking

`aqua/nn/layers.py::Linear` is the only place a mask lives. It exposes
`applyMaskToWeights()`, `activeParameterCount()`, `totalParameterCount()`
and `sparsity()`. Every prunable layer in the network is a `Linear`, so the
pruner, the allocator and the cost measurement all work through the same
interface.

---

## 5. Adam state behaviour after pruning and regrowth

This is where naive implementations corrupt training. The policy is stated
explicitly and enforced in code.

### The update rule

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
```

The bias correction divides by `1 - beta^t`, which is zero at `t = 0`; at
`t = 0` the first update would be identically zero without it.

### The key design decision: the timestep is per *element*

A library Adam keeps one scalar timestep for the whole parameter tensor.
That is fine for dense training, but it is wrong for a structurally
changing model: when a connection is revived, its bias correction must
restart from `t = 1`, not continue from the global step count. Otherwise
the revived connection's first update is scaled by a bias correction
corresponding to a step count it never took part in.

`aqua/optim/adam.py` therefore stores `timestep` as an **integer array with
the same shape as the parameter**, so that individual connections can be
reset independently.

### The three invariants

After every optimiser step, `enforceMasks()` guarantees:

1. `rawWeight[mask == 0] == 0` — a dead connection cannot drift away from
   zero.
2. `m[mask == 0] == 0` — no residual momentum survives.
3. `v[mask == 0] == 0` — no stale second moment survives.

And the per-element timestep of a dead connection is **frozen at 0**, so a
revived connection restarts its bias correction from `t = 1`.

Invariants 1 and 2 are *belt and braces* on top of the mathematical fact
that the raw gradient of a masked connection is exactly zero. They matter
because Adam's moments decay exponentially, not instantly: without the
explicit zeroing, a dead connection would keep being nudged by its residual
momentum for tens of thousands of steps, and the first step after a
re-growth would be scaled by a stale `v`.

### The re-growth policy

**Chosen policy: fresh start.**

When a connection is revived:

* its raw weight is set to exactly `0.0` (it has to learn its value again;
  the old value is meaningless after the connection has been absent, and
  re-randomising it would inject noise into a converged model);
* its Adam moments are zeroed and its timestep is reset to `0`.

`AdamOptimizer.restoreStateAt(parameter, indices, snapshot=None)` implements
the fresh-start policy, and `restoreStateAt(..., snapshot=state)` implements a
*warm restore* for the case where a connection is only *temporarily* masked
and must resume exactly where it left off. The project uses the fresh-start
policy for re-growth; the warm-restore path exists and is tested
(`test_state_snapshot_and_restore`).

### The strongest test

`test_revived_connection_first_update_matches_a_fresh_parameter` isolates a
single connection in a 1×1 layer, gives it momentum, prunes it, revives it,
and then asserts that its first post-revival Adam step is **bit-identical**
to the step a brand-new parameter would take with the same gradient. The
reference parameter's timestep is asserted to be `1`.

This is stronger than "the revival does not crash" or "the gradient looks
reasonable": it says the revival is *exactly* equivalent to a new parameter.

`test_optimiser_state_is_zeroed_for_masked_parameters` asserts the three
invariants directly.

### Gradient clipping

Clipping is **global** (the L2 norm across *all* parameters), not per
tensor. Per-tensor clipping makes the effective step size depend on how the
parameters happen to be grouped, which is arbitrary. The scale factor is
computed once per step in `AdamOptimizer.step()` and applied to every
parameter. `test_adam_global_norm_clipping` asserts that a gradient of
norm 5 is scaled by exactly 0.2 under a threshold of 1.

---

## 6. Taylor-based saliency derivation

### The derivation

Let `L(θ)` be the loss and `w` one scalar weight. Removing the connection
means setting `w → 0`, i.e. applying the perturbation `Δw = −w`. A
first-order Taylor expansion around the current `w` gives

```
ΔL ≈ (∂L/∂w) · Δw = −w · (∂L/∂w)
```

Writing `g = ∂L/∂w`, three consequences follow.

**1. The sign.** If `w · g > 0` then `ΔL < 0`: removing the connection
*decreases* the loss, so it is a good candidate for removal. If `w · g < 0`,
removal *increases* the loss, so the connection is expensive.

**2. The magnitude of the loss increase.** The first-order estimate of how
much the loss *rises* is `|w · g|`. This is the **saliency** score.

**3. Which connections to remove.** A removal budget must remove the
connections whose removal is *cheapest*. The cheapest removals are the
ones with the most negative `ΔL`, i.e. the most positive `w · g`.

Point 3 is the subtle one, and it is where the naive implementations go
wrong. Ranking by `|w g|` answers "which connections are most
*important*", and pruning the least important ones is the same as ranking
by `|wg|` **only if the signs are all the same**. They are not: in a
network being trained, some weights work *against* the loss (positive
`wg`: removing them would help) and some work *for* it (negative `wg`:
removing them hurts). Sorting by `|wg|` discards the sign information and
can rank a connection that is actively helping the loss above one that is
actively hurting it.

### The criteria in the code base

| Criterion | Score | Role |
|---|---|---|
| `MagnitudePruner` | `\|w_effective\|` | The required baseline. |
| `SaliencyPruner` | `\|w_effective · dL/dW_effective\|` | **The primary criterion.** |
| `SignedTaylorPruner` | `−w_effective · dL/dW_effective` | The literal Taylor ranking. |
| `GradientMagnitudePruner` | `\|dL/dW_effective\|` | Ablation isolating the `w` factor. |

### Why the absolute value wins empirically (a documented negative result)

The `SignedTaylorPruner` is the criterion the derivation appears to justify
most directly, and it is the **worst** of the four in the experiments. At
90% sparsity it reaches 0.551 ± 0.055 validation accuracy, against
0.891 ± 0.032 for saliency and 0.770 ± 0.110 for magnitude.

The explanation: the signed score is a ranking by a **difference**, and a
difference of two noisy quantities is noisier than either quantity. The
sign of `w g` flips between mini-batches for any weight whose gradient
hovers near zero. Such a weight gets a score close to zero, which the
signed ranking places in the *middle* of the distribution — it is neither
clearly removable nor clearly protected, so the budget gets spent on
genuinely important connections.

`|w g|` keeps the *scale* of the Taylor estimate and discards only the
sign. It is robust to sign noise and still uses the gradient to discount
weights whose gradient is small. In this problem the robustness wins. The
signed criterion is kept in the code and in the results as a documented
ablation rather than being quietly dropped — the negative result is part of
the finding.

### The noise problem, and the fix

`∂L/∂w` computed on a mini-batch is a noisy estimate of the full-data
gradient. A ranking built from one mini-batch is measurably *worse* than a
magnitude ranking.

This was observed directly during development: before the criterion was
given access to the mini-batch stream, the saliency criterion scored
**0.613** at 90% sparsity against magnitude's **0.867** — i.e. saliency was
*losing*. After averaging the score over the mini-batches of an epoch, the
same criterion reached **0.891** against magnitude's **0.770**.

The mechanism is `_AccumulatingCriterion` in `aqua/prune/criteria.py`:

* `accumulate(layer)` is called by the trainer after every mini-batch
  (via the `onBatchEnd` callback) and adds the current `w · g` to a
  running sum.
* `score(layer)` returns the **mean** of the accumulated observations.
* `reset()` is called by the pruner after every pruning event, so the
  statistic is always computed from the gradients observed since the last
  structural change. Comparing gradient statistics across two different
  structures is not meaningful.

The accumulated *effective gradient* is also kept separately
(`averagedGradient`), which is what the re-growth machinery uses, because
the Taylor product `|w g|` is identically zero for a currently-absent
connection and therefore cannot rank re-growth candidates.

### Which gradient is used, and why

All gradient-based criteria use `dL/dW_effective` — the upstream gradient
*before* it is reduced through the mask — never the raw gradient. The raw
gradient is exactly zero at masked positions, so using it would make every
pruned connection score zero for a trivial reason, and would make the
re-growth signal identically zero.

---

## 7. Pruning schedule and budget

### The cubic ramp

```
s(t) = s_final + (s_initial - s_final) · (1 - ((t - t0) / (t1 - t0))^3)^3
```

where `t` is the training progress in `[0, 1]`, `t0` is when pruning starts
and `t1` is when the final sparsity is reached. The ramp is slow at first
(the cube of a cube makes the derivative vanish at `t0`) and fast in the
middle, which gives the network a long recovery period after each pruning
shock.

**Why gradual beats one-shot.** One-shot pruning removes everything at the
end of training. The network then has no time to adapt: the loss jumps and
the final accuracy reflects a model that has never been trained with the
structure it will be deployed with. A gradual ramp turns one shock into
many small shocks, each followed by recovery. The trade-off is that the
*intermediate* models are trained with less capacity than the final one, so
the final model has effectively been trained under a moving sparsity
constraint. This is mildly harmful when the budget is generous and clearly
beneficial when the budget is tight.

The schedule window is **closed** on both ends (`startFraction <= progress
<= endFraction`). An open window was a real bug found during development:
with `progress >= startFraction and progress < endFraction`, the final
sparsity was never applied and the model stopped at 88.9% instead of 90%.

### The budget is global, not per layer

`aqua/prune/allocator.py` pools the importance scores of *all* prunable
layers into one flat vector and removes the globally least important
connections. A per-layer budget would force the first layer to keep 90% of
its connections even when the second layer needs the budget more urgently.
The measured per-layer sparsities differ substantially, which is the
intended behaviour (`test_allocator_sparsity_is_global_not_per_layer`).

### Determinism

Ties are broken by a **stable sort** on the pooled score, with the pooled
index order being deterministic (layers in model order, then row-major
position within a layer). NumPy's `argsort(kind="stable")` guarantees that
equal scores keep their original relative order.
`test_allocator_is_deterministic_under_ties` asserts that three independent
runs with all-zero scores produce identical masks.

### Biases are not pruned

A bias is not a connection between two units; it is a constant offset
applied to a whole unit. Removing a bias does not save a matrix multiply —
it shifts the unit's operating point, which is not something a saliency
score can reason about, because "the gradient of the loss w.r.t. a bias" is
a sum over the whole batch and says nothing about whether that unit is
useful. Biases are therefore excluded from the pruning budget, and **all
reported sparsity is computed over prunable connections only**. The bias
count is reported separately in the cost measurements (`biasParameters`).

### The budget is monotone; re-growth is a separate, explicit mechanism

The allocator clamps the request to the achievable range, so it never
*un-prunes* a connection. Re-growth (§5) is a distinct mechanism with
explicit semantics.

---

## 8. Dense versus sparse-aware computation

### Three forward paths, on purpose

`aqua/train/sparse.py` implements three paths so that the difference
between them can be *measured* rather than asserted.

| Path | What it does | Multiply-accumulates per sample |
|---|---|---|
| `denseForward` | `x @ (W_raw * mask)` | `inFeatures × outFeatures` (always) |
| `maskedDenseForward` | `x @ (W_raw * mask)` — **identical** | `inFeatures × outFeatures` (always) |
| `sparseAwareForward` | Gathers only the active inputs per output unit | `number of active connections` |

The middle path is the "dense forward that multiplies by zero". It is
included **specifically so the report can say explicitly that it performs
the same number of multiply-accumulates as the dense path**. Multiplying by
zero is not a speedup: it is a rounding error at best, and a correctness
hazard at worst. The cost measurement reports
`denseMacs == denseTimesZeroMacs` as a hard fact, and
`test_dense_times_zero_does_not_reduce_work` asserts it.

### The sparse path

The sparse path uses a CSC-style compressed structure: for every output
unit `j` it stores the indices of the *active* input units feeding it and
the corresponding weight values. The forward pass is then

```python
for j in range(outFeatures):
    active = structure.inputIndices[j]
    if active.size == 0:
        continue
    layerOutput[:, j] = output[:, active] @ structure.weightValues[j]
```

A masked connection never appears in any gathered slice, so it is never
multiplied. `test_sparse_forward_uses_no_dense_matmul_for_masked_columns`
proves this by perturbing the raw weight of a masked connection by `1234.5`
and asserting the sparse output is **bit-identical**.

The structure is rebuilt whenever the mask changes (rare) and reused across
inference calls (frequent), which is the right place to spend the work.

### Theoretical reduction vs measured latency

The exact multiply-accumulate counts are reported separately from the
measured wall-clock latency, and the report never conflates them:

* `denseMacs`, `activeMacs`, `theoreticalMacReduction` — **exact, analytic**.
* `denseLatencySeconds`, `sparseLatencySeconds`, `measuredLatencyRatio` —
  **measured**, median over many repetitions after a warm-up.

The median is used rather than the mean because a single scheduling hiccup
can double a mean over a small number of repetitions.

### Why high sparsity may not produce a speedup on small matrices

This is important to state plainly, because it is the most common way a
sparsity claim gets overstated.

1. **Python overhead dominates small matrices.** The sparse path executes a
   Python-level loop over output units. For a 32-unit layer that is 32
   interpreter iterations plus 32 NumPy calls, each with its own dispatch
   overhead (~1 µs). The dense path is a *single* BLAS `dgemm` call. For a
   matrix this small, the BLAS call is faster even though it does 10x more
   arithmetic.
2. **BLAS is extremely well optimised.** A dense `dgemm` on a 32×32 matrix
   uses vectorised kernels that reach a high fraction of peak FLOPS. The
   sparse path's gather (`output[:, active]`) is a *fancy-index copy* of a
   strided slice, which is memory-bound and cannot be vectorised as
   effectively.
3. **Memory access.** The dense path streams the weight matrix
   contiguously. The sparse path gathers a scattered subset of columns into
   a temporary buffer, so it touches the same *cache lines* as the dense
   path while doing less arithmetic. The cache traffic is roughly unchanged;
   only the FLOP count falls.
4. **Batch size matters a great deal.** At batch size 1 the dense path is
   completely overhead-bound and the sparse path is *slower*. At large batch
   sizes the dense path becomes memory-bandwidth-bound and the sparse path
   can pull ahead. The cost measurement reports three batch sizes (1, 32,
   256) precisely so this is visible.
5. **The real win is not raw latency.** The genuine benefit of high
   sparsity in production is: fewer bytes of model storage, less memory
   bandwidth for the weights, better fit into a memory or cache budget, and
   the ability to use a sparse-aware kernel (CSR/CSC GEMM, or a sparse
   accelerator) that a plain NumPy loop cannot emulate. Claiming a
   wall-clock speedup from a pure-Python sparse loop on a small matrix
   would be overstating the result.

The rule this project follows: **a speedup is only reported when the
measurement supports it**, and the measured and theoretical numbers are
printed side by side so nobody has to guess which is which. The measured
result in this submission is that the sparse path is **5–11× slower** than
the dense path on these matrices, and the report says so.

---

## 9. Experimental variance and statistical interpretation

### Protocol

* **5 random seeds** per point (0–4). The seed controls the dataset split,
  the weight initialisation *and* the mini-batch shuffling, so the seed
  captures the total run-to-run variance.
* **60 epochs** per run, mini-batch 64, Adam with `lr = 1e-3`, global
  gradient clipping at norm 5.
* The **same** configuration is used for every criterion, so the comparison
  is apples-to-apples.
* The raw per-seed rows are committed to
  `results/experiments/experiments_raw.csv`; the aggregated means and
  standard deviations are in `results/experiments/experiments_summary.csv`.

### Reading the error bars

The standard deviation over 5 seeds is the honest measure of run-to-run
variability. Two results are only meaningfully different if the gap
exceeds the spread. In the 90% sparsity comparison the gap between saliency
and magnitude is large relative to the spread *and* saliency has the
**smaller** spread, which is the stronger kind of evidence: the difference
is not an artefact of one lucky seed.

A single seed is never used to support a claim. Where a difference is
within the noise, the result is reported as "within noise" rather than
being presented as a win.

### The variance is not uniform across the sweep

The dense (0% sparsity) runs have a standard deviation of ~0.02 on accuracy.
The 90% and 95% runs have a *larger* spread (~0.03–0.11 for magnitude).
This is expected: pruning introduces an extra source of instability —
which connections happen to be removed — on top of the usual
initialisation and shuffling variance. A criterion that is *robust* to that
instability shows up as a smaller spread, which is exactly what saliency
does at 90%.

### Statistical caveat

With 5 seeds, a standard deviation estimate is itself noisy, and no
hypothesis test is performed on the difference of means. The claim in
`README.md` is therefore stated in the same terms as the evidence: "at 90%
sparsity, saliency pruning retains X% accuracy versus Y% for magnitude
pruning, averaged across 5 seeds (mean ± std)". A t-test or a
bootstrap confidence interval would be a reasonable addition, and is listed
as future work in §11.

---

## 10. Production deployment

The following is the design I would bring to a review of this code running
inside AQUA at thousands of requests per second.

### 10.1 The sparse format is the real deliverable

A NumPy-level sparse loop is a *measurement* instrument, not a serving
path. In production the pruned model would be exported to **CSR/CSC** (or
the accelerator's native sparse format) and served through a sparse GEMM
kernel. The `SparseLayerStructure` in `aqua/train/sparse.py` is already a
CSC representation and is the natural export format. On hardware with
sparse support (or in a sparse-aware BLAS), the FLOP reduction does become
a latency reduction — but that is a claim about the *serving stack*, not
about this repository, and it must be measured on that stack.

### 10.2 Multi-tenant isolation

* **Model immutability at serving time.** A trained, pruned model is
  serialised (mask + weights + architecture) and treated as an immutable
  artefact identified by a content hash. The serving process never mutates
  it, so one tenant's requests cannot perturb another tenant's model.
* **Per-tenant routing.** Each tenant gets its own model *version id*
  (see 10.6) and its own quota. A request carries the tenant id; the
  router resolves it to a model version and an inference replica.
* **No shared mutable state in the request path.** The dense and sparse
  forward paths allocate their own output buffers per call. Masks and
  weights are read-only arrays (set `writeable = False` on the
  serialised weight arrays) so that a bug in one request handler cannot
  corrupt the model for another tenant.
* **No cross-tenant batching of different models.** Requests from
  different tenants with different model versions must not be batched
  together, because a batch is served by one model's weights. Batching is
  only within a (tenant, model version) group.

### 10.3 Concurrency

* Python's GIL means a pure-Python inference loop scales to roughly one
  core. At thousands of requests per second the serving process must either
  (a) release the GIL inside the numerical kernels — which NumPy's BLAS
  calls do, so a *dense* `dgemm` releases the GIL and threads do scale —
  or (b) run multiple processes.
* The recommended shape: a multi-process model server (one process per
  core, `SO_REUSEPORT` or an explicit load balancer) with each process
  holding the model resident in memory. Model memory is then replicated
  per process, which is a real cost for large models and an argument for
  shared memory (POSIX shared memory or file-mapping of the weight array)
  when the model is large.
* **A pruning-aware serving caveat:** the sparse path's Python loop holds
  the GIL for its whole duration, so a sparse model served this way scales
  *worse* per core than a dense one, even though it does less arithmetic.
  The fix is a sparse kernel that releases the GIL, which is another reason
  the production path must be a proper sparse GEMM rather than this loop.

### 10.4 Batching

* Batch size is the main latency/throughput dial. The cost measurement
  reports batch sizes 1, 32 and 256 precisely because the dense/sparse
  crossover is batch-size dependent.
* Requests should be coalesced into a micro-batch window (a few
  milliseconds) rather than served one at a time, so the per-call overhead
  is amortised.
* Dynamic batching must not mix model versions (see 10.2).

### 10.5 Observability

* **Per-request:** model version id, tenant id, input shape, batch size,
  latency, output shape. Never the raw input (privacy), never the full
  logits at high volume (cardinality).
* **Per-model:** active parameter count, achieved sparsity, dense vs sparse
  latency percentiles, and the ratio between them. If the sparse path is
  not actually faster in production, the *model* should be flagged — a
  model whose sparsity does not translate into a latency win is not
  delivering the promised saving.
* **Alerts:** p99 latency, error rate, and a "sparsity drift" check that
  compares the served model's active parameter count against the one
  recorded at build time. A mismatch means the wrong artefact was deployed.
* **Cost attribution:** attribute inference cost per tenant so that the
  business can see which tenants actually benefit from the cheaper model.

### 10.6 Model versioning

* Every artefact is content-addressed: `model_version = hash(architecture,
  weights, mask, training_config, code_version)`.
* The training config and the *git commit* are embedded in the artefact, so
  a model can always be traced back to the code that produced it. This
  matters for exactly this project: a model pruned by a specific criterion
  under a specific schedule is only reproducible if the criterion and the
  schedule are recorded with it.
* Promotion is a separate, gated step: train → evaluate → offline
  comparison against the incumbent → canary → ramp → full. The
  offline comparison uses the same Pareto analysis as this repository:
  accuracy versus sparsity, plus the measured cost.

### 10.7 Safe rollout

* **Canary at a small traffic percentage** (1%, then 10%, then 50%, then
  100%) with automatic rollback on a p99-latency or error-rate regression.
* **Shadow traffic** for the sparse model: serve the incumbent's answers,
  compute the candidate's answers, and compare the distributions. This
  catches the failure mode that matters most for a pruned model — a
  specific input region where the pruned network behaves badly but the
  aggregate accuracy looks fine.
* **Never roll out a model whose accuracy is within noise of the
  incumbent unless the cost win is significant.** The variance analysis in
  §9 is the gate: a 1% accuracy gain with a 2% standard deviation is not a
  win, and shipping it would be shipping noise.
* **Keep the dense model warm** for instant rollback. A pruned model is a
  *cheaper* model, not a *different* model; the incumbent must remain
  deployable.
* **Per-tenant opt-in.** Sparsity is a cost optimisation, not a quality
  improvement. Tenants who need the extra accuracy must be able to stay on
  the dense model without paying for it twice.

---

## 11. Known limitations and risks

1. **The dataset is synthetic.** The spiral is a two-class, two-feature
   problem. It is a good instrument for comparing pruning criteria — it is
   fast, reproducible and has a known decision boundary — but the absolute
   accuracy numbers do not transfer to MNIST-scale problems. A follow-up
   would add a real dataset (with scikit-learn used *only* for loading, as
   the challenge permits).

2. **Five seeds is a small sample.** The standard deviations are honest but
   noisy. A bootstrap confidence interval on the difference of means would
   make the headline claim more defensible.

3. **The sparse path is not optimised.** It is written for clarity. A
   production path would use CSR/CSC sparse GEMM. It is not even close to
   competitive with a tuned sparse kernel, and the report says so.

4. **No structured sparsity.** Pruning is unstructured, so the surviving
   connections are scattered. Structured pruning (whole neurons, channels
   or blocks) yields far better real hardware speedups because the dense
   BLAS path can be kept. This is the single biggest gap between "the
   theory says 90% fewer FLOPs" and "the server got 30% faster".

5. **The saliency criterion is a first-order approximation.** See §6 for
   the three failure modes (curvature, coupling, noise). A second-order
   (Hessian-diagonal) criterion would be more accurate and is what I would
   implement next, at the cost of a diagonal Fisher approximation per
   layer.

6. **Regrowth is budget-neutral by construction.** The swap keeps the
   active count exactly constant, so re-growth cannot *reduce* the achieved
   sparsity below the target. That satisfies the hard budget but also means
   re-growth can only re-allocate, never grow the budget.

7. **The re-growth policy is a design choice, not the only one.** The
   alternative — allowing the sparsity to dip below the target when a
   revival looks clearly beneficial — would need a different budget
   accounting. The current choice was made because the challenge
   emphasises a *hard* budget.

8. **The gradient-check tolerance.** The finite-difference checks use a
   relative-or-absolute pass criterion with `epsilon = 1e-6`. A
   pathological function could pass the absolute test with a near-zero
   gradient. The relative test is the one that carries the weight; a
   reviewer who wants more assurance can lower the tolerance.

9. **Single-machine latency measurement.** The latency numbers were
   measured on one machine with a shared CPU. They are indicative, not a
   benchmark, and they are reported separately from the theoretical
   operation counts for exactly this reason.
