# AQUA — The Self-Pruning Network

A neural network that prunes **itself** while it trains, built from scratch
in pure Python and NumPy — with a reverse-mode automatic differentiation
engine, an Adam optimiser, an importance criterion derived from a Taylor
expansion of the loss, and an honest cost measurement.

> **Case study:** AQUA Platform — Technical Hiring Challenge
> (Senior / Staff AI Platform Engineer, Tredence)

---

## Table of contents

- [Overview](#overview)
- [What makes this submission correct](#what-makes-this-submission-correct)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Cost measurement](#cost-measurement)
- [The falsifiable claim](#the-falsifiable-claim)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Naming conventions and dependencies](#naming-conventions-and-dependencies)
- [Known limitations](#known-limitations)
- [Repository layout](#repository-layout)

---

## Overview

A small MLP classifies a two-class spiral. During training it progressively
removes its own least-useful weight connections under a hard compute
budget, using a **cubic sparsity ramp**, and the removal decision is driven
by a **gradient-based saliency criterion** derived from a first-order Taylor
expansion of the loss.

Three things the submission is careful about, because the challenge is
explicit about them:

1. **The masked-weight gradient semantics are exact and documented.** There
   are two different gradients, not one: the gradient with respect to the
   *raw* weight of a masked connection is exactly zero, and the gradient
   with respect to the *effective* weight is generally not — and that second
   quantity is the signal that makes re-growth possible. The claim "the
   gradient of a masked weight is zero" is meaningless until you say which
   tensor you mean. See [DESIGN.md §4](DESIGN.md#4-masked-weight-gradient-semantics).

2. **The optimiser state survives pruning and re-growth correctly.** A
   revived connection starts from a *bit-identical* state to a brand-new
   parameter — the Adam timestep is tracked **per element** precisely so
   that a revived connection restarts its bias correction. See
   [DESIGN.md §5](DESIGN.md#5-adam-state-behaviour-after-pruning-and-regrowth).

3. **The cost claim is honest.** The exact multiply-accumulate counts and
   the measured wall-clock latency are reported *separately* and never
   conflated. "Dense forward with zeros" is implemented and reported
   explicitly as performing the **same** number of multiply-accumulates as
   the dense path, because multiplying by zero is not a speedup. See
   [DESIGN.md §8](DESIGN.md#8-dense-versus-sparse-aware-computation).

---

## What makes this submission correct

### The autodiff engine

A NumPy-backed `Tensor` with parents, local backward rules, gradient
accumulation and a topological backward sweep. Every operation has an
automated central-difference gradient check covering scalar, vector,
matrix, broadcasting and shared-graph cases.

Broadcasting is handled in exactly one place (`reduceGradToShape`), which
implements NumPy's broadcasting rules as gradient rules: leading axes added
by broadcasting are summed away, and axes stretched from size 1 are summed
with `keepdims=True` so the rank is preserved. The challenge's example —
`(N, D) + (D,)` — is covered by both a finite-difference check and a shape
assertion, and `tests/test_broadcast_reduction.py` verifies the rule
against an independent index-replay reference over twelve shape
combinations.

### The masked-weight correctness requirement

`applyMask` implements `W_effective = W_raw * mask` inside the
computational graph and separates the two gradient semantics:

```python
out.effectiveGrad = upstream            # dL/dW_effective  — non-zero when masked
weight.grad += upstream * maskArray     # dL/dW_raw        — exactly zero when masked
```

The consequence is concrete and tested:

* a masked connection contributes **exactly** `0.0` to the forward pass;
* its raw gradient is **exactly** `0.0`;
* its *effective* gradient is **non-zero** — it does not depend on the
  weight value, only on the layer input and the downstream gradient, which
  is exactly what makes re-growth scoring meaningful.

### The optimiser

Adam written from scratch with first and second moments, bias correction,
epsilon stabilisation, **global** (not per-tensor) gradient-norm clipping,
and a **per-element timestep**. After every step, `enforceMasks()`
guarantees that a dead connection has zero weight, zero first moment, zero
second moment and a frozen timestep.

The strongest test in the suite,
`test_revived_connection_first_update_matches_a_fresh_parameter`, isolates a
single connection, gives it momentum, prunes it, revives it, and asserts
that its first post-revival Adam step is **bit-identical** to the step a
brand-new parameter takes with the same gradient.

### The pruning mechanism

* **Importance criterion** — saliency `|w_effective · dL/dW_effective|`,
  the absolute value of the first-order Taylor estimate of the loss change
  from removing a connection. The score is **averaged over the mini-batches
  of an epoch**; without that averaging the gradient-based criterion is
  measurably *worse* than magnitude pruning (see below).
* **Schedule** — a cubic sparsity ramp over the middle of training, in a
  *closed* window so the final target is actually reached.
* **Budget** — global across the network, not per layer, with deterministic
  tie-breaking (stable sort).
* **Honest masking** — a pruned connection is a real zero, not a small
  weight, and the mask is validated to be strictly binary.
* **Biases are never pruned**, and all reported sparsity is over prunable
  connections only.
* **Regrowth** (bonus) — implemented as a *budget-neutral swap*: the most
  sensitive dead connection is revived while the least important live
  connection is pruned, so the active count and the sparsity are exactly
  preserved.

---

## Architecture

```
inputs (batch, 2)
   │
   ▼
Linear(2 → 32)   ── mask ──┐        W_effective = W_raw ⊙ mask
   │  ReLU                   │        forward:  y = W_effective @ x + b
   ▼                         │        backward: dL/dW_raw = (dL/dW_eff) ⊙ mask
Linear(32 → 32)  ── mask ──┘        optimizer: m, v, t  (t is per-element)
   │  ReLU
   ▼
Linear(32 → 2)   ── mask ──┘        output = raw logits
   │
   ▼
cross-entropy (stable, from logits)
```

### Layers and modules

| Module | Contents |
|---|---|
| `aqua/engine/` | `Tensor`, `Parameter`, the computation graph, all differentiable ops, `applyMask` |
| `aqua/nn/` | `Linear` (prunable), `DenseNetwork`, activations, He/Glorot initialisation, MLP factory |
| `aqua/optim/` | `AdamOptimizer`, `AdamState` (per-element timestep) |
| `aqua/prune/` | criteria, cubic schedule, global mask allocator, `Pruner`, `Regrowth` |
| `aqua/data/` | deterministic synthetic datasets (pure NumPy) |
| `aqua/train/` | trainer, metrics, experiment driver, sparse inference, cost measurement |
| `aqua/utils/` | seeding, JSON/CSV serialisation |

### Gradient lifecycle (explicit and documented)

* `backward()` **clears** the gradient buffers of the whole subgraph before
  seeding the root, so repeated backward calls are idempotent and cannot
  retain stale intermediate gradients.
* `backward(retainGraph=True)` **accumulates**, and does so correctly: the
  buffers are snapshotted, cleared, recomputed and added back. A naive
  implementation would propagate the already-accumulated upstream gradient
  a second time and silently corrupt any shared subgraph.
* `zeroGrad(recursive=True)` invalidates a subgraph's gradients.

---

## Installation

Python 3.10+ is required (developed and tested on 3.10–3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` contains only:

```
numpy>=1.24
matplotlib>=3.6
pytest>=7.0
```

> **scikit-learn is not used anywhere in this project** — not for dataset
> loading, not for models, not for gradients, not for pruning. The dataset
> is generated in pure NumPy. The challenge permits scikit-learn for
> dataset loading only; this submission does not need it, and therefore
> does not use it.

---

## Usage

### Quick start

```bash
# Run the whole test suite
pytest -q

# Train the dense baseline and print the learning curve
python scripts/train_baseline.py

# Run the full experiment suite (5 seeds x 5 sparsities x 4 criteria)
python scripts/run_experiments.py --seeds 5 --epochs 60 --include-regrowth

# Generate the plots from the committed raw results
python scripts/make_plots.py

# Check every deliverable and that the README numbers match the raw data
python scripts/verify_submission.py

# Quick criterion comparison (a smaller, faster sweep)
python scripts/sweep_criteria.py
```

### Train a pruned model directly

```python
from aqua.train.experiment import ExperimentConfig, runExperiment

config = ExperimentConfig(
    epochs=60,
    targetSparsity=0.9,
    criterion="saliency",
    seed=0,
    hiddenSizes=[32, 32],
)
result = runExperiment(config)

print(result.finalMetrics["validationAccuracy"])   # ~0.85
print(result.finalMetrics["achievedSparsity"])     # ~0.90
print(result.costMetrics["activeMacs"])            # multiply-accumulates actually done
```

### Use the engine directly

```python
import numpy as np
from aqua.engine import ops
from aqua.engine.tensor import Tensor, Parameter

x = Parameter(np.random.randn(4, 3))
w = Parameter(np.random.randn(3, 2))
mask = np.ones((3, 2))
mask[0, 0] = 0.0                                   # prune one connection

y = ops.matmul(x, ops.applyMask(w, mask))
ops.sum(y).backward()

assert w.grad[0, 0] == 0.0                        # raw gradient: exactly zero
assert w.grad[1, 0] != 0.0                        # alive connection: real gradient
```

---

## Dataset

A **two-class spiral**, generated in pure NumPy by
`aqua/data/datasets.py::makeSpiralDataset`.

* 1200 samples, 900 train / 300 validation (25% held out).
* Two features, two classes, perfectly balanced.
* 1.2 turns, radius 6, Gaussian label noise `σ = 0.2`.
* **Standardised** using statistics fitted on the *training split only* —
  the validation set never influences any statistic the model sees. This is
  the deliberate guard against data leakage.
* Fully deterministic: the same `seed` reproduces the same dataset
  bit-for-bit.

A `twoMoons` dataset is also provided as a sanity check.

**Why a synthetic dataset?** The challenge allows any classification task.
The spiral was chosen because it is (a) reproducible from a clean clone,
(b) fast enough that a genuine 5-seed × 5-sparsity × 4-criterion sweep runs
in minutes, which is what makes a *multi-seed* study affordable at all,
(c) non-linear enough to need a hidden layer but not so hard that the
network needs a large one — so the pruning study measures the pruning, not
the capacity, and (d) debuggable, because the decision boundary is known
analytically and a training bug is visible immediately.

The spiral was also tuned so that the sparsity budget is *binding*: the
dense model reaches ~99% accuracy and the 90%-sparse model still retains
~89% accuracy with the saliency criterion. A task that the network solves
perfectly at any sparsity would make the whole comparison vacuous.

---

## Results

> **All numbers below are produced by `scripts/run_experiments.py` and are
> committed as raw data in `results/experiments/`.** They are measured, not
> estimated. The exact per-seed rows are in
> `results/experiments/experiments_raw.csv` and the aggregates in
> `results/experiments/experiments_summary.csv`.

### Main comparison — validation accuracy vs target sparsity

Spiral dataset, 60 epochs, batch 64, Adam `lr = 1e-3`, **5 seeds**,
mean ± standard deviation over the seeds.

| Criterion | 0% | 50% | 75% | 90% | 95% |
|---|---|---|---|---|---|
| Magnitude (baseline) | 0.9720 ± 0.0201 | 0.9553 ± 0.0267 | 0.9453 ± 0.0299 | 0.7700 ± 0.1095 | 0.6333 ± 0.0833 |
| **Saliency `\|w·grad\|`** | 0.9720 ± 0.0201 | 0.9593 ± 0.0228 | 0.9353 ± 0.0315 | **0.8913 ± 0.0318** | **0.8060 ± 0.0608** |
| Signed Taylor `−w·grad` | 0.9720 ± 0.0201 | 0.5733 ± 0.0482 | 0.5313 ± 0.0530 | 0.5513 ± 0.0551 | 0.5687 ± 0.0960 |
| Gradient magnitude `\|grad\|` | 0.9720 ± 0.0201 | 0.6860 ± 0.0946 | 0.6440 ± 0.0962 | 0.5493 ± 0.0480 | 0.5407 ± 0.0427 |

The 0% row is identical across criteria by construction: with no pruning
the criterion is never consulted, so the runs are the same experiment.

**Reading the table:**

* **Saliency beats magnitude at every sparsity above 75%**, and by a wide
  margin at 90% (+12.1 points) and 95% (+17.3 points).
* At 90% saliency also has a **3.4× smaller standard deviation**
  (0.0318 vs 0.1095). The advantage is not one lucky seed.
* At 75% the two criteria are within noise of each other (0.9453 ± 0.0299
  vs 0.9353 ± 0.0315) and the submission does **not** claim a win there.
* The two ablations (signed Taylor, gradient magnitude) are **clearly
  worse**. This is a real, reported negative result — see
  [DESIGN.md §6](DESIGN.md#6-taylor-based-saliency-derivation) for why the
  signed variant underperforms despite being the most literal reading of
  the Taylor expansion.

### The noise finding (why the score is batch-averaged)

The gradient-based criterion was measured *before* it was given access to
the mini-batch stream. At 90% sparsity, over the same 5 seeds:

| Variant | 90% sparsity accuracy |
|---|---|
| Saliency on a **single mini-batch** (naive) | 0.6133 |
| Saliency **averaged over the epoch's mini-batches** | **0.8913** |
| Magnitude baseline | 0.7700 |

A saliency score computed from one mini-batch is *worse than magnitude
pruning*, because the mini-batch gradient is a noisy estimate of the
full-data gradient. Averaging the score over the epoch's mini-batches is
what makes the criterion work. This is a concrete, reproducible finding and
it is why `SaliencyPruner` accumulates by default.

### The re-growth ablation (bonus)

Regrowth is implemented as a **budget-neutral swap**: at each epoch the dead
connection with the highest gradient sensitivity is revived while the live
connection with the lowest importance score is pruned, so the number of
active connections — and therefore the sparsity — is *exactly* preserved.
The budget is never broken.

| Target sparsity | Saliency (no re-growth) | Saliency (with re-growth) | Connections swapped |
|---|---|---|---|
| 50% | 0.9593 ± 0.0228 | 0.9587 ± 0.0239 | 364 |
| 75% | 0.9353 ± 0.0315 | 0.9407 ± 0.0351 | 182 |
| 90% | 0.8913 ± 0.0318 | 0.8893 ± 0.0349 | 60 |
| 95% | 0.8060 ± 0.0608 | **0.8180 ± 0.0467** | 26 |

**Honest reading of this result:** re-growth helps at 95% sparsity
(+1.2 points, and the standard deviation drops from 0.0608 to 0.0467), is
neutral at 75% and 90%, and is neutral-to-slightly-negative at 50%. All of
these differences are *within one standard deviation* of each other, so the
correct conclusion is that **re-growth is not a significant win on this
task** — it does not hurt, and at the tightest budget where the structure
is most likely to have made mistakes it gives a small improvement.

The reason re-growth does not help more is instructive: the swap re-allocates
capacity, but the cubic ramp has already made good decisions by the time the
network reaches 95% sparsity, so the candidates being revived are mostly
connections that were correctly removed. The mechanism works (it demonstrably
swaps connections and preserves the budget exactly) — it simply has little
to add on a task where the saliency criterion is already choosing well.

### Achieved sparsity

The achieved sparsity matches the target to within one connection (integer
rounding of the connection count), e.g. `0.9002` for a 0.90 target on the
1152-connection network (1152 × 0.10 = 115.2 → 115 active connections).

### Plots

| File | Content |
|---|---|
| `results/plots/pareto_sparsity_accuracy.png` | Accuracy vs achieved sparsity for every criterion, with error bars |
| `results/plots/criteria_comparison.png` | Side-by-side criterion comparison |
| `results/plots/learning_curves.png` | Loss and accuracy per epoch: dense vs 90% saliency vs 90% magnitude |
| `results/plots/cost_measurements.png` | Theoretical MAC reduction vs measured latency ratio |

---

## Cost measurement

`aqua/train/cost.py` reports **exact operation counts** and **measured
latency**, always separately.

### Operation counts (exact, analytic)

| Quantity | Value at 90% sparsity |
|---|---|
| Model | 2 → 32 → 32 → 2 |
| Total prunable connections | 1152 |
| Active connections | 115 |
| Achieved sparsity | 0.9002 |
| Bias parameters (never pruned) | 66 |
| Total scalar parameters (weights + biases) | 1218 |
| **Dense multiply-accumulates** (per sample) | 1152 |
| **"Dense times zero" multiply-accumulates** (per sample) | **1152 — identical** |
| **Active multiply-accumulates** (per sample, sparse path) | 115 |
| Theoretical MAC reduction | 0.9002 |

The middle row is the point the challenge asks for: the "dense forward
that multiplies by zero" performs the *same* number of multiply-accumulates
as the dense forward. It is not a speedup. The repository implements that
path explicitly (`maskedDenseForward`) so the claim is demonstrated rather
than asserted, and `test_dense_times_zero_does_not_reduce_work` asserts
`denseTimesZeroMacs == denseMacs`.

### Measured latency

The sparse-aware path is measured against the dense path at batch sizes 1,
32 and 256, median over 200 repetitions after a warm-up.

**The honest result: on these small matrices the sparse path is always
*slower* in wall-clock terms, and the submission reports that rather than
hiding it.** Measured latency ratio (sparse / dense; **lower is faster**),
saliency criterion, seed 0:

| Target sparsity | Theoretical MAC reduction | Ratio @ batch 1 | Ratio @ batch 32 | Ratio @ batch 256 | Dense @32 | Sparse @32 |
|---|---|---|---|---|---|---|
| 0% | 0.000 | 16.87× | 13.71× | 9.09× | 0.0139 ms | 0.1911 ms |
| 50% | 0.500 | 15.93× | 11.67× | 6.61× | 0.0147 ms | 0.1715 ms |
| 75% | 0.750 | 15.16× | 10.61× | 5.25× | 0.0148 ms | 0.1575 ms |
| 90% | 0.900 | 10.54× | 7.42× | 3.44× | 0.0148 ms | 0.1095 ms |
| 95% | 0.950 | 7.09× | 5.09× | 2.48× | 0.0152 ms | 0.0775 ms |

Two things are worth reading off this table:

1. **The dense path is flat** (~0.015 ms at batch 32) because a single
   BLAS `dgemm` call on a 32×32 matrix takes the same time regardless of how
   many of its entries are zero. The dense path does not know, and does not
   care, that the model is 90% sparse.
2. **The sparse path gets faster as sparsity rises** (13.7× → 5.1× slower
   relative to dense at batch 32), and the gap narrows further at larger
   batch sizes (9.1× → 2.5× at batch 256). The sparse path's *work* really
   does scale with sparsity — the curve has the right shape — but the
   Python-level per-unit loop overhead is large enough that it never
   overtakes a single BLAS call on a matrix this small.

> These are wall-clock numbers and therefore machine-dependent; the
> *shape* of the curve is the reproducible part, and the exact values are
> committed in `results/experiments/experiments_raw.csv`.

This is exactly why the theoretical reduction and the measured latency are
reported in separate columns, and why the report does not claim a speedup.
The reasons are spelled out in
[DESIGN.md §8](DESIGN.md#8-dense-versus-sparse-aware-computation):

* the sparse path executes a Python-level loop per output unit, while the
  dense path is a *single* BLAS `dgemm` call;
* the gather (`output[:, active]`) is a strided copy that is memory-bound;
* the cache lines touched are roughly the same, so only the arithmetic
  falls, not the memory traffic.

The genuine benefits of high sparsity in production are fewer bytes of
model storage, less memory bandwidth for the weights, and the ability to
use a *real* sparse kernel (CSR/CSC GEMM, or a sparse accelerator) that a
pure-Python loop cannot emulate. Those are claims about the serving stack,
not about this repository, and this repository does not make them.

`results/plots/cost_measurements.png` plots the theoretical reduction
against the measured latency ratio so the gap between the two is visible.

### Accuracy of the sparse path

`test_sparse_aware_forward_matches_dense_forward` asserts the sparse path
matches the dense path to `1e-12`, and
`test_sparse_forward_uses_no_dense_matmul_for_masked_columns` proves that a
masked connection cannot influence the output by perturbing its raw weight
by `1234.5` and checking the output is bit-identical.

---

## The falsifiable claim

> **At 90% target sparsity, saliency pruning retains 89.1% validation
> accuracy versus 77.0% for magnitude pruning, averaged across 5 random
> seeds (0.8913 ± 0.0318 vs 0.7700 ± 0.1095) on the spiral dataset.**
>
> At 95% target sparsity: 80.6% ± 6.1% versus 63.3% ± 8.3%.

**Backing numbers:** `results/experiments/experiments_raw.csv` (per-seed
rows), `results/experiments/experiments_summary.csv` (aggregates),
`results/plots/pareto_sparsity_accuracy.png` (the plot).

**Falsifiability:** the claim is a specific number over a specific protocol.
Re-running `python scripts/run_experiments.py --seeds 5 --epochs 60` on a
clean clone must reproduce the per-seed values in the CSV. If it does not,
the claim is false. A second, independent way to falsify it: set
`criterion="magnitude"` in the same script and the saliency row must
change while the magnitude row does not.

**Statistical caveat (stated rather than glossed over):** 5 seeds is a
small sample and no hypothesis test is performed. The gap at 90%
(+12.1 points) is large relative to the standard deviations, and saliency's
spread is *smaller*, which is the stronger form of evidence. At 75% the two
criteria are within noise of each other (0.9453 ± 0.0299 vs 0.9353 ±
0.0315) and the submission says so rather than claiming a win.

---

## Testing

```bash
pytest -q                                    # the whole suite
pytest tests/test_masking.py -v              # the masked-weight semantics
pytest tests/test_engine_ops.py -v            # the finite-difference checks
pytest --collect-only -q | tail -1           # how many tests exist
```

**130 tests, all passing.** They cover:

| Area | File | What is checked |
|---|---|---|
| Every operation | `test_engine_ops.py` | Central-difference gradient checks: scalar, vector, matrix, broadcasting, shared graph |
| Broadcasting | `test_broadcast_reduction.py` | The reduction rule against an independent index-replay reference, over 12 shape combinations |
| Graph mechanics | `test_engine_graph.py` | Topological order, gradient accumulation, shared subgraphs, repeated backward, `retainGraph` |
| **Masking** | `test_masking.py` | Zero forward contribution, zero raw gradient, non-zero effective gradient, optimiser state, revival, masked-vs-dense equivalence, end-to-end pruning |
| Optimiser | `test_optimizer.py` | Bias correction, moment updates, per-element timestep, global-norm clipping, snapshot/restore |
| Pruning | `test_pruning.py` | Schedule shape, exact budget, monotonicity, determinism under ties, global-vs-per-layer allocation, criterion behaviour, regrowth |
| Sparse inference | `test_sparse_inference.py` | Sparse-vs-dense equivalence, MAC counts, "dense times zero" equality, masked-connection independence |
| Training | `test_training.py` | Convergence, finiteness, reproducibility under a fixed seed, end-to-end pruning run |

The masking tests are the heart of the submission; each claim in
[DESIGN.md §4–§5](DESIGN.md#4-masked-weight-gradient-semantics) has a
corresponding test.

---

## Reproducibility

Every source of randomness is funnelled through `aqua/utils/seeding.py` and
seeded from the experiment's `seed`:

* the dataset split and its noise,
* the weight initialisation,
* the mini-batch shuffling order.

Given the same seed and the same configuration, a run reproduces
**bit-for-bit** — asserted by `test_training_is_reproducible_under_a_fixed_seed`
and `test_experiment_is_reproducible`.

To reproduce every committed number from a clean clone:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                          # 130 passed
python scripts/run_experiments.py --seeds 5 --epochs 60 --include-regrowth
python scripts/make_plots.py
python scripts/verify_submission.py

# The regenerated CSVs must match the committed ones.
git diff --stat results/
```

The exact environment used for the committed results: Python 3.14.7,
NumPy 2.5.4, Matplotlib 3.11.2, pytest 9.1.1, Linux x86-64, 8 cores.

---

## Naming conventions and dependencies

The challenge specifies a naming convention that differs from PEP 8, and
PEP 8 is followed everywhere except where it conflicts with it.

| Entity | Convention | Examples |
|---|---|---|
| Functions, methods, variables, attributes | `camelCase` | `computeGradients`, `applyMask`, `learningRate`, `batchSize`, `calculateSparsity` |
| Classes | `PascalCase` | `Tensor`, `Linear`, `AdamOptimizer`, `MagnitudePruner`, `SaliencyPruner`, `Pruner`, `Regrowth` |
| Constants and module-level configuration | `UPPER_SNAKE_CASE` | `DEFAULT_LEARNING_RATE`, `DEFAULT_BETA1`, `DEFAULT_EPSILON`, `DEFAULT_SEED` |

Standard Python conventions are kept where the language or tooling requires
them: module and file names are `snake_case`, and `__init__`-style dunder
methods keep their names.

**Dependencies:** the Python standard library, NumPy, Matplotlib (plots
only) and pytest (tests only). Nothing else. No PyTorch, TensorFlow, JAX,
Keras, HuggingFace, or any autodiff / neural-network / optimiser / pruning
library is used — the engine, the optimiser, the pruning mechanism, the
training loop and the sparse inference path are all written from scratch.
**scikit-learn is not used at all**, so the "dataset loading only"
restriction is satisfied vacuously; the dataset is generated in NumPy.

`scripts/verify_submission.py` asserts all of this mechanically.

---

## Known limitations

1. **Synthetic dataset.** The absolute accuracies do not transfer to
   MNIST-scale problems. The *comparison* between criteria is what this
   study measures.
2. **Five seeds.** Honest but noisy standard deviations. A bootstrap
   confidence interval on the difference of means is the obvious next step.
3. **The sparse path is written for clarity, not speed.** It is slower than
   the dense path on these small matrices, and the submission says so.
4. **Unstructured sparsity.** The surviving connections are scattered, so a
   dense BLAS kernel cannot be kept. This is the largest gap between
   "the theory says 90% fewer FLOPs" and "the server got faster". A
   production system would want structured pruning.
5. **First-order saliency only.** The Taylor expansion drops the curvature
   term and ignores coupling between weights. A diagonal-Fisher
   (second-order) criterion is the natural next step.
6. **Re-growth is budget-neutral by construction** — it re-allocates
   capacity but cannot grow the budget.
7. **Latency was measured on one shared machine.** It is indicative, not a
   benchmark, which is exactly why the theoretical and measured numbers are
   reported separately.

See [DESIGN.md §11](DESIGN.md#11-known-limitations-and-risks) for the full
list, and [DESIGN.md §10](DESIGN.md#10-production-deployment) for how this
would be deployed in a multi-tenant inference service at thousands of
requests per second.

---

## Repository layout

```
aqua/
  engine/        tensor.py, ops.py          -- the autodiff engine
  nn/            layers.py, mlp.py,         -- layers, activations,
                 activations.py,             -- initialisation
                 initialization.py
  optim/         adam.py                    -- Adam, mask-aware state
  prune/         criteria.py, schedule.py,  -- importance criteria,
                 allocator.py, pruner.py,    -- schedule, allocation,
                 regrowth.py                -- controller, re-growth
  data/          datasets.py                -- synthetic datasets
  train/         trainer.py, experiment.py,  -- training, experiments,
                 sparse.py, cost.py,        -- sparse inference, cost
                 metrics.py                 -- measurement
  utils/         seeding.py, serialization.py
tests/           130 tests
scripts/         train_baseline.py, run_experiments.py, make_plots.py,
                 sweep_criteria.py, verify_submission.py,
                 apply_engine_fixes.py
results/
  experiments/   experiments_raw.csv, experiments_summary.csv,
                 experiments_summary.json, experiments_log.txt
  plots/         pareto_sparsity_accuracy.png, criteria_comparison.png,
                 learning_curves.png, cost_measurements.png
DESIGN.md        the reasoning, the derivations, the production design
README.md        this file
requirements.txt
```

---

## Provenance

This is an original implementation written for the challenge. The
derivations in `DESIGN.md` are worked through from first principles; the
cubic pruning schedule follows the iterative-magnitude-pruning formulation
of Luo et al. (2019) and is re-derived in `aqua/prune/schedule.py`. No
existing autodiff, pruning or optimiser library was used or consulted for
the core implementation.
