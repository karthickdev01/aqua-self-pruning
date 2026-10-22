# Project Instructions — AQUA Self-Pruning Neural Network

Act as a Senior/Staff AI Platform Engineer, Python architect, numerical computing expert, and technical interview mentor.

Build a complete, reproducible Python project for the AQUA Self-Pruning Network technical hiring challenge. Follow the attached case study as the primary specification.

## 1. General coding rules

- Use Python 3.10+ and NumPy for the neural network, autodiff engine, optimizer, and pruning implementation.
- Use camelCase for all user-defined variables, function names, method names, and local identifiers.
- Use PascalCase for class names.
- Use UPPER_SNAKE_CASE for constants.
- Follow standard Python conventions only where required by Python itself, external APIs, or tool configuration. Document any unavoidable exceptions.
- Use clear, descriptive names. Avoid unnecessary abbreviations.
- Add type hints to public functions and methods.
- Add meaningful docstrings to classes and public methods.
- Follow PEP 8 for formatting, except where it conflicts with the explicitly requested naming convention.
- Use modular, readable, maintainable code. Avoid giant files, unnecessary abstractions, duplicated logic, and premature optimization.
- Handle numerical edge cases, invalid inputs, and errors explicitly.
- Never silently ignore exceptions or return fabricated results.
- Do not introduce unnecessary dependencies.

## 2. Strict library restrictions

Allowed:
- Python standard library
- NumPy
- Matplotlib for plots
- pytest for testing
- Scikit-learn only if needed to load a standard dataset; never use it for model training, gradients, optimizers, or pruning.

Do not use PyTorch, TensorFlow, JAX, Keras, HuggingFace, or any automatic differentiation, neural network, optimizer, or pruning library for the core implementation.

Implement the following from scratch:
- Tensor abstraction and computation graph
- Reverse-mode automatic differentiation
- Forward and backward operations
- MLP layers and activation functions
- Adam optimizer
- Pruning and masking
- Training and evaluation loops
- Sparse-aware inference
- Metrics and cost measurement

## 3. Architecture and project structure

Create a clean project structure with separate modules for:

- engine: Tensor, computation graph, differentiable operations, and backward propagation
- nn: Linear layer, activations, MLP, and parameter management
- optim: Adam optimizer and optimizer-state management
- prune: Importance criteria, pruning schedules, masks, and optional regrowth
- train: Mini-batch training, evaluation, experiments, and reproducibility
- tests: Unit tests, gradient checks, masking tests, optimizer tests, and integration tests
- scripts: Training, evaluation, sparsity sweeps, and plotting
- results: Raw experiment results and generated plots
- README.md: Setup, usage, results, limitations, and reproduction instructions
- DESIGN.md: Mathematical derivations and engineering decisions
- requirements.txt: Minimal dependencies

You may refine the structure if there is a clear technical reason. Keep module responsibilities separate.

## 4. Naming conventions and code quality

Use camelCase consistently for identifiers such as:
- learningRate
- batchSize
- computeGradients()
- applyMask()
- updateParameters()
- calculateSparsity()

Use PascalCase for classes such as:
- Tensor
- Linear
- AdamOptimizer
- MagnitudePruner
- SaliencyPruner

Use UPPER_SNAKE_CASE for constants such as:
- RANDOM_SEED
- DEFAULT_LEARNING_RATE

Avoid one-letter identifiers except for standard mathematical notation in short equations or tightly scoped numerical loops.

Do not rename public APIs unnecessarily after they are established. Keep imports explicit and avoid circular dependencies.

## 5. Autodiff engine requirements

Implement a NumPy-backed Tensor class with:
- Data
- Gradient storage
- Parent tracking
- Backward propagation functions
- Gradient accumulation
- Topological ordering
- Correct handling of shared subgraphs
- Broadcasting-aware gradient reduction

Implement at least:
- Addition
- Subtraction
- Elementwise multiplication
- Matrix multiplication
- Division, if needed
- Negation
- Sum and mean reductions
- Reshape, if needed
- ReLU
- Tanh or sigmoid
- Stable softmax and cross-entropy, with mathematically correct gradients

Prevent repeated backward calls from accidentally retaining stale intermediate gradients. Define and document a clear gradient lifecycle.

Every operation must have automated finite-difference gradient checks. Include scalar, vector, matrix, broadcasting, and shared-graph test cases.

## 6. Critical masked-gradient and optimizer-state requirements

Treat masking as an explicit part of the computational model, not merely as setting weights to zero once.

Maintain an explicit binary mask for each prunable weight tensor.

The forward pass must use effective weights:

effectiveWeights = rawWeights * mask

Ensure masked connections contribute exactly zero to the forward pass.

Define the gradient semantics precisely:
- Gradients with respect to effective weights may remain nonzero for masked connections, representing their potential utility if reintroduced.
- Gradients with respect to raw weights must respect the masking transformation.
- The optimizer must not update masked parameters through residual momentum or stale Adam moments.
- Masked parameters must remain exactly zero in the effective forward computation after every update.

Choose and document one consistent policy for permanently frozen parameters versus temporarily masked parameters.

If regrowth is implemented, preserve separate raw parameters and masks, explicitly define how candidate importance is measured, and reset or correctly manage optimizer state when a connection is revived.

Write tests proving that masking, backward propagation, optimizer updates, pruning, and regrowth do not corrupt training.

Do not claim a masked gradient must always be zero without specifying which tensor the gradient is taken with respect to.

## 7. MLP, optimizer, and training

Implement a small MLP using composable layers.

Implement Adam from scratch, including:
- First and second moments
- Bias correction
- Epsilon stabilization
- Learning-rate configuration
- Per-parameter state
- Correct masked-parameter handling
- State reset or restoration policies for revived connections

Use a deterministic classification dataset. Prefer a synthetic two-dimensional classification dataset if it makes debugging and reproducibility easier.

Implement mini-batch training and report:
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Epoch duration
- Numerical stability checks

Use suitable weight initialization, such as He initialization for ReLU layers and Xavier initialization for tanh layers. Explain the mathematical rationale.

Avoid data leakage between training and validation sets. Fix random seeds and document them.

## 8. Self-pruning algorithm

Implement at least two criteria:

1. Magnitude baseline: absolute weight magnitude.
2. Gradient-based saliency: absolute weight multiplied by the gradient of the loss with respect to the effective weight.

Derive and document the Taylor approximation:

DeltaLoss ≈ -weight * gradient

for removing an individual weight, with the sign and assumptions explained carefully. For a small perturbation setting a weight to zero, the first-order approximation is:

DeltaLoss ≈ -weight * gradient

Use the absolute value of this first-order estimate as a saliency score, and discuss why it is only an approximation and can be noisy.

Use a gradual pruning schedule, preferably a cubic sparsity ramp, and explain its parameters.

At every pruning event:
- Compute importance scores consistently.
- Prune the globally least-important eligible connections.
- Respect the target sparsity budget.
- Update masks.
- Enforce masks after optimizer steps.
- Track achieved sparsity and active parameter counts.

Ensure that biases are excluded from pruning unless there is a documented reason otherwise.

Make pruning deterministic under fixed seeds and tie-breaking rules.

## 9. Honest sparse inference and cost measurement

Implement both:
- A normal dense forward pass for the baseline.
- A sparse-aware forward path that skips masked connections rather than multiplying every connection by zero.

Use a straightforward, correct sparse-aware implementation first. Optimize only after measuring it.

Report:
- Total prunable parameters
- Active parameters
- Achieved sparsity
- Estimated dense multiply-accumulate operations
- Estimated active multiply-accumulate operations
- Dense inference latency
- Sparse-aware inference latency
- Model accuracy

Clearly distinguish theoretical operation-count reductions from measured wall-clock speedups.

Do not claim a speedup unless measurements support it. Explain Python overhead, NumPy behavior, memory access, and why high sparsity may not yield lower latency on small matrices.

## 10. Experiments and required evidence

Run target sparsity experiments at:
- 0%
- 50%
- 75%
- 90%
- 95%

Compare magnitude pruning and gradient-based saliency pruning at matching achieved sparsity levels.

Use multiple random seeds, preferably at least five for final experiments, subject to the runtime budget.

Report mean and standard deviation for accuracy, achieved sparsity, and latency where appropriate.

Generate and commit:
- A sparsity-versus-accuracy Pareto plot
- Raw experiment results in CSV or JSON
- A comparison plot or table for both pruning criteria
- Cost and active-parameter measurements
- Learning curves for loss and accuracy

Clearly label preliminary results versus completed experiments. Never fabricate results or claim that an experiment ran when it did not.

## 11. Testing requirements

Create automated tests for:
- Every implemented operation
- Broadcasting gradients
- Matrix multiplication gradients
- Shared computation graphs
- Gradient accumulation
- Finite-difference gradient checking
- Adam bias correction
- Masked forward computation
- Masked gradients
- Optimizer-state handling
- Pruning budget and exact achieved sparsity
- Sparse-aware versus dense-forward numerical equivalence
- Model training convergence
- Reproducibility under fixed seeds
- End-to-end training and pruning

Run the complete test suite and fix failures before claiming completion.

## 12. Documentation requirements

README.md must include:
- Project overview
- Architecture
- Installation
- Usage commands
- Dataset details
- Naming conventions and dependencies
- Explicit statement that scikit-learn, if used, is only for dataset loading
- Training and pruning methodology
- Test commands
- Reproducibility instructions
- Actual experiment results
- Cost-measurement methodology
- Known limitations

DESIGN.md must explain:
- Reverse-mode autodiff
- Topological backward propagation
- Broadcasting gradient reduction
- Masked-weight gradient semantics
- Adam state behavior after pruning and regrowth
- Taylor-based saliency derivation
- Pruning schedule and budget
- Dense versus sparse-aware computation
- Experimental variance and statistical interpretation
- Production deployment at thousands of requests per second
- Multi-tenant isolation, concurrency, batching, observability, model versioning, and safe rollout

## 13. Git and delivery

Maintain a meaningful, incremental Git commit history. Do not squash all changes into one commit.

Use clear commit messages describing individual milestones.

Do not include secrets, credentials, local environment files, or generated caches in the repository.

Provide commands to:
- Create the environment
- Install dependencies
- Run tests
- Train the baseline
- Run pruning experiments
- Generate plots
- Reproduce final results

Do not claim to have pushed to GitHub unless the push actually succeeded.

## 14. Working method

Do not generate the entire project blindly in one step.

Work in the following phases:
1. Convert the case study into a checklist and identify mathematical risks.
2. Design the architecture and explain important decisions.
3. Implement the autodiff engine.
4. Test every operation with finite differences.
5. Implement and test masked gradients and optimizer-state semantics.
6. Implement the MLP and Adam.
7. Train and validate the baseline.
8. Implement pruning and compare importance criteria.
9. Implement sparse-aware inference and cost measurements.
10. Run reproducible multi-seed experiments.
11. Generate plots and raw results.
12. Write README.md and DESIGN.md.
13. Run all tests and perform a final audit.
14. Prepare me for the live technical defense.

At every phase:
- Explain what was implemented and why.
- Show the exact commands to run.
- Show the expected test results without pretending they were observed.
- Wait for actual output when execution or environment access is unavailable.
- Fix issues before advancing.
- Identify assumptions and unresolved risks.
- Never hide failures or invent benchmark numbers.

My goal is to submit an original, correct, reproducible solution that I understand well enough to explain and extend live during the interview.