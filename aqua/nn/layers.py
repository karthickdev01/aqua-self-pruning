"""Prunable :class:`Linear` layer.

This layer is the heart of the AQUA challenge.  It owns three tensors:

``rawWeights``
    The free parameter of shape ``(inFeatures, outFeatures)``.  The optimiser
    sees *this* tensor and nothing else.
``bias``
    A free parameter of shape ``(outFeatures,)``.  Biases are **never**
    pruned (see :mod:`aqua.prune` for the rationale).
``mask``
    A binary array of shape ``(inFeatures, outFeatures)``.  ``0`` means the
    connection is structurally removed.

The forward pass is

``y = (rawWeights * mask) @ x + bias``

so a masked connection contributes *exactly* zero -- not a small number, not
a numerical artefact -- to the forward computation.

Gradient semantics
------------------
``applyMask`` separates two different quantities (see
:func:`aqua.engine.ops.applyMask`):

``dLoss/drawWeights``
    ``(dLoss/dW_effective) * mask``.  Exactly zero at masked positions, so
    the optimiser sees no signal to move a dead connection.

``dLoss/dW_effective``
    The sensitivity of the loss to the *effective* weight, computed with the
    mask removed.  This is generally **non-zero** even at masked positions and
    is the signal used to score candidates for re-growth.

The layer stores the effective gradient after every forward pass in
``effectiveWeightGrad`` so that the pruner can read it without re-running the
backward pass.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from aqua.engine import ops
from aqua.engine.tensor import Parameter, Tensor
from aqua.nn.activations import Activation, Identity
from aqua.nn.initialization import initializeWeight, zeros

__all__ = ["Linear", "DenseNetwork"]


class Linear:
    """A fully connected, prunable layer.

    Parameters
    ----------
    inFeatures:
        Size of the layer input.
    outFeatures:
        Size of the layer output.
    bias:
        Whether to add a learnable bias.  Biases are excluded from pruning.
    activation:
        Optional activation applied *after* the affine map.  The default is
        the identity, which keeps the layer composable.
    rng:
        Optional NumPy generator used for the weight initialisation.  Passing
        a seeded generator makes the whole network reproducible.
    """

    def __init__(
        self,
        inFeatures: int,
        outFeatures: int,
        bias: bool = True,
        activation: Optional[Activation] = None,
        rng: Optional[np.random.Generator] = None,
        name: Optional[str] = None,
    ) -> None:
        if inFeatures <= 0 or outFeatures <= 0:
            raise ValueError("inFeatures and outFeatures must be positive.")
        self.inFeatures = int(inFeatures)
        self.outFeatures = int(outFeatures)
        self.useBias = bool(bias)
        self.activation: Activation = activation if activation is not None else Identity()
        self.name = name or f"linear_{inFeatures}x{outFeatures}"

        self.rawWeights: Parameter = Parameter(
            initializeWeight((self.inFeatures, self.outFeatures), self.activation.initialisationKey, rng),
            name=f"{self.name}.rawWeights",
        )
        self.bias: Optional[Parameter] = (
            Parameter(zeros((self.outFeatures,)), name=f"{self.name}.bias")
            if self.useBias
            else None
        )

        # Structural mask: 1.0 = connection alive, 0.0 = connection removed.
        self.mask: np.ndarray = np.ones((self.inFeatures, self.outFeatures), dtype=np.float64)

        # dLoss/dW_effective captured during the last backward pass.
        self.effectiveWeightGrad: Optional[np.ndarray] = None
        # The raw gradient of the last backward pass (already mask-reduced).
        self.rawWeightGrad: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def parameters(self) -> list[Parameter]:
        """Return the trainable parameters owned by this layer."""
        params: list[Parameter] = [self.rawWeights]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def prunableParameters(self) -> list[Parameter]:
        """Return the parameters that are eligible for pruning (weights only)."""
        return [self.rawWeights]

    # ------------------------------------------------------------------
    # Mask management
    # ------------------------------------------------------------------
    def setMask(self, mask: np.ndarray) -> None:
        """Replace the structural mask.

        The mask is validated to be a binary array of the correct shape.  A
        float mask with intermediate values would silently turn "pruned" into
        "scaled down", which is exactly the failure mode the challenge warns
        about, so it is rejected.
        """
        maskArray = np.asarray(mask)
        if maskArray.shape != (self.inFeatures, self.outFeatures):
            raise ValueError(
                f"Mask shape {maskArray.shape} does not match layer shape "
                f"{(self.inFeatures, self.outFeatures)}."
            )
        unique = np.unique(maskArray)
        if not np.all(np.isin(unique, (0.0, 1.0))):
            raise ValueError(
                "The mask must be strictly binary (0.0 or 1.0); "
                f"got values {unique.tolist()}."
            )
        self.mask = maskArray.astype(np.float64, copy=True)

    def applyMaskToWeights(self) -> np.ndarray:
        """Return the effective weights ``rawWeights * mask`` as a NumPy array."""
        return self.rawWeights.data * self.mask

    def activeParameterCount(self) -> int:
        """Number of weight connections that are still alive."""
        return int(np.count_nonzero(self.mask))

    def totalParameterCount(self) -> int:
        """Number of weight connections in the dense layer."""
        return int(self.mask.size)

    def sparsity(self) -> float:
        """Fraction of weight connections that have been removed."""
        if self.mask.size == 0:
            return 0.0
        return 1.0 - (self.activeParameterCount() / self.mask.size)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------
    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the layer output for ``inputs`` of shape ``(batch, inFeatures)``."""
        if inputs.ndim != 2:
            raise ValueError(
                f"Linear layer expects a 2-D input (batch, inFeatures); got shape {inputs.shape}."
            )
        if inputs.shape[1] != self.inFeatures:
            raise ValueError(
                f"Linear layer expects {self.inFeatures} input features; got {inputs.shape[1]}."
            )

        effectiveWeights = ops.applyMask(self.rawWeights, self.mask)
        # Keep a handle so the backward pass can publish the effective gradient.
        self._effectiveWeightsNode = effectiveWeights

        outputs = ops.matmul(inputs, effectiveWeights)
        if self.bias is not None:
            outputs = ops.add(outputs, self.bias)
        return self.activation(outputs)

    def publishEffectiveGradient(self) -> None:
        """Copy the captured effective gradient onto the layer.

        Must be called after ``backward()``; the masking node stores
        ``dLoss/dW_effective`` on itself and this is the moment we read it.
        """
        node = getattr(self, "_effectiveWeightsNode", None)
        if node is None:
            return
        if node.effectiveGrad is not None:
            self.effectiveWeightGrad = node.effectiveGrad
        if self.rawWeights.grad is not None:
            self.rawWeightGrad = self.rawWeights.grad

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"Linear(in={self.inFeatures}, out={self.outFeatures}, "
            f"bias={self.useBias}, activation={self.activation.name})"
        )


class DenseNetwork:
    """A stack of :class:`Linear` layers with a single terminal activation.

    The terminal activation is ``identity`` by default so that the network
    can be used directly with :func:`aqua.engine.ops.crossEntropyFromLogits`,
    which expects raw logits.
    """

    def __init__(
        self,
        layers: list[Linear],
        name: Optional[str] = None,
    ) -> None:
        if not layers:
            raise ValueError("A DenseNetwork needs at least one layer.")
        self.layers = list(layers)
        self.name = name or "denseNetwork"

    def parameters(self) -> list[Parameter]:
        """All parameters of the network, in layer order."""
        params: list[Parameter] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def prunableLayers(self) -> list[Linear]:
        """Layers whose weights are eligible for pruning."""
        return list(self.layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the forward pass through every layer."""
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def publishEffectiveGradients(self) -> None:
        """Publish the captured effective weight gradients on every layer."""
        for layer in self.layers:
            layer.publishEffectiveGradient()

    def clearEffectiveGradients(self) -> None:
        """Drop the captured effective weight gradients."""
        for layer in self.layers:
            layer.effectiveWeightGrad = None
            layer.rawWeightGrad = None

    def activeParameterCount(self) -> int:
        """Total number of alive weight connections."""
        return int(sum(layer.activeParameterCount() for layer in self.layers))

    def totalParameterCount(self) -> int:
        """Total number of weight connections in the dense network."""
        return int(sum(layer.totalParameterCount() for layer in self.layers))

    def sparsity(self) -> float:
        """Fraction of weight connections removed across the whole network."""
        total = self.totalParameterCount()
        if total == 0:
            return 0.0
        return 1.0 - (self.activeParameterCount() / total)

    def parameterCount(self) -> int:
        """Total number of scalar parameters including biases."""
        return int(sum(parameter.size for parameter in self.parameters()))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"DenseNetwork({self.layers})"
