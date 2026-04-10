"""Neural network building blocks for the DPVO recurrent update operator.

This module contains:

- **Normalization layers**: :class:`LayerNorm1D` (channel-last layer norm for
  1-D feature maps).
- **Residual blocks**: :class:`GatedResidual` -- a gated residual connection
  used in the GRU update (see Sec. 3.2 of Teed et al. 2022).
- **Neighbor aggregation**: :class:`SoftAgg` and :class:`SoftAggBasic` --
  attention-weighted scatter-based message passing that pools information
  across edges sharing the same patch index (spatial neighbours) or the same
  (frame_i, frame_j) pair (temporal neighbours).
- **Gradient manipulation**: :class:`GradClip`, :class:`GradZero`,
  :class:`GradMag` -- custom autograd functions that modify the backward
  pass to improve training stability (clamp / zero / monitor gradient
  magnitudes).
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .scatter_utils import scatter_softmax, scatter_sum


class LayerNorm1D(nn.Module):
    """Layer normalization applied along the last (feature) dimension.

    Standard :class:`nn.LayerNorm` expects the normalized dimension to be
    last, but 1-D feature maps often have shape ``(B, C, L)``.  This
    wrapper transposes to ``(B, L, C)``, applies layer norm, then
    transposes back.

    Attributes:
        norm: Underlying ``nn.LayerNorm`` instance.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(dim, eps=1e-4)

    def forward(self, x: Float[Tensor, "batch dim length"]) -> Float[Tensor, "batch dim length"]:
        """Apply layer norm along the channel dimension.

        Args:
            x: Input tensor of shape ``(batch, dim, length)``.

        Returns:
            Normalized tensor with the same shape.
        """
        return self.norm(x.transpose(1,2)).transpose(1,2)


class GatedResidual(nn.Module):
    """Gated residual block: ``x + gate(x) * res(x)``.

    The gate is a learned sigmoid function that controls how much of the
    residual branch is added.  This provides a smooth interpolation between
    identity and a two-layer MLP update, which stabilizes training of the
    recurrent update operator.  See Sec. 3.2 of Teed et al. (2022).

    Attributes:
        gate: ``Linear -> Sigmoid`` producing element-wise gate values in [0, 1].
        res: ``Linear -> ReLU -> Linear`` residual branch.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.gate: nn.Sequential = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid())

        self.res: nn.Sequential = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

    def forward(self, x: Float[Tensor, "*batch dim"]) -> Float[Tensor, "*batch dim"]:
        """Compute gated residual update.

        Args:
            x: Input features.

        Returns:
            ``x + gate(x) * res(x)`` with the same shape as input.
        """
        return x + self.gate(x) * self.res(x)


class SoftAgg(nn.Module):
    """Attention-weighted scatter aggregation for neighbor communication.

    Groups edges by their index ``ix`` (which may encode patch identity or
    frame-pair identity) and aggregates features within each group using a
    learned attention mechanism:

    1. Compute per-edge attention logits via ``g(x)``  (dim -> dim).
    2. Apply group-wise softmax to get attention weights ``w``.
    3. Weighted sum: ``y[group] = sum_i  w_i * f(x_i)``.
    4. Project aggregated features via ``h(y)`` and (if ``expand=True``)
       scatter back to per-edge resolution.

    This is used twice in the :class:`~mini_dpvo.net.Update` operator:
    once grouping by patch index ``kk`` (spatial aggregation -- patches
    observed in different frames share information) and once by the hash
    ``ii * 12345 + jj`` (temporal aggregation -- different patches
    connecting the same frame pair share information).

    See Sec. 3.2 of Teed et al. (2022) for details.

    Attributes:
        dim: Feature dimensionality.
        expand: If True, the aggregated group features are scattered back
            to per-edge resolution.  If False, only group-level features
            are returned.
        f: Value projection (Linear, dim -> dim).
        g: Key/attention projection (Linear, dim -> dim).
        h: Output projection applied after aggregation (Linear, dim -> dim).
    """

    def __init__(self, dim: int = 512, expand: bool = True) -> None:
        super().__init__()
        self.dim: int = dim
        self.expand: bool = expand
        self.f: nn.Linear = nn.Linear(self.dim, self.dim)
        self.g: nn.Linear = nn.Linear(self.dim, self.dim)
        self.h: nn.Linear = nn.Linear(self.dim, self.dim)

    def forward(self, x: Float[Tensor, "batch edges dim"], ix: Int[Tensor, "edges"]) -> Float[Tensor, "..."]:
        """Aggregate edge features by group index.

        Args:
            x: Per-edge features of shape ``(batch, n_edges, dim)``.
            ix: Group index for each edge.  Edges sharing the same index
                are aggregated together.

        Returns:
            If ``expand=True``: per-edge aggregated features of shape
            ``(batch, n_edges, dim)``.  Otherwise: per-group features
            of shape ``(batch, n_groups, dim)``.
        """
        _: Tensor
        jx: Tensor
        # Map raw indices to contiguous 0..n_groups-1 for scatter ops
        _, jx = torch.unique(ix, return_inverse=True)
        # Attention weights via group-wise softmax over key projections
        w: Float[Tensor, "..."] = scatter_softmax(self.g(x), jx, dim=1)
        # Weighted aggregation of value projections
        y: Float[Tensor, "..."] = scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            # Scatter group features back to per-edge resolution
            return self.h(y)[:,jx]

        return self.h(y)


class SoftAggBasic(nn.Module):
    """Simplified attention-weighted scatter aggregation with scalar attention.

    Identical to :class:`SoftAgg` except the attention logits are
    scalar-valued (``g: dim -> 1``) instead of vector-valued
    (``g: dim -> dim``).  This reduces parameters but provides less
    expressive per-dimension weighting.

    Attributes:
        dim: Feature dimensionality.
        expand: If True, scatter aggregated features back to per-edge shape.
        f: Value projection (Linear, dim -> dim).
        g: Scalar attention projection (Linear, dim -> 1).
        h: Output projection (Linear, dim -> dim).
    """

    def __init__(self, dim: int = 512, expand: bool = True) -> None:
        super().__init__()
        self.dim: int = dim
        self.expand: bool = expand
        self.f: nn.Linear = nn.Linear(self.dim, self.dim)
        self.g: nn.Linear = nn.Linear(self.dim,        1)
        self.h: nn.Linear = nn.Linear(self.dim, self.dim)

    def forward(self, x: Float[Tensor, "batch edges dim"], ix: Int[Tensor, "edges"]) -> Float[Tensor, "..."]:
        """Aggregate edge features by group index with scalar attention.

        Args:
            x: Per-edge features of shape ``(batch, n_edges, dim)``.
            ix: Group index for each edge.

        Returns:
            Aggregated features, per-edge if ``expand=True``, otherwise
            per-group.
        """
        _: Tensor
        jx: Tensor
        _, jx = torch.unique(ix, return_inverse=True)
        w: Float[Tensor, "..."] = scatter_softmax(self.g(x), jx, dim=1)
        y: Float[Tensor, "..."] = scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]

        return self.h(y)


### Gradient Clipping and Zeroing Operations ###
# These custom autograd functions modify gradients during the backward pass
# to improve training stability.  The forward pass is always an identity.

GRAD_CLIP: float = 0.1
"""Threshold for :class:`GradZero` -- gradients with magnitude above this
value are zeroed out entirely."""


class GradClip(torch.autograd.Function):
    """Custom autograd function that clamps gradients to [-0.01, 0.01].

    The forward pass is the identity function.  During backpropagation,
    NaN gradients are replaced with zeros and all values are hard-clamped.
    This prevents exploding gradients from the delta (pixel displacement)
    and weight (confidence) prediction heads.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return x

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        # Replace NaN gradients (can arise from degenerate projections)
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)


class GradientClip(nn.Module):
    """Module wrapper around :class:`GradClip` for use in ``nn.Sequential``.

    Inserted after the delta and weight prediction heads in the
    :class:`~mini_dpvo.net.Update` module to prevent large gradients
    from destabilizing training.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return GradClip.apply(x)


class GradZero(torch.autograd.Function):
    """Custom autograd function that zeros out large gradients entirely.

    Unlike :class:`GradClip` which clamps, this function sets any gradient
    component with absolute value exceeding :data:`GRAD_CLIP` to zero.
    This is a more aggressive strategy that completely suppresses outlier
    gradient contributions.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return x

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        grad_x = torch.where(torch.abs(grad_x) > GRAD_CLIP, torch.zeros_like(grad_x), grad_x)
        return grad_x


class GradientZero(nn.Module):
    """Module wrapper around :class:`GradZero` for use in ``nn.Sequential``."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return GradZero.apply(x)


class GradMag(torch.autograd.Function):
    """Debugging autograd function that prints the mean gradient magnitude.

    The forward pass is the identity.  During backpropagation the mean
    absolute gradient is printed to stdout.  Useful for diagnosing
    vanishing or exploding gradients at specific points in the network.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        return x

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_x: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
        print(grad_x.abs().mean())
        return grad_x
