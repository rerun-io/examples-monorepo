"""Python wrappers for CUDA-accelerated correlation and patchification kernels.

This module provides custom :class:`torch.autograd.Function` subclasses that
delegate their forward and backward passes to the ``_cuda_corr`` extension.
Two high-level convenience functions are exposed:

- :func:`corr` -- correlation volume lookup between two feature maps.
- :func:`patchify` -- patch extraction at sub-pixel coordinates with an
  optional bilinear interpolation mode.
"""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from dpvo import _cuda_corr  # pyrefly: ignore[missing-module-attribute]


class CorrLayer(torch.autograd.Function):
    """Custom autograd function for computing local correlation volumes.

    Given two multi-frame feature maps and a set of 2-D coordinate grids,
    the forward pass computes dot-product correlation within a local
    ``(2r + 1)²`` neighbourhood for each edge ``(ii[e], jj[e])``.

    The backward pass distributes gradients back to both feature maps,
    optionally applying random edge dropout to reduce memory/compute.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        fmap1: Float[Tensor, "1 mem channels h4 w4"],
        fmap2: Float[Tensor, "1 mem channels h8 w8"],
        coords: Float[Tensor, "1 n_edges 2 ps ps"],
        ii: Int[Tensor, "n_edges"],
        jj: Int[Tensor, "n_edges"],
        radius: int,
        dropout: float,
    ) -> Float[Tensor, "1 n_edges neighborhood neighborhood ps ps"]:
        """Compute forward correlation between feature maps at given coordinates.

        Args:
            ctx: Autograd context for saving tensors needed in backward.
            fmap1: Feature map at 1/4 resolution indexed by ``ii``.
            fmap2: Feature map at 1/8 resolution indexed by ``jj``.
            coords: 2-D lookup coordinates per edge and patch pixel.
            ii: Source frame indices for each edge.
            jj: Target frame indices for each edge.
            radius: Search radius defining the ``(2r + 1)²`` neighbourhood.
            dropout: Fraction of edges to keep during the backward pass
                (1.0 = keep all, <1.0 = randomly drop edges).

        Returns:
            Correlation volume of shape ``(1, n_edges, 2R + 1, 2R + 1, ps, ps)``.
        """
        ctx.save_for_backward(fmap1, fmap2, coords, ii, jj)
        ctx.radius = radius  # pyrefly: ignore[missing-attribute]
        ctx.dropout = dropout  # pyrefly: ignore[missing-attribute]
        corr, = _cuda_corr.forward(fmap1, fmap2, coords, ii, jj, radius)

        return corr

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: torch.autograd.function.FunctionCtx,
        grad: Float[Tensor, "..."],
    ) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."], None, None, None, None, None]:
        """Backward pass for correlation -- distributes gradients to feature maps.

        When ``dropout < 1.0``, a random subset of edges is kept so that
        gradient computation is cheaper at the cost of higher variance.

        Args:
            ctx: Autograd context containing saved tensors and attributes.
            grad: Upstream gradient w.r.t. the correlation output.

        Returns:
            Gradients for ``(fmap1, fmap2)`` and ``None`` for non-differentiable
            inputs (coords, ii, jj, radius, dropout).
        """
        fmap1: Float[Tensor, "..."]
        fmap2: Float[Tensor, "..."]
        coords: Float[Tensor, "..."]
        ii: Int[Tensor, "..."]
        jj: Int[Tensor, "..."]
        fmap1, fmap2, coords, ii, jj = ctx.saved_tensors  # pyrefly: ignore[missing-attribute]

        # Optionally drop a random subset of edges to save compute
        if ctx.dropout < 1:  # pyrefly: ignore[missing-attribute]
            perm: Bool[Tensor, "n_edges"] = torch.rand(len(ii), device="cuda") < ctx.dropout  # pyrefly: ignore[missing-attribute]
            coords = coords[:,perm]
            grad = grad[:,perm]
            ii = ii[perm]
            jj = jj[perm]

        fmap1_grad: Float[Tensor, "..."]
        fmap2_grad: Float[Tensor, "..."]
        fmap1_grad, fmap2_grad = \
            _cuda_corr.backward(fmap1, fmap2, coords, ii, jj, grad, ctx.radius)  # pyrefly: ignore[missing-attribute]

        return fmap1_grad, fmap2_grad, None, None, None, None, None


class PatchLayer(torch.autograd.Function):
    """Custom autograd function for extracting feature patches at coordinates.

    Given a feature tensor and a set of 2-D coordinates, the forward pass
    extracts ``(2r + 2)²`` patches centred on each coordinate (the extra
    +1 in each dimension supports bilinear interpolation in
    :func:`patchify`).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        net: Float[Tensor, "..."],
        coords: Float[Tensor, "..."],
        radius: int,
    ) -> Float[Tensor, "..."]:
        """Extract patches from ``net`` at the given coordinates.

        Args:
            ctx: Autograd context for saving tensors needed in backward.
            net: Feature tensor to extract patches from.
            coords: 2-D coordinates (y, x) for each patch centre.
            radius: Half-size of the extraction window.

        Returns:
            Extracted patches tensor.
        """
        ctx.radius = radius  # pyrefly: ignore[missing-attribute]
        ctx.save_for_backward(net, coords)

        patches: Float[Tensor, "..."]
        patches, = _cuda_corr.patchify_forward(net, coords, radius)
        return patches

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: torch.autograd.function.FunctionCtx,
        grad: Float[Tensor, "..."],
    ) -> tuple[Float[Tensor, "..."], None, None]:
        """Backward pass for patchification.

        Args:
            ctx: Autograd context containing the saved feature tensor and
                coordinates.
            grad: Upstream gradient w.r.t. the extracted patches.

        Returns:
            Gradient for ``net`` and ``None`` for the non-differentiable
            inputs (coords, radius).
        """
        net: Float[Tensor, "..."]
        coords: Float[Tensor, "..."]
        net, coords = ctx.saved_tensors  # pyrefly: ignore[missing-attribute]
        grad, = _cuda_corr.patchify_backward(net, coords, grad, ctx.radius)  # pyrefly: ignore[missing-attribute]

        return grad, None, None

def patchify(net: Float[Tensor, "..."], coords: Float[Tensor, "..."], radius: int, mode: str = 'bilinear') -> Float[Tensor, "..."]:
    """Extract feature patches at sub-pixel coordinates.

    Delegates to :class:`PatchLayer` for the raw extraction and then
    optionally applies bilinear interpolation using the fractional part of
    the coordinates to produce a ``(2r + 1)²`` patch from the
    ``(2r + 2)²`` raw output.

    Args:
        net: Feature tensor to extract patches from.
        coords: 2-D floating-point coordinates for each patch centre.
        radius: Half-size of the output patch window.
        mode: Interpolation mode. ``"bilinear"`` performs 4-corner weighted
            interpolation; any other value returns the raw (integer-aligned)
            patches.

    Returns:
        Interpolated (or raw) feature patches.
    """
    patches: Float[Tensor, "..."] = PatchLayer.apply(net, coords, radius)  # pyrefly: ignore[bad-assignment]

    if mode == 'bilinear':
        # Compute sub-pixel offsets and perform bilinear interpolation
        # across the four integer-offset corners.
        offset: Float[Tensor, "..."] = (coords - coords.floor()).to(net.device)
        dx: Float[Tensor, "..."]
        dy: Float[Tensor, "..."]
        dx, dy = offset[:,:,None,None,None].unbind(dim=-1)

        d: int = 2 * radius + 1
        x00: Float[Tensor, "..."] = (1-dy) * (1-dx) * patches[...,:d,:d]
        x01: Float[Tensor, "..."] = (1-dy) * (  dx) * patches[...,:d,1:]
        x10: Float[Tensor, "..."] = (  dy) * (1-dx) * patches[...,1:,:d]
        x11: Float[Tensor, "..."] = (  dy) * (  dx) * patches[...,1:,1:]

        return x00 + x01 + x10 + x11

    return patches


def corr(
    fmap1: Float[Tensor, "..."],
    fmap2: Float[Tensor, "..."],
    coords: Float[Tensor, "..."],
    ii: Int[Tensor, "..."],
    jj: Int[Tensor, "..."],
    radius: int = 1,
    dropout: float = 1.0,
) -> Float[Tensor, "..."]:
    """Compute local correlation volumes between two feature maps.

    Convenience wrapper around :meth:`CorrLayer.apply`.

    Args:
        fmap1: Feature map indexed by ``ii`` (source frames).
        fmap2: Feature map indexed by ``jj`` (target frames).
        coords: 2-D lookup coordinates for each edge and patch pixel.
        ii: Source frame indices for each edge.
        jj: Target frame indices for each edge.
        radius: Search radius for the correlation neighbourhood.
        dropout: Edge keep-probability during backward (1.0 = no dropout).

    Returns:
        Correlation volume tensor.
    """
    return CorrLayer.apply(fmap1, fmap2, coords, ii, jj, radius, dropout)  # pyrefly: ignore[bad-return]
