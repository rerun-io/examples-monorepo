"""Correlation and patch extraction on the kernels :mod:`dpvo.kernels` serves for the input tensors' device.

- :func:`corr` -- correlation volume lookup between two feature maps.
- :func:`patchify` -- patch extraction at sub-pixel coordinates with an
  optional bilinear interpolation mode.

Forward passes only: the kernels have no backward passes (training is not supported).
"""

from jaxtyping import Float, Int
from torch import Tensor

from dpvo.kernels import corr_kernels


def patchify(net: Float[Tensor, "..."], coords: Float[Tensor, "..."], radius: int, mode: str = 'bilinear') -> Float[Tensor, "..."]:
    """Extract feature patches at sub-pixel coordinates.

    The kernel extracts ``(2r + 2)²`` integer-aligned patches; ``"bilinear"``
    then interpolates them with the fractional part of the coordinates to
    ``(2r + 1)²``.

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
    patches: Float[Tensor, "..."]
    patches, = corr_kernels(net.device).patchify_forward(net, coords, radius)

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
    fmap1: Float[Tensor, "1 mem channels h4 w4"],
    fmap2: Float[Tensor, "1 mem channels h8 w8"],
    coords: Float[Tensor, "1 n_edges 2 ps ps"],
    ii: Int[Tensor, "n_edges"],
    jj: Int[Tensor, "n_edges"],
    radius: int = 1,
) -> Float[Tensor, "1 n_edges neighborhood neighborhood ps ps"]:
    """Dot-product correlation of ``fmap1[ii]`` against ``fmap2[jj]`` in a ``(2r + 1)²`` window around ``coords``.

    Args:
        fmap1: Feature map indexed by ``ii`` (source frames).
        fmap2: Feature map indexed by ``jj`` (target frames).
        coords: 2-D lookup coordinates for each edge and patch pixel.
        ii: Source frame indices for each edge.
        jj: Target frame indices for each edge.
        radius: Search radius for the correlation neighbourhood.

    Returns:
        Correlation volume of shape ``(1, n_edges, 2r + 1, 2r + 1, ps, ps)``.
    """
    volume: Tensor
    volume, = corr_kernels(fmap1.device).forward(fmap1, fmap2, coords, ii, jj, radius)
    return volume
