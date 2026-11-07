"""Correlation and patchify on the Mojo kernels: the `CorrKernels` protocol of `dpvo.kernels`.

Same signatures, shapes and dtypes as the CUDA extension `dpvo._cuda_corr` (float16 or float32
features, float32 coordinates, int64 indices). `forward` runs the integer-offset kernel and then the
extension's bilinear interpolation, as the same PyTorch ops in the same order. The backward passes
are training-only and not ported.
"""

import torch
from jaxtyping import Float, Float32, Int64
from torch import Tensor

from dpvo.mojo_backend._launch import launch

_FEATURE_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.float32)


def _check_features(op: str, *maps: Tensor) -> None:
    if maps[0].dtype not in _FEATURE_DTYPES or any(m.dtype != maps[0].dtype for m in maps):
        raise TypeError(f"dpvo.mojo_backend.{op}: features must share dtype float16 or float32, got {[m.dtype for m in maps]}")


def forward(
    fmap1: Float[Tensor, "b n1 c p p"],
    fmap2: Float[Tensor, "b n2 c h w"],
    coords: Float32[Tensor, "b e 2 p p"],
    ii: Int64[Tensor, " e"],
    jj: Int64[Tensor, " e"],
    radius: int,
) -> tuple[Float[Tensor, "b e d1 d1 p p"]]:
    """Correlation of patch features `fmap1[ii]` with `fmap2[jj]` around `coords`, (2r + 1)^2 offsets per patch pixel."""
    _check_features("corr", fmap1, fmap2)
    batch, num_edges, _, grid_h, grid_w = coords.shape
    d: int = 2 * radius + 2
    corr: Tensor = fmap1.new_empty(batch, num_edges, d, d, grid_h, grid_w)
    if corr.numel() > 0:
        launch("corr", (fmap1.dtype == torch.float16, radius), (fmap1.contiguous(), fmap2.contiguous(), coords.contiguous(), ii.contiguous(), jj.contiguous(), corr))
    x: Tensor = coords[:, :, 0, None, None]
    y: Tensor = coords[:, :, 1, None, None]
    dx: Tensor = (x - x.floor()).to(fmap1.dtype)
    dy: Tensor = (y - y.floor()).to(fmap2.dtype)
    out: Tensor = (1 - dx) * (1 - dy) * corr[:, :, 0 : d - 1, 0 : d - 1]
    out += dx * (1 - dy) * corr[:, :, 0 : d - 1, 1:d]
    out += (1 - dx) * dy * corr[:, :, 1:d, 0 : d - 1]
    out += dx * dy * corr[:, :, 1:d, 1:d]
    return (out.permute(0, 1, 3, 2, 4, 5),)


def backward(fmap1: Tensor, fmap2: Tensor, coords: Tensor, ii: Tensor, jj: Tensor, grad: Tensor, radius: int) -> tuple[Tensor, Tensor]:
    """Training only: not ported to Mojo."""
    raise NotImplementedError("dpvo.mojo_backend: the correlation backward pass (training) is CUDA-only")


def patchify_forward(net: Float[Tensor, "b c h w"], coords: Float32[Tensor, "b m 2"], radius: int) -> tuple[Float[Tensor, "b m c d d"]]:
    """(2r + 2)^2 windows of `net` at `coords` (x, y), zero outside the map."""
    _check_features("patchify", net)
    batch, num_patches = coords.shape[:2]
    d: int = 2 * radius + 2
    patches: Tensor = torch.zeros(batch, num_patches, net.shape[1], d, d, dtype=net.dtype, device=net.device)
    if patches.numel() > 0:
        launch("patchify", (net.dtype == torch.float16, radius), (net.contiguous(), coords.contiguous(), patches))
    return (patches,)


def patchify_backward(net: Tensor, coords: Tensor, grad: Tensor, radius: int) -> tuple[Tensor]:
    """Training only: not ported to Mojo."""
    raise NotImplementedError("dpvo.mojo_backend: the patchify backward pass (training) is CUDA-only")
