"""`vipe_ext.droid_net_ext` in Mojo: the two correlation lookups MetricTrail's DROID network calls.

`corr_index_forward` and `altcorr_index_forward` have vipe_ext's signatures and return one-element
lists, as the pybind functions do. Every other droid_net_ext function (the backward passes,
`altcorr_forward`) raises NotImplementedError when called.
"""

import os

import torch
from jaxtyping import Float, Float32, Int64
from torch import Tensor

from vipe_mojo._launch import launch
from vipe_mojo._unported import module_getattr

__getattr__ = module_getattr("droid_net_ext")

_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.float32)


def corr_index_forward(
    volume: Float[Tensor, "b h1 w1 h2 w2"], coords: Float32[Tensor, "b 2 h1 w1"], radius: int
) -> list[Float[Tensor, "b rd rd h1 w1"]]:
    """Bilinear lookups in the all-pairs correlation volume, (2r+1) x (2r+1) per query pixel (`corr_index_cuda_forward`)."""
    if volume.dtype not in _DTYPES:
        raise TypeError(f"corr_index_forward: volume must be float16 or float32, got {volume.dtype}")
    volume = volume.contiguous()
    coords = coords.contiguous()
    rd: int = 2 * radius + 1
    corr: Tensor = torch.zeros(volume.shape[0], rd, rd, volume.shape[1], volume.shape[2], dtype=volume.dtype, device=volume.device)
    if corr.numel():
        launch("corr_index", (volume.dtype == torch.float16, radius), (volume, coords, corr))
    return [corr]


def altcorr_index_forward(
    fmap1: Float[Tensor, "b n1 h w c"],
    fmap2_pyramid: list[Float[Tensor, "b n2 _ _ c"]],
    coords: Float32[Tensor, "b m h w 2"],
    ii: Int64[Tensor, " m"],
    jj: Int64[Tensor, " m"],
    radius: int,
) -> list[Float32[Tensor, "b m l_rd_rd h w"]]:
    """Correlation of edge features at (2r+1)^2 offsets around coords, every pyramid level (`altcorr_index_cuda_forward`).

    As in vipe_ext, half features with C % 8 == 0 and C <= 128 take the per-pixel fast path unless
    VIPE_ALTCORR_PIX=0; other inputs take the staged path (float sums per 32-channel chunk).
    """
    if fmap1.dtype not in _DTYPES or any(f.dtype != fmap1.dtype for f in fmap2_pyramid):
        raise TypeError("altcorr_index_forward: fmap1 and the pyramid must share one dtype, float16 or float32")
    fmap1 = fmap1.contiguous()
    coords = coords.contiguous()
    ii = ii.contiguous()
    jj = jj.contiguous()
    b, m, h, w = coords.shape[:4]
    rd: int = 2 * radius + 1
    corr: Tensor = torch.zeros(b, m, len(fmap2_pyramid) * rd * rd, h, w, dtype=torch.float32, device=coords.device)
    channels: int = fmap1.shape[4]
    half: bool = fmap1.dtype == torch.float16
    pix: bool = half and channels % 8 == 0 and channels <= 128 and os.environ.get("VIPE_ALTCORR_PIX", "1")[:1] != "0"
    if corr.numel():
        for level, fmap2 in enumerate(fmap2_pyramid):
            launch("altcorr_index", (half, pix, radius, level * rd * rd, float(1 << level)), (fmap1, fmap2.contiguous(), coords, ii, jj, corr))
    return [corr]
