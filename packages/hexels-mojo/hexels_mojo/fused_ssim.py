"""fused-ssim's API (rahul-goel/fused-ssim@1272e21 `fused_ssim/__init__.py`) on the Mojo kernels.

`fused_ssim(img1, img2)` is the mean SSIM (11x11 Gaussian, sigma 1.5, C1 = 0.01^2, C2 = 0.03^2) of two
(B, C, H, W) float32 CUDA images; it is differentiable in `img1` only, as upstream.
"""

from typing import Any, Literal

import torch
from jaxtyping import Float32
from torch import Tensor

from hexels_mojo._launch import launch

C1: float = 0.01**2
C2: float = 0.03**2


def _check(img1: Tensor, img2: Tensor) -> None:
    if img1.dim() != 4 or img1.shape != img2.shape:
        raise ValueError(f"fused_ssim needs two (B, C, H, W) images of one shape, got {tuple(img1.shape)} and {tuple(img2.shape)}")
    if img1.dtype != torch.float32 or img2.dtype != torch.float32:
        raise TypeError(f"fused_ssim needs float32 images, got {img1.dtype} and {img2.dtype}")


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, c1: float, c2: float, img1: Tensor, img2: Tensor, padding: str = "same", train: bool = True) -> Tensor:
        _check(img1, img2)
        img1, img2 = img1.contiguous(), img2.contiguous()
        ssim_map: Tensor = torch.empty_like(img1)
        derivatives: list[Tensor] = [torch.empty_like(img1) if train else ssim_map for _ in range(3)]
        launch("fused_ssim", (c1, c2, train), (img1, img2, ssim_map, *derivatives))
        ctx.save_for_backward(img1.detach(), img2, *derivatives)
        ctx.padding = padding
        return ssim_map[:, :, 5:-5, 5:-5] if padding == "valid" else ssim_map

    @staticmethod
    def backward(ctx: Any, opt_grad: Tensor) -> tuple[None, None, Tensor, None, None, None]:
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        dl_dmap: Tensor = opt_grad
        if ctx.padding == "valid":
            dl_dmap = torch.zeros_like(img1)
            dl_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad: Tensor = torch.empty_like(img1)
        launch("fused_ssim_backward", (), (img1, img2, dl_dmap.contiguous(), grad, dm_dmu1, dm_dsigma1_sq, dm_dsigma12))
        return None, None, grad, None, None, None


def fused_ssim(
    img1: Float32[Tensor, "b c h w"], img2: Float32[Tensor, "b c h w"], padding: Literal["same", "valid"] = "same", train: bool = True
) -> Float32[Tensor, ""]:
    """Mean SSIM of img1 against img2; `padding="valid"` drops the 5-pixel border of the map before the mean."""
    ssim_map: Tensor = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)  # pyrefly: ignore[bad-assignment]
    return ssim_map.mean()
