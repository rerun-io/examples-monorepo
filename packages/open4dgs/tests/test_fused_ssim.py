"""TRiGS's reconstruction loss takes fused-ssim (recipes/fused-ssim-src) when it is installed; its SSIM term matches the
torchmetrics path the loss falls back to, value and gradient (both use the 11x11 sigma-1.5 window over the "valid" map)."""

import pytest
import torch

losses = pytest.importorskip("trigs.core.losses", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")

pytestmark = pytest.mark.integration


def _loss_and_grad(backend: str, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    saved = losses._SSIM_BACKEND
    losses._SSIM_BACKEND = backend
    try:
        x = pred.clone().requires_grad_(True)
        loss = losses.reconstruction_loss(x, target, ssim_lambda=1.0)  # the SSIM term only
        loss.backward()
    finally:
        losses._SSIM_BACKEND = saved
    assert x.grad is not None
    return loss.detach(), x.grad


def test_fused_ssim_is_the_backend_and_matches_torchmetrics() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (fused-ssim kernels)")
    pytest.importorskip("fused_ssim", reason="fused-ssim comes from recipes/fused-ssim-src (open4dgs envs)")
    losses._SSIM_BACKEND = None
    assert losses._get_ssim_backend() == "fused"
    gen = torch.Generator(device="cuda").manual_seed(0)
    # A smooth target, the prediction a noisy, slightly shifted copy: an (H, W, 3) render / ground-truth pair as the trainer passes it.
    base = torch.nn.functional.interpolate(torch.rand((1, 3, 24, 40), generator=gen, device="cuda"), size=(96, 160), mode="bilinear")
    target = base[0].permute(1, 2, 0).contiguous()
    pred = (target.roll(1, dims=1) + 0.05 * torch.randn(target.shape, generator=gen, device="cuda")).clamp(0, 1)
    fused, g_fused = _loss_and_grad("fused", pred, target)
    reference, g_ref = _loss_and_grad("torchmetrics", pred, target)
    torch.testing.assert_close(fused, reference, rtol=0.0, atol=1e-5)
    cos = torch.nn.functional.cosine_similarity(g_fused.flatten().double(), g_ref.flatten().double(), dim=0)
    assert cos > 0.9999
    assert ((g_fused - g_ref).norm() / g_ref.norm()) < 1e-3
