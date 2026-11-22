"""TRiGS's reconstruction loss takes fused-ssim (recipes/fused-ssim-src) when it is installed, as upstream intends.

Its SSIM term is fused_ssim(..., padding="valid"): the mean of the 11x11 sigma-1.5 SSIM map without its 5-pixel border.
That is NOT the torchmetrics fallback's number: torchmetrics 1.9 averages the full map of a reflect-padded image. The test
pins the fused term (value and gradient) to torchmetrics' own map with the same border dropped, which only reads interior
windows, so the two agree up to float rounding.
"""

import pytest
import torch

losses = pytest.importorskip("trigs.core.losses", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")

pytestmark = pytest.mark.integration


def test_fused_ssim_is_the_backend_and_equals_the_valid_map_mean() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (fused-ssim kernels)")
    pytest.importorskip("fused_ssim", reason="fused-ssim comes from recipes/fused-ssim-src (open4dgs envs)")
    from torchmetrics.functional import structural_similarity_index_measure

    losses._SSIM_BACKEND = None
    assert losses._get_ssim_backend() == "fused"
    gen = torch.Generator(device="cuda").manual_seed(0)
    # A smooth target and a noisy, shifted prediction: an (H, W, 3) render / ground-truth pair as the trainer passes it.
    base = torch.nn.functional.interpolate(torch.rand((1, 3, 24, 40), generator=gen, device="cuda"), size=(96, 160), mode="bilinear")
    target = base[0].permute(1, 2, 0).contiguous()
    pred = (target.roll(1, dims=1) + 0.05 * torch.randn(target.shape, generator=gen, device="cuda")).clamp(0, 1)

    x = pred.clone().requires_grad_(True)
    fused = losses.reconstruction_loss(x, target, ssim_lambda=1.0)  # the SSIM term only: 1 - fused_ssim(valid)
    fused.backward()
    y = pred.clone().requires_grad_(True)
    _, ssim_map = structural_similarity_index_measure(y.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], data_range=1.0, return_full_image=True)
    reference = 1.0 - ssim_map[..., 5:-5, 5:-5].mean()
    reference.backward()
    assert x.grad is not None and y.grad is not None
    torch.testing.assert_close(fused.detach(), reference.detach(), rtol=0.0, atol=2e-5)
    g, h = x.grad.flatten().double(), y.grad.flatten().double()
    assert torch.nn.functional.cosine_similarity(g, h, dim=0) > 0.9999
    assert (g - h).norm() / h.norm() < 1e-2
