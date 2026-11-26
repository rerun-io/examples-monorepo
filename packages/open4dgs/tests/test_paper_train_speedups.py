"""Training-side techniques from PennyGS (arXiv 2611.27525) and HiGS (arXiv 2611.31918) on Open4dgsTrainer's seams:
strided pixel sampling (PennyGS 3.4), the fused loss (3.5), bf16 Adam moments (HiGS 4.3), plus recipe patch 0009.

Strided sampling renders one pixel of every 2x2 block through a camera whose intrinsics put the half-resolution pixel
centres on full-resolution pixel centres, with gsplat's 2D dilation scaled by 1/4: the render must equal the
full-resolution render at those pixels.
"""

import pytest
import torch

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")


def _scene(dev: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    n = 2000
    return {
        "means": torch.randn(n, 3, device=dev) * 0.3,
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1),
        "scales": torch.rand(n, 3, device=dev) * 0.03 + 0.005,
        "opacities": torch.rand(n, device=dev) * 0.9 + 0.05,
        "colors": torch.rand(n, 3, device=dev),
    }


@pytest.mark.integration
@pytest.mark.parametrize("offset", [(0, 0), (1, 1), (0, 1), (1, 0)])
@pytest.mark.parametrize("size", [(96, 128), (95, 127)])
def test_strided_camera_render_equals_full_render_subsampled(offset: tuple[int, int], size: tuple[int, int]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (gsplat rasterization)")
    from gsplat import rasterization

    from open4dgs.strided import STRIDE
    from open4dgs.trigs_adapter import camera_config
    from open4dgs.trigs_trainer import strided_camera_config

    dev = "cuda"
    h, w = size
    world_T_cam = torch.eye(4)[None].repeat(2, 1, 1)
    world_T_cam[:, 2, 3] = -2.0
    world_T_cam[1, 0, 3] = 0.1
    K = torch.tensor([[110.0, 0.0, 63.2], [0.0, 105.0, 47.7], [0.0, 0.0, 1.0]]).expand(2, 3, 3).contiguous()
    full = camera_config(world_T_cam, K, [(w, h), (w, h)], torch.device(dev))
    strided = strided_camera_config(full, offset)
    s = _scene(dev)
    oy, ox = offset
    ref, ref_a, _ = rasterization(**s, viewmats=full.viewmats, Ks=full.Ks, width=w, height=h, packed=False)
    out, out_a, _ = rasterization(
        **s,
        viewmats=strided.viewmats,
        Ks=strided.Ks,
        width=strided.width_for(0),
        height=strided.height_for(0),
        packed=False,
        eps2d=0.3 / STRIDE**2,
    )
    assert out.shape[1:3] == ref[:, oy::STRIDE, ox::STRIDE].shape[1:3]
    assert (strided.width_for(0), strided.height_for(0)) == (len(range(ox, w, STRIDE)), len(range(oy, h, STRIDE)))
    torch.testing.assert_close(out, ref[:, oy::STRIDE, ox::STRIDE], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(out_a, ref_a[:, oy::STRIDE, ox::STRIDE], atol=2e-5, rtol=1e-4)


def test_strided_frames_index_follows_the_active_offset() -> None:
    from open4dgs.strided import StrideState
    from open4dgs.trigs_trainer import strided_frames

    state = StrideState()
    frames = {f: [torch.arange(f * 100, f * 100 + 30).reshape(5, 6), torch.zeros(5, 6, 3)] for f in range(3)}
    view = strided_frames(frames, state)
    assert torch.equal(view[1][0], frames[1][0])
    state.offset = (1, 0)
    assert torch.equal(view[1][0], frames[1][0][1::2, 0::2])
    assert view[2][1].shape == (2, 3, 3)
    assert list(view) == [0, 1, 2]
    assert len(view[0]) == 2
    state.offset = None
    assert view[0][1].shape == (5, 6, 3)


@pytest.mark.integration
def test_adam_bf16_tracks_torch_adam() -> None:
    """Same updates as torch.optim.Adam up to bf16 moment rounding: 200 steps of a noisy quadratic end close."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (Triton)")
    from open4dgs.adam_bf16 import AdamBf16

    torch.manual_seed(0)
    target = torch.randn(10_001, device="cuda")
    ref = torch.zeros_like(target, requires_grad=True)
    ours = torch.zeros_like(target, requires_grad=True)
    betas, eps = (0.9**8, 0.999**8), 1e-15 / 8**0.5  # TRiGS's batch-8 rule
    opt_ref = torch.optim.Adam([ref], lr=1e-2, betas=betas, eps=eps)
    opt_ours = AdamBf16([ours], lr=1e-2, betas=betas, eps=eps)
    for _ in range(200):
        noise = 0.1 * torch.randn_like(target)
        for p, opt in ((ref, opt_ref), (ours, opt_ours)):
            p.grad = (p.detach() - target) + noise
            opt.step()
    assert opt_ours.state[ours]["exp_avg"].dtype == torch.bfloat16
    err_ref = (ref.detach() - target).abs().mean()
    assert (ours.detach() - ref.detach()).abs().mean() < 0.05 * err_ref + 1e-3
    assert (ours.detach() - target).abs().mean() < 1.1 * err_ref


@pytest.mark.integration
def test_bf16_stochastic_rounding_is_unbiased() -> None:
    """Rounding one value 20k times: the mean of the bf16 results equals the float32 value (nearest rounding would not)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (Triton)")
    from open4dgs.adam_bf16 import AdamBf16

    x = torch.full((20_000,), 1.0 + 3 * 2**-10, device="cuda")  # between two bf16 neighbours (spacing 2**-7)
    p = torch.zeros_like(x, requires_grad=True)
    opt = AdamBf16([p], lr=0.0, betas=(0.0, 0.0), eps=1.0)
    p.grad = x.clone()
    opt.step()  # beta1 = 0: exp_avg = grad, stored rounded
    m = opt.state[p]["exp_avg"].float()
    assert set(m.unique().tolist()) == {1.0, 1.0078125}
    assert abs(float(m.mean()) - float(x[0])) < 2e-4


@pytest.mark.integration
@pytest.mark.parametrize("bg", [(0.0, 0.0, 0.0), (0.2, 0.5, 1.0)])
def test_fused_loss_equals_trigs_photometric_losses(bg: tuple[float, float, float]) -> None:
    """The compiled masked-composite loss returns TRiGS's three losses and the same render/alpha gradients."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from trigs.core.trainer import TrigsTrainingConfig

    from open4dgs.trigs_trainer import Open4dgsTrainer

    torch.manual_seed(0)
    h, w = 48, 80
    config = TrigsTrainingConfig(masked_rgb_loss=True, masked_rgb_composite=True, masked_rgb_bg_color=bg, ssim_lambda=0.2)
    colors = torch.rand(1, h, w, 3, device="cuda") * 1.2 - 0.1
    alphas = torch.rand(1, h, w, 1, device="cuda")
    gt = torch.rand(h, w, 3, device="cuda")
    masks = {0: [torch.rand(h, w, device="cuda").round()]}
    outs, grads = [], []
    for fused in (False, True):
        trainer = Open4dgsTrainer(device="cuda", fused_loss=fused)
        c, a = colors.clone().requires_grad_(True), alphas.clone().requires_grad_(True)
        losses = trainer._photometric_losses(torch.nn.ParameterDict(), c, a, gt, masks, 0, 0, config)
        sum(losses).backward()
        outs.append(torch.stack([x.detach() for x in losses]))
        grads.append((c.grad, a.grad))
    torch.testing.assert_close(outs[1], outs[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grads[1][0], grads[0][0], rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(grads[1][1], grads[0][1], rtol=1e-4, atol=1e-8)


def test_rigid_coherence_loss_value_and_gradients_unchanged() -> None:
    """Recipe patch 0009 (index_select gathers) against upstream's advanced-indexing form, on repeated neighbour indices."""
    from trigs.core.losses import rigid_coherence_loss

    torch.manual_seed(0)
    n, k, lambda_c = 500, 8, 10.0
    zeta, color = torch.randn(n, 6, dtype=torch.float64), torch.rand(n, 3, dtype=torch.float64)
    nn_idx = torch.randint(0, n, (n, k))

    def upstream(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        weight = torch.exp(-lambda_c * ((c.unsqueeze(1) - c[nn_idx]) ** 2).sum(-1))
        return (weight * ((z.unsqueeze(1) - z[nn_idx]) ** 2).sum(-1)).mean()

    z1, c1 = zeta.clone().requires_grad_(True), color.clone().requires_grad_(True)
    z2, c2 = zeta.clone().requires_grad_(True), color.clone().requires_grad_(True)
    ours = rigid_coherence_loss(z1, c1, nn_idx, lambda_c, detach_color_weight=False)
    ours.backward()
    ref = upstream(z2, c2)
    ref.backward()
    torch.testing.assert_close(ours, ref)
    torch.testing.assert_close(z1.grad, z2.grad)
    torch.testing.assert_close(c1.grad, c2.grad)
