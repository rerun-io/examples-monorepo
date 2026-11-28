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
    import inspect

    from gsplat import rasterization

    from open4dgs.strided import GSPLAT_EPS2D, STRIDE
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
        eps2d=GSPLAT_EPS2D / STRIDE**2,
    )
    assert inspect.signature(rasterization).parameters["eps2d"].default == GSPLAT_EPS2D
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
def test_fused_step_equals_each_optimizer_step() -> None:
    """adam_bf16.fused_step (one launch for every parameter of several AdamBf16s) equals each optimizer's own step bit for
    bit (parameters, both moments, step counters), across lr changes, a parameter without a gradient and cleared state."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (Triton)")
    from open4dgs.adam_bf16 import AdamBf16, fused_step

    def make() -> tuple[list[torch.Tensor], list[AdamBf16]]:
        torch.manual_seed(0)
        shapes = [(3001, 3), (3001,), (3001, 15, 3), (7,), (5000, 4)]
        ps = [torch.randn(s, device="cuda").requires_grad_(True) for s in shapes]
        opts = [
            AdamBf16([ps[0], ps[1]], lr=1e-2, betas=(0.9, 0.999), eps=1e-15, seed=0),
            AdamBf16([{"params": [ps[2]], "lr": 3e-3, "betas": (0.8, 0.99), "eps": 1e-8}], seed=1),
            AdamBf16([ps[3], ps[4]], lr=1e-1, betas=(0.5, 0.9), eps=1e-6, seed=2),
        ]
        return ps, opts

    ref_p, ref_o = make()
    new_p, new_o = make()
    g = torch.Generator(device="cuda").manual_seed(1)
    for step in range(12):
        grads = [torch.randn(p.shape, device="cuda", generator=g) for p in ref_p]
        for i, (a, b) in enumerate(zip(ref_p, new_p, strict=True)):
            skip = i == 3 and step % 3 == 1
            a.grad = None if skip else grads[i].clone()
            b.grad = None if skip else grads[i].clone()
        if step == 5:
            for opts in (ref_o, new_o):
                opts[1].param_groups[0]["lr"] = 1e-3
                opts[2].state[opts[2].param_groups[0]["params"][1]].clear()
        for o in ref_o:
            o.step()
            o.zero_grad(set_to_none=True)
        fused_step(new_o)
        assert all(p.grad is None for p in new_p)
    for opt_ref, opt_new in zip(ref_o, new_o, strict=True):
        params_ref = [p for grp in opt_ref.param_groups for p in grp["params"]]
        params_new = [p for grp in opt_new.param_groups for p in grp["params"]]
        for a, b in zip(params_ref, params_new, strict=True):
            assert torch.equal(a.detach(), b.detach())
            sa, sb = opt_ref.state[a], opt_new.state[b]
            assert float(sa["step"]) == float(sb["step"])
            assert torch.equal(sa["exp_avg"], sb["exp_avg"]) and torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"])


@pytest.mark.integration
def test_fused_step_steps_a_shared_parameter_once() -> None:
    """A parameter in two AdamBf16s: fused_step steps it as the optimizers' own steps do (the first clears its gradient,
    so the second skips it), not twice in one launch."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (Triton)")
    from open4dgs.adam_bf16 import AdamBf16, fused_step

    def make() -> tuple[list[torch.Tensor], list[AdamBf16]]:
        torch.manual_seed(0)
        shared, own = torch.randn(1000, device="cuda").requires_grad_(True), torch.randn(10, device="cuda").requires_grad_(True)
        return [shared, own], [AdamBf16([shared], seed=0), AdamBf16([shared, own], seed=1)]

    (ref_p, ref_o), (new_p, new_o) = make(), make()
    for a, b in zip(ref_p, new_p, strict=True):
        a.grad = torch.randn(a.shape, device="cuda")
        b.grad = a.grad.clone()
    for o in ref_o:
        o.step()
        o.zero_grad(set_to_none=True)
    fused_step(new_o)
    for a, b in zip(ref_p, new_p, strict=True):
        assert torch.equal(a.detach(), b.detach())
        assert b.grad is None
    assert not new_o[1].state[new_p[0]] and float(new_o[0].state[new_p[0]]["step"]) == 1.0


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


@pytest.mark.parametrize("bg", [(0.0, 0.0, 0.0), (0.2, 0.5, 1.0)])
def test_batched_composite_terms_equal_per_camera_terms(bg: tuple[float, float, float]) -> None:
    """masked_composite_terms of a camera batch (uint8 targets, channel-first composites) gives each camera's terms of a
    one-camera call on its float target and the same render / alpha gradients (eager, CPU)."""
    from open4dgs.fused_loss import masked_composite_terms

    torch.manual_seed(0)
    b, h, w = 3, 24, 40
    colors = torch.rand(b, h, w, 3) * 1.2 - 0.1
    alphas = torch.rand(b, h, w, 1)
    gt = (torch.rand(b, h, w, 3) * 255).round().to(torch.uint8)
    masks = torch.rand(b, h, w).round()
    masks[1] *= 255.0  # a 0-255 mask is rescaled per camera
    bgt = torch.tensor(bg)
    c1, a1 = colors.clone().requires_grad_(True), alphas.clone().requires_grad_(True)
    ref = [masked_composite_terms(c1[i : i + 1], a1[i : i + 1], gt[i : i + 1].float().div(255.0), masks[i : i + 1], bgt, 1e-4) for i in range(b)]
    c2, a2 = colors.clone().requires_grad_(True), alphas.clone().requires_grad_(True)
    l1, pred_l, gt_l, alpha_out, alpha_mask = masked_composite_terms(c2, a2, gt, masks, bgt, 1e-4)
    for got, k in ((l1, 0), (alpha_out, 3), (alpha_mask, 4)):
        torch.testing.assert_close(got, torch.cat([r[k] for r in ref]), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(pred_l, torch.cat([r[1] for r in ref]))
    torch.testing.assert_close(gt_l, torch.cat([r[2] for r in ref]))
    assert pred_l.is_contiguous() and gt_l.is_contiguous()
    sum((r[0] + r[3] + 0.5 * r[4] + r[1].square().mean()).sum() for r in ref).backward()
    (l1 + alpha_out + 0.5 * alpha_mask + pred_l.square().mean(dim=(1, 2, 3))).sum().backward()
    torch.testing.assert_close(c2.grad, c1.grad, rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(a2.grad, a1.grad, rtol=1e-4, atol=1e-8)
