"""SharedTimeGroup.camera_losses (recipe patch 0010's seam): every camera's loss of a train step in one batched pass (one
fused composite loss, one SSIM, the MoGe-2 prior's terms per camera at once) against the per-camera loop it replaces.

One train step on a small synthetic scene, with the batched pass and with the seam returning None (the loop); the
step's parameter gradients and the logged losses must agree up to float summation order.
"""

import math

import numpy as np
import pytest
import torch

trigs = pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")

pytestmark = pytest.mark.integration


def _one_step(batched: bool, prior: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, torch.Tensor], dict[str, list], int]:
    from gsplat import rasterization
    from trigs.core.trainer import TrigsTrainingConfig
    from trigs.init.from_colmap import build_canonical_from_npz

    from open4dgs.adam_bf16 import AdamBf16
    from open4dgs.prior import ClipPriors, MonoPriorLoss, PriorConfig, encode_normals
    from open4dgs.trigs_trainer import HostSizedCameraConfig, Open4dgsTrainer, SharedTimeGroup

    dev = "cuda"
    torch.manual_seed(0)
    F, C, H, W, FPS, n = 6, 5, 64, 80, 30.0, 3000
    pos = torch.randn(n, 3, device=dev) * 0.2
    color = torch.rand(n, 3, device=dev)

    def look_at(eye: torch.Tensor) -> torch.Tensor:
        fwd = -eye / eye.norm()
        right = torch.linalg.cross(fwd, torch.tensor([0.0, -1.0, 0.0]))
        right = right / right.norm()
        c2w = torch.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, torch.linalg.cross(fwd, right), fwd, eye
        return c2w

    c2w = torch.stack([look_at(torch.tensor([2.0 * math.cos(2 * math.pi * i / C), -0.2, 2.0 * math.sin(2 * math.pi * i / C)])) for i in range(C)])
    viewmats = torch.linalg.inv(c2w).to(dev)
    K = torch.tensor([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]], device=dev).expand(C, 3, 3).contiguous()
    gt_images: dict[int, list[torch.Tensor]] = {}
    gt_masks: dict[int, list[torch.Tensor]] = {}
    depths: list[list[torch.Tensor]] = [[] for _ in range(C)]
    with torch.no_grad():
        for f in range(F):
            moved = pos + torch.tensor([0.01 * f, 0.0, 0.0], device=dev)
            rgbd, alpha, _ = rasterization(
                moved,
                torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4),
                torch.full((n, 3), 0.03, device=dev),
                torch.full((n,), 0.9, device=dev),
                color,
                viewmats,
                K,
                W,
                H,
                packed=False,
                render_mode="RGB+ED",
            )
            gt_images[f] = [(rgbd[c, ..., :3].clamp(0, 1) * 255).round().to(torch.uint8) for c in range(C)]
            gt_masks[f] = [(alpha[c, ..., 0] > 0.5).float() for c in range(C)]
            for c in range(C):
                depths[c].append(torch.where(alpha[c, ..., 0] > 0.5, 1.2 * rgbd[c, ..., 3], 0.0).cpu())
    idx = torch.randperm(n, device=dev)[:1500]
    keyframes = {
        "positions": (pos[idx] + 0.02 * torch.randn(1500, 3, device=dev)).cpu().numpy().astype(np.float32),
        "velocities": np.tile(np.asarray([[0.01, 0.0, 0.0]], np.float32), (1500, 1)),
        "colors": color[idx].cpu().numpy().astype(np.float32),
        "times": np.zeros((1500, 1), np.float32),
        "has_velocity": np.ones((1500,), bool),
        "frame_start": np.asarray(0),
        "frame_end": np.asarray(F - 1),
        "keyframe_step": np.asarray(3),
        "velocity_units": np.asarray("world_per_video_frame"),
    }
    params, sh_degree, scene_extent = build_canonical_from_npz(keyframes, fps=FPS, keyframe_stride=3, device=dev)
    config = TrigsTrainingConfig(
        n_steps=1,
        fps=FPS,
        keyframe_stride=3,
        holdout_views=(),
        psnr_eval_every=0,
        tb_log_every=1,
        tb_stats_every=0,
        viewer_snapshot_every=0,
        train_seed=0,
        batch_cameras=4,
        masked_rgb_loss=True,
        masked_rgb_composite=True,
        masked_rgb_bg_color=(0.0, 0.0, 0.0),
        render_depth=prior,
    )
    trainer = Open4dgsTrainer(device=dev, pixel_stride=2, adam_moments="bf16", fused_loss=True)
    if prior:
        maps = ClipPriors(
            tuple(f"c{c}" for c in range(C)),
            torch.arange(F),
            tuple(torch.stack(d).half() for d in depths),
            tuple(torch.zeros(F, H, W, 3, dtype=torch.uint8) + encode_normals(torch.tensor([0.0, 0.0, -1.0]), torch.tensor(True)) for _ in range(C)),
        )
        trainer.camera_loss_hook = MonoPriorLoss(
            PriorConfig(schedule="steps", start_step=0, ramp_end_step=0, log_every=1, depth_edge_max=10.0, min_valid_fraction=0.0),
            maps,
            tuple(torch.stack([m[c].cpu() for m in gt_masks.values()]) for c in range(C)),
            tuple(range(C)),
            K.cpu(),
        )
    grads: dict[str, torch.Tensor] = {}
    optimizer_step = Open4dgsTrainer._optimizer_step

    def step(self, optimizers):
        assert all(isinstance(o, AdamBf16) for o in optimizers.values())
        if not grads:
            grads.update({name: p.grad.detach().clone() for name, p in params.items() if p.grad is not None})
        return optimizer_step(self, optimizers)

    monkeypatch.setattr(Open4dgsTrainer, "_optimizer_step", step)
    calls = {"batched": 0}
    camera_losses = SharedTimeGroup.camera_losses

    def counted(self, *x, **k):
        out = camera_losses(self, *x, **k) if batched else None
        calls["batched"] += out is not None
        return out

    monkeypatch.setattr(SharedTimeGroup, "camera_losses", counted)
    rows: dict[str, list] = {}

    class Scalars:
        def add_scalar(self, name: str, value: float, step: int) -> None:
            rows.setdefault(name, []).append(float(value))

        def __getattr__(self, name: str):
            return lambda *a, **k: None

    cameras = HostSizedCameraConfig.from_matrices(viewmats=viewmats, Ks=K, widths=[W] * C, heights=[H] * C)
    trainer._sequence_t_max = (F - 1) / FPS
    trainer.train_from_params(
        params=params,
        sh_degree=sh_degree,
        scene_extent=scene_extent,
        cameras=cameras,
        n_frames=F,
        gt_images_per_frame=gt_images,
        gt_alphas_per_frame=gt_masks,
        config=config,
        writer=Scalars(),
    )
    monkeypatch.undo()
    if prior:
        trainer.camera_loss_hook.finish()
        rows.update({k: [v for _, v in r] for k, r in trainer.camera_loss_hook.rows.items()})
    return grads, rows, calls["batched"]


@pytest.mark.parametrize("prior", [False, True])
def test_batched_camera_losses_equal_the_camera_loop(prior: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (gsplat rasterization, Mojo SSIM)")
    g_batch, rows_batch, n_batch = _one_step(True, prior, monkeypatch)
    g_loop, rows_loop, n_loop = _one_step(False, prior, monkeypatch)
    assert (n_batch, n_loop) == (1, 0)
    assert g_batch.keys() == g_loop.keys() and "means" in g_batch
    for name, g in g_loop.items():
        scale = float(g.abs().max())
        if scale == 0.0:
            assert float(g_batch[name].abs().max()) == 0.0, name
            continue
        assert float((g_batch[name] - g).abs().max()) <= 1e-4 * scale, name
    for name in ("loss/rgb", "loss/alpha_outside_mask", "loss/alpha_mask", "loss/total") + (("prior/depth", "prior/normal") if prior else ()):
        assert rows_batch[name] == pytest.approx(rows_loop[name], rel=1e-5), name
    if prior:
        assert rows_loop["prior/depth"][0] > 0.0
