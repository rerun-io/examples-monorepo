"""recipes/open4dgs-trigs-src patch 0005: cameras of a train step that share one time share one row gather and deform.

One train step on a small synthetic scene, with the shared path and with the per-camera path (forced by reporting rolling
shutter on, which changes nothing else: the cameras carry no rolling-shutter signs). The renders must be bit-identical and
every parameter gradient the same up to float summation order.
"""

import math

import numpy as np
import pytest
import torch

trigs = pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")

pytestmark = pytest.mark.integration


def _one_step(shared: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], int]:
    from gsplat import rasterization
    from trigs.core.camera import CameraConfig
    from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
    from trigs.init.from_colmap import build_canonical_from_npz

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
    with torch.no_grad():
        for f in range(F):
            moved = pos + torch.tensor([0.01 * f, 0.0, 0.0], device=dev)
            rgb, alpha, _ = rasterization(moved, torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4), torch.full((n, 3), 0.03, device=dev),
                                          torch.full((n,), 0.9, device=dev), color, viewmats, K, W, H, packed=False)
            gt_images[f] = [(rgb[c].clamp(0, 1) * 255).round().to(torch.uint8).cpu() for c in range(C)]
            gt_masks[f] = [(alpha[c, ..., 0] > 0.5).float().cpu() for c in range(C)]
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
    config = TrigsTrainingConfig(n_steps=1, fps=FPS, keyframe_stride=3, holdout_views=(), psnr_eval_every=0, tb_log_every=1, tb_stats_every=0,
                                 viewer_snapshot_every=0, train_seed=0, batch_cameras=4)
    if not shared:
        monkeypatch.setattr(TrigsTrainingConfig, "rs_enabled", lambda self: True)
    renders: list[torch.Tensor] = []
    grads: dict[str, torch.Tensor] = {}
    trainer = TrigsTrainer(device=dev)
    render_inputs, photometric = trainer._render_inputs, trainer._photometric_losses
    used = {"inputs": 0}

    def count(*x, **k):
        used["inputs"] += 1
        return render_inputs(*x, **k)

    def keep(p, render_colors, *x, **k):
        renders.append(render_colors.detach().clone())
        return photometric(p, render_colors, *x, **k)

    monkeypatch.setattr(trainer, "_photometric_losses", keep)
    monkeypatch.setattr(trainer, "_render_inputs", count)
    adam_step = torch.optim.Adam.step

    def step(self, *x, **k):
        if not grads:
            grads.update({name: p.grad.detach().clone() for name, p in params.items() if p.grad is not None})
        return adam_step(self, *x, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", step)

    class Scalars:
        def __getattr__(self, name: str):
            return lambda *a, **k: None

    cameras = CameraConfig.from_matrices(viewmats=viewmats, Ks=K, widths=[W] * C, heights=[H] * C)
    trainer._sequence_t_max = (F - 1) / FPS
    trainer.train_from_params(params=params, sh_degree=sh_degree, scene_extent=scene_extent, cameras=cameras, n_frames=F,
                              gt_images_per_frame=gt_images, gt_alphas_per_frame=gt_masks, config=config, writer=Scalars())
    monkeypatch.undo()
    return renders, grads, used["inputs"]


def test_shared_deform_step_matches_per_camera_step(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (gsplat rasterization)")
    r_shared, g_shared, n_shared = _one_step(True, monkeypatch)
    r_cam, g_cam, n_cam = _one_step(False, monkeypatch)
    # One active-row query + gather + deform for the shared step, one per camera otherwise.
    assert len(r_shared) == len(r_cam) == 4
    assert (n_shared, n_cam) == (1, 4)
    for a, b in zip(r_shared, r_cam, strict=True):
        assert torch.equal(a, b)
    assert g_shared.keys() == g_cam.keys() and "means" in g_shared
    for name, g in g_cam.items():
        scale = float(g.abs().max())
        if scale == 0.0:
            assert float(g_shared[name].abs().max()) == 0.0, name
            continue
        assert float((g_shared[name] - g).abs().max()) <= 1e-4 * scale, name
