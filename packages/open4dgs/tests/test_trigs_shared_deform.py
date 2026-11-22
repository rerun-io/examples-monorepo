"""open4dgs.trigs_trainer.Open4dgsTrainer on the recipe's seams (patches 0005 + 0006): cameras of a train step that share one
time share one row gather and deform (lever 2), and with batch_cameras one rasterization call per image size (lever 3).

One train step on a small synthetic scene with Open4dgsTrainer and with the stock TrigsTrainer, whose seams are no-ops
(the per-camera path). The renders must be bit-identical and every parameter gradient the same up to float summation order.
"""

import math

import numpy as np
import pytest
import torch

trigs = pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")

pytestmark = pytest.mark.integration


def _one_step(kind: str, monkeypatch: pytest.MonkeyPatch) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], tuple[int, int]]:
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
    renders: list[torch.Tensor] = []
    grads: dict[str, torch.Tensor] = {}
    from open4dgs.trigs_trainer import Open4dgsTrainer

    trainer = TrigsTrainer(device=dev) if kind == "stock" else Open4dgsTrainer(device=dev, batch_cameras=kind == "batched")
    render_inputs, photometric, rasterize = trainer._render_inputs, trainer._photometric_losses, trainer._rasterize_inputs
    used = {"inputs": 0, "rasterize": 0}

    def count_rasterize(*x, **k):
        used["rasterize"] += 1
        return rasterize(*x, **k)

    def count(*x, **k):
        used["inputs"] += 1
        return render_inputs(*x, **k)

    def keep(p, render_colors, *x, **k):
        renders.append(render_colors.detach().clone())
        return photometric(p, render_colors, *x, **k)

    monkeypatch.setattr(trainer, "_photometric_losses", keep)
    monkeypatch.setattr(trainer, "_render_inputs", count)
    monkeypatch.setattr(trainer, "_rasterize_inputs", count_rasterize)
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
    return renders, grads, (used["inputs"], used["rasterize"])


@pytest.mark.parametrize(("kind", "calls"), [("batched", (1, 1)), ("shared", (1, 4))])
def test_open4dgs_trainer_step_matches_the_stock_per_camera_step(kind: str, calls: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (gsplat rasterization)")
    r_fast, g_fast, n_fast = _one_step(kind, monkeypatch)
    r_cam, g_cam, n_cam = _one_step("stock", monkeypatch)
    # (render-input calls, rasterize calls) for 4 same-size cameras: the stock trainer goes per camera through the seams.
    assert len(r_fast) == len(r_cam) == 4
    assert (n_fast, n_cam) == (calls, (4, 4))
    for a, b in zip(r_fast, r_cam, strict=True):
        assert torch.equal(a, b)
    assert g_fast.keys() == g_cam.keys() and "means" in g_fast
    for name, g in g_cam.items():
        scale = float(g.abs().max())
        if scale == 0.0:
            assert float(g_fast[name].abs().max()) == 0.0, name
            continue
        assert float((g_fast[name] - g).abs().max()) <= 1e-4 * scale, name
