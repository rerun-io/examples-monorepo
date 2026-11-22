"""TRiGS (recipes/open4dgs-trigs-src) trains in memory on a tiny synthetic dynamic multi-view scene on the GPU.

No files: cameras, frames, masks and the init cloud (the keyframes arrays as a dict, recipe patch 0002) are handed to
the trainer in memory; PyTorch3D's ops come from open4dgs.pytorch3d_ops (patch 0001). Held-out camera 5 never
enters training; its PSNR must rise.
"""

import math

import numpy as np
import pytest
import torch

trigs = pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")

pytestmark = pytest.mark.integration


def test_trigs_trains_in_memory_and_holdout_psnr_rises() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (gsplat rasterization)")
    from gsplat import rasterization
    from trigs.core.camera import CameraConfig
    from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
    from trigs.init.from_colmap import build_canonical_from_npz

    n_steps = 200
    dev = "cuda"
    torch.manual_seed(0)
    F, C, H, W, FPS, HOLD = 12, 6, 120, 160, 30.0, 5

    # Ground truth: two coloured blobs; blob A translates along +x, blob B orbits around y.
    n = 4000
    base = torch.randn(n, 3, device=dev) * 0.12
    which = torch.arange(n, device=dev) < n // 2
    centre_a = torch.tensor([-0.3, 0.0, 0.0], device=dev)
    color = torch.where(which[:, None], torch.tensor([0.9, 0.3, 0.2], device=dev), torch.tensor([0.2, 0.5, 0.9], device=dev))
    color = (color + 0.1 * torch.randn_like(color)).clamp(0, 1)

    def gt_means(frame: int) -> torch.Tensor:
        a = base + centre_a + torch.tensor([0.02 * frame, 0.0, 0.0], device=dev)
        ang = 0.08 * frame
        b = base + torch.tensor([0.35 * math.cos(ang), 0.1, 0.35 * math.sin(ang)], device=dev)
        return torch.where(which[:, None], a, b)

    # Cameras on a ring of radius 2.5 looking at the origin (OpenCV RDF, world_T_cam inverted to gsplat viewmats).
    def look_at(eye: torch.Tensor) -> torch.Tensor:
        fwd = -eye / eye.norm()
        right = torch.linalg.cross(fwd, torch.tensor([0.0, -1.0, 0.0]))
        right = right / right.norm()
        down = torch.linalg.cross(fwd, right)
        c2w = torch.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
        return c2w

    c2w = torch.stack([look_at(torch.tensor([2.5 * math.cos(2 * math.pi * i / C), -0.3, 2.5 * math.sin(2 * math.pi * i / C)])) for i in range(C)])
    viewmats = torch.linalg.inv(c2w).to(dev)
    K = torch.tensor([[140.0, 0, W / 2], [0, 140.0, H / 2], [0, 0, 1]], device=dev).expand(C, 3, 3).contiguous()
    quats = torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4)
    scales = torch.full((n, 3), 0.025, device=dev)
    opac = torch.full((n,), 0.9, device=dev)
    # [frame][camera]: RGB uint8 (H, W, 3) and mask float (H, W), the trainer's Dict[int, List[Tensor]].
    gt_images: dict[int, list[torch.Tensor]] = {}
    gt_masks: dict[int, list[torch.Tensor]] = {}
    with torch.no_grad():
        for f in range(F):
            rgb, alpha, _ = rasterization(gt_means(f), quats, scales, opac, color, viewmats, K, W, H, packed=False)
            gt_images[f] = [(rgb[c].clamp(0, 1) * 255).round().to(torch.uint8).cpu() for c in range(C)]
            gt_masks[f] = [(alpha[c, ..., 0] > 0.5).float().cpu() for c in range(C)]

    # Init cloud ("keyframes" arrays, in memory): noisy subsample of the truth at every 3rd frame, velocity per video frame.
    step_k = 3
    rows = []
    for kf in range(0, F, step_k):
        idx = torch.randperm(n, device=dev)[:800]
        p0 = gt_means(kf)[idx]
        p1 = gt_means(min(kf + 1, F - 1))[idx]
        vel = (p1 - p0) if kf + 1 < F else (gt_means(kf)[idx] - gt_means(kf - 1)[idx])
        rows.append((p0 + 0.01 * torch.randn_like(p0), vel, (color[idx] + 0.15 * torch.randn_like(color[idx])).clamp(0, 1), kf))
    keyframes = {
        "positions": torch.cat([r[0] for r in rows]).cpu().numpy().astype(np.float32),
        "velocities": torch.cat([r[1] for r in rows]).cpu().numpy().astype(np.float32),
        "colors": torch.cat([r[2] for r in rows]).cpu().numpy().astype(np.float32),
        "times": np.concatenate([np.full((800, 1), r[3] / (F - 1), np.float32) for r in rows]),
        "has_velocity": np.ones((800 * len(rows),), bool),
        "frame_start": np.asarray(0),
        "frame_end": np.asarray(F - 1),
        "keyframe_step": np.asarray(step_k),
        "velocity_units": np.asarray("world_per_video_frame"),
    }
    params, sh_degree, scene_extent = build_canonical_from_npz(keyframes, fps=FPS, keyframe_stride=step_k, device=dev)

    config = TrigsTrainingConfig(
        n_steps=n_steps,
        fps=FPS,
        keyframe_stride=step_k,
        holdout_views=(HOLD,),
        psnr_eval_every=100,
        psnr_eval_num_samples=12,
        tb_log_every=10,
        tb_stats_every=0,
        viewer_snapshot_every=0,
        train_seed=0,
        batch_cameras=4,
    )

    class Scalars:
        """Duck-typed SummaryWriter: the trainer reports losses and eval PSNR through writer.add_scalar."""

        def __init__(self) -> None:
            self.rows: dict[str, list[tuple[int, float]]] = {}

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.rows.setdefault(tag, []).append((step, float(value)))

        def __getattr__(self, name: str):
            return lambda *a, **k: None

    trainer = TrigsTrainer(device=dev)

    @torch.no_grad()
    def holdout_psnr(p) -> tuple[float, torch.Tensor]:
        vals = []
        for f in range(F):
            rgb, _, _, _ = trainer._render(p, cameras, HOLD, f / FPS, sh_degree, config)
            gt = gt_images[f][HOLD].to(dev).float() / 255
            vals.append(float(-10 * torch.log10(((rgb[0].clamp(0, 1) - gt) ** 2).mean())))
        rgb, _, _, _ = trainer._render(p, cameras, HOLD, (F // 2) / FPS, sh_degree, config)
        return float(np.mean(vals)), rgb[0].clamp(0, 1)

    cameras = CameraConfig.from_matrices(viewmats=viewmats, Ks=K, widths=[W] * C, heights=[H] * C)
    trainer._sequence_t_max = (F - 1) / FPS
    psnr_before, _ = holdout_psnr(params)

    writer = Scalars()
    result = trainer.train_from_params(
        params=params,
        sh_degree=sh_degree,
        scene_extent=scene_extent,
        cameras=cameras,
        n_frames=F,
        gt_images_per_frame=gt_images,
        gt_alphas_per_frame=gt_masks,
        config=config,
        writer=writer,
    )
    torch.cuda.synchronize()
    psnr_after, render = holdout_psnr(result.params)

    losses = writer.rows["loss/rgb"]
    assert np.mean([v for _, v in losses[-3:]]) < 0.5 * np.mean([v for _, v in losses[:3]])
    assert psnr_after > psnr_before + 5.0
    assert render.shape == (H, W, 3)
