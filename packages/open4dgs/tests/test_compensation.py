"""The compensation switches map onto TRiGS (defaults unchanged), the rolling-shutter signs, capture times, brightness split (CPU)."""

import dataclasses

import numpy as np
import pytest
import torch


def test_defaults_leave_every_compensation_off_as_trigs_does() -> None:
    from trigs.core.trainer import TrigsTrainingConfig

    from open4dgs.trigs_adapter import TrainConfig

    config = TrainConfig()
    ours = config.to_trigs(fps=30.0, keyframe_step=3, holdout_positions=(1,))
    stock = TrigsTrainingConfig()
    assert ours.learn_camera_appearance is stock.learn_camera_appearance is False
    assert ours.learn_camera_time_offsets is stock.learn_camera_time_offsets is False
    assert ours.camera_time_offset_warmup_steps == stock.camera_time_offset_warmup_steps == 8000
    assert ours.rs_readout_ms == stock.rs_readout_ms == 0.0
    assert not ours.rs_enabled()
    assert config.rs_signs(4) is None


def test_switches_reach_the_trainer_config() -> None:
    from open4dgs.trigs_adapter import TrainConfig

    config = TrainConfig(learn_appearance=True, learn_time_offsets=True, time_offset_warmup_steps=2000, rs_readout_ms=(20.0,), rs_axis="row")
    trigs = config.to_trigs(fps=25.0, keyframe_step=3, holdout_positions=())
    assert trigs.learn_camera_appearance and trigs.learn_camera_time_offsets
    assert trigs.camera_time_offset_warmup_steps == 2000
    assert trigs.camera_time_offset_clamp_s == pytest.approx(0.02)  # half a frame at 25 fps
    assert trigs.rs_readout_ms == 20.0 and trigs.rs_axis == "row" and trigs.rs_enabled()
    explicit = dataclasses.replace(config, time_offset_clamp_ms=5.0).to_trigs(fps=25.0, keyframe_step=3, holdout_positions=())
    assert explicit.camera_time_offset_clamp_s == pytest.approx(0.005)


def test_rolling_shutter_signs_carry_direction_and_readout_share() -> None:
    from open4dgs.trigs_adapter import TrainConfig

    assert TrainConfig(rs_readout_ms=(20.0,)).rs_signs(3) == (1.0, 1.0, 1.0)
    assert TrainConfig(rs_readout_ms=(20.0,), rs_sign=(-1.0,)).rs_signs(2) == (-1.0, -1.0)
    per_camera = TrainConfig(rs_readout_ms=(10.0, 20.0, 0.0), rs_sign=(-1.0, 1.0, 1.0))
    assert per_camera.rs_signs(3) == (-0.5, 1.0, 0.0)
    assert per_camera.to_trigs(fps=25.0, keyframe_step=3, holdout_positions=()).rs_readout_ms == 20.0
    with pytest.raises(ValueError, match="2 values for 3 cameras"):
        TrainConfig(rs_readout_ms=(10.0, 20.0)).rs_signs(3)
    with pytest.raises(ValueError, match=">= 0"):
        TrainConfig(rs_readout_ms=(-1.0,))
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        TrainConfig(rs_sign=(0.5,))


def test_capture_times_are_relative_to_the_reference_camera_start() -> None:
    from open4dgs.trigs_adapter import capture_times_per_frame

    times = np.array([[39.0, 39.04, 39.08], [39.0039, 39.0439, 39.0839]])
    gt = capture_times_per_frame(times)
    assert sorted(gt) == [0, 1, 2]
    assert gt[0][0] == 0.0
    assert gt[2][0] == pytest.approx(0.08)
    assert gt[1][1] == pytest.approx(0.0439)


def test_brightness_split_separates_transparency_from_colour() -> None:
    from open4dgs.compensation import brightness_split

    gt = torch.rand(2, 8, 8, 3) * 0.8 + 0.1
    mask = torch.zeros(2, 8, 8)
    mask[:, 2:6, 2:6] = 1.0
    opaque = torch.ones(2, 8, 8)
    darker = brightness_split(0.5 * gt, opaque, gt, mask)
    assert darker.ratio == pytest.approx(0.5) and darker.alpha_part == pytest.approx(1.0) and darker.colour_part == pytest.approx(0.5)
    half = 0.5 * opaque
    transparent = brightness_split(half[..., None] * gt, half, gt, mask)
    assert transparent.ratio == pytest.approx(0.5) and transparent.alpha_part == pytest.approx(0.5)
    assert transparent.colour_part == pytest.approx(1.0) and transparent.alpha_mean == pytest.approx(0.5)


@pytest.mark.integration
def test_holdout_fit_recovers_an_exposure_change_and_never_moves_a_gaussian() -> None:
    """A held-out camera whose truth is the model's own render at gain 0.8: the test-time fit finds the gain, identity does not."""
    pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (gsplat rasterization)")
    import math

    from trigs.core.trainer import TrigsTrainer

    from open4dgs.catalog_clip import MultiViewClip
    from open4dgs.compensation import evaluate_holdout_fit
    from open4dgs.evaluate import Metrics
    from open4dgs.trigs_adapter import TrainConfig, TrainedClip, camera_config

    dev = torch.device("cuda")
    frames_n, cams_n, h, w, fps = 6, 2, 96, 128, 30.0
    gen = torch.Generator().manual_seed(0)
    n = 3000
    positions = (torch.randn(n, 3, generator=gen) * 0.15).numpy().astype(np.float32)
    keyframes = {
        "positions": np.concatenate([positions, positions]),
        "velocities": np.zeros((2 * n, 3), np.float32),
        "colors": np.tile(torch.rand(n, 3, generator=gen).numpy().astype(np.float32), (2, 1)),
        "times": np.concatenate([np.zeros((n, 1), np.float32), np.ones((n, 1), np.float32)]),
        "has_velocity": np.ones((2 * n,), bool),
        "frame_start": np.asarray(0),
        "frame_end": np.asarray(frames_n - 1),
        "keyframe_step": np.asarray(frames_n - 1),
        "velocity_units": np.asarray("world_per_video_frame"),
    }
    from trigs.init.from_colmap import build_canonical_from_npz

    params, sh_degree, _ = build_canonical_from_npz(keyframes, fps=fps, keyframe_stride=frames_n - 1, device=str(dev))

    def look_at(angle: float) -> torch.Tensor:
        eye = torch.tensor([2.0 * math.cos(angle), 0.0, 2.0 * math.sin(angle)])
        fwd = -eye / eye.norm()
        right = torch.linalg.cross(fwd, torch.tensor([0.0, -1.0, 0.0]))
        right = right / right.norm()
        pose = torch.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, torch.linalg.cross(fwd, right), fwd, eye
        return pose

    world_T_cam = torch.stack([look_at(0.0), look_at(1.0)])
    K = torch.tensor([[120.0, 0, w / 2], [0, 120.0, h / 2], [0, 0, 1]]).expand(cams_n, 3, 3).clone()
    config = TrainConfig(learn_appearance=True)
    trainer = TrigsTrainer(device=str(dev))
    trainer._sequence_t_max = (frames_n - 1) / fps
    model = TrainedClip(
        params=params,
        trainer=trainer,
        trigs_config=config.to_trigs(fps, frames_n - 1, (1,)),
        cameras=camera_config(world_T_cam, K, [(w, h)] * cams_n, dev),
        sh_degree=sh_degree,
        fps=fps,
        num_frames=frames_n,
        scalars={},
        train_seconds=0.0,
        peak_gpu_bytes=0,
    )
    renders = [[model.render(model.cameras, c, f) for f in range(frames_n)] for c in range(cams_n)]
    truth = tuple(torch.stack([(0.8 * rgb * 255).round().to(torch.uint8).cpu() for rgb, _ in renders[c]]) for c in range(cams_n))
    masks = tuple(torch.stack([(alpha[..., 0] > 0.5).float().cpu() for _, alpha in renders[c]]) for c in range(cams_n))
    clip = MultiViewClip(
        camera_names=("rig_00/cam_00", "rig_01/cam_00"),
        frames=truth,
        valid=tuple(torch.ones(h, w, dtype=torch.bool) for _ in range(cams_n)),
        K=K,
        world_T_cam=world_T_cam,
        fps=fps,
        frame_indices=torch.arange(frames_n),
        people=None,
    )
    before = {name: p.detach().clone() for name, p in params.items()}
    (fit,) = evaluate_holdout_fit(model, clip, masks, ("rig_01/cam_00",), Metrics(dev), steps=200, fit_appearance=True, fit_offset=False)
    assert all(torch.equal(before[name], p.detach()) for name, p in params.items())
    # gain and bias are not separately identifiable (the loss clamps the composite at 0, so a negative bias costs nothing on
    # the black background): check the mapped colour, not the gain. A mid-grey of the render must map to about 0.8x.
    mapped = [g * 0.5 + b for g, b in zip(fit.compensation.gain, fit.compensation.bias, strict=True)]
    assert all(0.33 < m < 0.47 for m in mapped), fit.compensation
    identity = np.mean([row.crop_raw_psnr for row in fit.identity.metrics])
    fitted = np.mean([row.crop_raw_psnr for row in fit.fitted.metrics])
    assert fitted > identity + 2.0, (identity, fitted)
