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
