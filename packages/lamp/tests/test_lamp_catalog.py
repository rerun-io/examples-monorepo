"""Pure tests for the Robocap catalog application's time and config contracts."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lamptrack.apis.lamp_catalog import Config, best_detection_window, build_time_grid, interpolate_pose


def test_catalog_defaults_select_outward_robocap_cameras() -> None:
    """The default stream order is the settled four-view LAMP order."""
    config = Config()

    assert config.segment_id == "robocap__f408193e6447b3b0__s00000029"
    assert config.cams == ("cam_00", "cam_01", "cam_04", "cam_05")
    assert config.fps == 10.0
    assert config.start_s == 1152.0
    assert config.max_seconds == 120.0


def test_catalog_rejects_more_than_four_cameras_even_when_four_are_distinct() -> None:
    """Runtime validation rejects malformed CLI camera lists, not only duplicates."""
    with pytest.raises(ValueError, match="four distinct"):
        Config(cams=("cam_00", "cam_01", "cam_04", "cam_05", "cam_00"))


def test_build_time_grid_uses_shared_video_overlap_and_absolute_start() -> None:
    """Sampling begins at the requested video-time second within all streams."""
    video_times = [
        np.arange(437, 1001, dtype="timedelta64[s]").astype("timedelta64[ns]"),
        np.arange(438, 1002, dtype="timedelta64[s]").astype("timedelta64[ns]"),
    ]

    grid = build_time_grid(video_times, fps=2.0, start_s=880.0, max_seconds=2.0)

    assert np.array_equal(grid, np.asarray([880.0, 880.5, 881.0, 881.5], dtype=np.float64) * 1e9)


def test_interpolate_pose_interpolates_translation_and_rotation() -> None:
    """Rig motion uses linear translation and quaternion slerp."""
    times = np.asarray([0, 2_000_000_000], dtype=np.int64)
    poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    poses[1, :3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    poses[1, :3, 3] = np.asarray([2.0, 4.0, 6.0])

    midpoint = interpolate_pose(times, poses, 1_000_000_000)

    assert np.allclose(midpoint[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(midpoint[:3, :3], Rotation.from_euler("z", 45.0, degrees=True).as_matrix())
    assert np.array_equal(midpoint[3], [0.0, 0.0, 0.0, 1.0])


def test_best_detection_window_uses_earliest_maximum() -> None:
    """The 120-second selection is deterministic when two windows tie."""
    sample_seconds = np.arange(600.0, 901.0)
    counts = np.zeros_like(sample_seconds, dtype=np.int64)
    counts[(sample_seconds >= 650.0) & (sample_seconds < 660.0)] = 2
    counts[(sample_seconds >= 780.0) & (sample_seconds < 790.0)] = 2

    start_s, total = best_detection_window(sample_seconds, counts, window_seconds=120.0)

    assert start_s == 600.0
    assert total == 20
