"""Frameset grouping and pose association, on synthetic times (no catalog)."""

import numpy as np
import pytest

from robocap_live.catalog_segment import CatalogPoses, group_framesets, nearest_poses


def test_a_frameset_opens_at_the_earliest_frame_and_takes_each_cameras_next_frame_within_the_tolerance() -> None:
    cameras = [
        np.array([0, 33_000_000, 66_000_000], dtype=np.int64),
        np.array([1_000_000, 34_500_000, 67_000_000], dtype=np.int64),
        # Starts late and drifts out of the tolerance on the second frame.
        np.array([2_500_000, 40_000_000], dtype=np.int64),
    ]
    index = group_framesets(cameras, tolerance_ns=3_000_000)
    assert index.t_ns.tolist() == [0, 33_000_000, 40_000_000, 66_000_000]
    assert index.frame.tolist() == [[0, 0, 0], [1, 1, -1], [-1, -1, 1], [2, 2, -1]]
    assert index.cam_t_ns[1].tolist() == [33_000_000, 34_500_000, 0]
    assert index.head(2).t_ns.tolist() == [0, 33_000_000]


def test_grouping_refuses_frame_times_that_do_not_increase_and_a_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="camera 1"):
        group_framesets([np.array([0, 1], dtype=np.int64), np.array([5, 5], dtype=np.int64)], tolerance_ns=3)
    with pytest.raises(ValueError, match="negative"):
        group_framesets([np.array([0, 1], dtype=np.int64)], tolerance_ns=-1)


def test_each_frameset_takes_the_nearest_pose_within_the_tolerance_and_nan_otherwise() -> None:
    poses = CatalogPoses(t_ns=np.array([100, 200], dtype=np.int64), world_from_rig=np.stack([np.eye(4), 2.0 * np.eye(4)]))
    matched = nearest_poses(np.array([95, 151, 149, 260, 207], dtype=np.int64), poses, tolerance_ns=10)
    assert matched[0, 0, 0] == 1.0 and matched[4, 0, 0] == 2.0
    assert np.isnan(matched[1:4]).all()
    assert np.isnan(nearest_poses(np.array([1], dtype=np.int64), CatalogPoses(np.zeros(0, np.int64), np.zeros((0, 4, 4))), 10)).all()
