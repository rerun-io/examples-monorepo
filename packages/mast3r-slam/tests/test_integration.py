"""Integration lane: the Rerun round trips of the edge Types through an in-process server (a catalog). The Gauss-Newton
Op's GPU parity with the CUDA extension is in test_mojo_vs_cuda.py."""

import numpy as np
import pytest
from dagkit.types import rerun_roundtrip

from mast3r_slam.types import KeyframeCloud, MapUpdate, TrackedFrame

pytestmark = pytest.mark.integration


def test_tracked_frame_rerun_round_trip() -> None:
    value = TrackedFrame(42, np.array([0.5, -1.0, 2.0, 0.0, 0.6, 0.0, 0.8, 0.25], dtype=np.float32), world=1)
    [(t_ns, back)] = rerun_roundtrip(value, "/mast3r_slam/tracked/cam_00", "video_time", 1_500_000_000)
    assert t_ns == 1_500_000_000 and back.frame_index == 42 and back.world == 1
    np.testing.assert_array_equal(back.world_sim3_cam, value.world_sim3_cam)


@pytest.mark.parametrize("n_edges", [0, 3])
def test_map_update_rerun_round_trip(n_edges: int) -> None:
    poses = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32), (3, 1))
    poses[:, 0] = [0.0, 1.0, 2.0]
    value = MapUpdate(poses, np.array([0, 1, 0][:n_edges], dtype=np.int64), np.array([1, 2, 2][:n_edges], dtype=np.int64), world=2)
    [(t_ns, back)] = rerun_roundtrip(value, "/mast3r_slam/map", "video_time", 7)
    assert t_ns == 7 and back.world == 2
    np.testing.assert_array_equal(back.world_sim3_cam, poses)
    np.testing.assert_array_equal(np.stack([back.ii, back.jj]), np.stack([value.ii, value.jj]))


def test_keyframe_cloud_rerun_round_trip() -> None:
    value = KeyframeCloud(12, np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32), np.array([[255, 0, 10], [1, 2, 3]], dtype=np.uint8))
    [(t_ns, back)] = rerun_roundtrip(value, "/mast3r_slam/map/keyframes", "video_time", 9)
    assert t_ns == 9 and back.index == 12
    np.testing.assert_array_equal(back.points, value.points)
    np.testing.assert_array_equal(back.colors, value.colors)
