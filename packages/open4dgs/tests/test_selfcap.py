"""Unit tests for the SelfCap reader helpers and the mesh-free init helpers (no catalog, no GPU)."""

import numpy as np
import pytest

from open4dgs.selfcap_clip import ClipJoints, nearest_frames, rigid_from_columns
from open4dgs.selfcap_init import joint_box_points, robust_affine


def test_rigid_from_columns_reads_column_major() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = rigid_from_columns(rotation.T.reshape(-1), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(transform[:3, :3], rotation)
    np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])


def test_nearest_frames_picks_and_limits() -> None:
    times = np.arange(10) * 0.04 + 0.0189
    np.testing.assert_array_equal(nearest_frames(times, np.array([0.0, 0.08, 0.36]), 0.0201), [0, 2, 9])
    with pytest.raises(ValueError):
        nearest_frames(times, np.array([1.0]), 0.02)


def test_joints_interpolate_and_gap() -> None:
    positions = np.zeros((3, 1, 3), dtype=np.float32)
    positions[:, 0, 0] = [0.0, 1.0, 2.0]
    joints = ClipJoints("j", np.array([0.0, 0.04, 0.5]), positions)
    out = joints.at(np.array([0.02, 0.3, 0.6]))
    assert out[0, 0, 0] == pytest.approx(0.5)
    assert np.isnan(out[1]).all()  # 0.46 s gap
    assert np.isnan(out[2]).all()  # past the end


def test_robust_affine_ignores_outlier() -> None:
    d = np.linspace(1.0, 2.0, 12)
    z = 0.7 * d + 0.5
    z[3] += 1.0
    a, b = robust_affine(d, z)
    assert a == pytest.approx(0.7, abs=0.02)
    assert b == pytest.approx(0.5, abs=0.03)


def test_joint_box_points_adds_head_above_shoulders() -> None:
    joints = np.full((1, 133, 3), np.nan, dtype=np.float32)
    joints[0, 5] = [-0.2, 1.4, 0.0]
    joints[0, 6] = [0.2, 1.4, 0.0]
    points = joint_box_points(joints, np.array([0.0, 1.0, 0.0]))
    assert points.shape == (1, 12 + 48 + 3, 3)
    assert np.nanmax(points[0, :, 1]) == pytest.approx(1.72)
