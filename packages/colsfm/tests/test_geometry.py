"""SE(3) conventions: axis-angle degrees, TUM lines, relative poses.

Expected values are taken from `docs/spec/export.md` §3.4 and §4.3, which quote
them off cuSFM's own artifacts, so nothing here is recomputed the way the code
computes it.
"""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest

from colsfm.geometry import (
    AxisAngleDegrees,
    axis_angle_degrees_from_rigid3d,
    format_tum_line,
    parse_tum_line,
    quaternion_wxyz,
    relative_pose,
    relative_rotation_degrees,
    rigid3d_from_axis_angle_degrees,
)


def test_axis_angle_degrees_inverts_to_the_colmap_pose_in_the_spec() -> None:
    """Image 2559's `camera_to_world` inverts to the shipped `images.txt` row.

    Literals from docs/spec/export.md §3.4.
    """
    world_T_cam: pycolmap.Rigid3d = rigid3d_from_axis_angle_degrees(
        axis_xyz=np.array([-0.6497364711905401, -0.5354993856596588, 0.5395210153117233]),
        angle_degrees=113.49233777718922,
        translation_xyz=np.array([-0.19074984903419873, 0.1490971270102635, 0.350389792773112]),
    )
    cam_T_world: pycolmap.Rigid3d = world_T_cam.inverse()
    expected_quat_wxyz: np.ndarray = np.array([0.548349146992837, 0.543341794014345, 0.447811089263254, -0.451174175017098])
    expected_translation: np.ndarray = np.array([-0.110030856003945, 0.348453786441002, -0.218773020527753])
    assert np.allclose(quaternion_wxyz(cam_T_world), expected_quat_wxyz, atol=1e-14)
    assert np.allclose(cam_T_world.translation, expected_translation, atol=1e-14)


def test_quaternion_wxyz_is_canonicalised_to_non_negative_w() -> None:
    """COLMAP's `images.txt` never carries a negative QW (export.md §3.4 step 4)."""
    flipped: pycolmap.Rigid3d = pycolmap.Rigid3d(pycolmap.Rotation3d(np.array([0.1, 0.2, 0.3, -0.927])), np.zeros(3))
    assert quaternion_wxyz(flipped)[0] > 0.0


def test_axis_angle_round_trips_through_rigid3d() -> None:
    """Axis-angle in degrees survives the trip to `Rigid3d` and back."""
    axis_xyz: np.ndarray = np.array([-0.6497364711905401, -0.5354993856596588, 0.5395210153117233])
    pose: pycolmap.Rigid3d = rigid3d_from_axis_angle_degrees(
        axis_xyz=axis_xyz, angle_degrees=113.49233777718922, translation_xyz=np.array([1.0, -2.0, 3.0])
    )
    recovered: AxisAngleDegrees = axis_angle_degrees_from_rigid3d(pose)
    assert np.allclose(recovered.axis_xyz, axis_xyz, atol=1e-14)
    assert recovered.angle_degrees == pytest.approx(113.49233777718922, abs=1e-12)
    assert np.allclose(recovered.translation_xyz, np.array([1.0, -2.0, 3.0]), atol=1e-15)


def test_zero_rotation_round_trips_to_a_unit_axis() -> None:
    """An identity rotation still yields a usable unit axis rather than zeros."""
    recovered: AxisAngleDegrees = axis_angle_degrees_from_rigid3d(pycolmap.Rigid3d())
    assert recovered.angle_degrees == pytest.approx(0.0)
    assert np.linalg.norm(recovered.axis_xyz) == pytest.approx(1.0)


def test_tum_timestamp_round_trips_as_integer_microseconds() -> None:
    """Odd microseconds survive the round trip that cuSFM's truncating parser loses.

    export.md §5.4: `1707938736402913` comes back as `...912` in the blob because
    it truncates; the port rounds.
    """
    timestamp_microseconds: int = 1707938736402913
    line: str = format_tum_line(timestamp_microseconds, pycolmap.Rigid3d())
    assert parse_tum_line(line).timestamp_microseconds == timestamp_microseconds


def test_relative_pose_composes_back_to_the_target() -> None:
    """`source_T_target` maps the target frame into the source frame."""
    world_T_source: pycolmap.Rigid3d = rigid3d_from_axis_angle_degrees(
        axis_xyz=np.array([0.0, 0.0, 1.0]), angle_degrees=30.0, translation_xyz=np.array([1.0, 0.0, 0.0])
    )
    world_T_target: pycolmap.Rigid3d = rigid3d_from_axis_angle_degrees(
        axis_xyz=np.array([0.0, 1.0, 0.0]), angle_degrees=-12.0, translation_xyz=np.array([0.0, 2.0, -1.0])
    )
    source_T_target: pycolmap.Rigid3d = relative_pose(world_T_source, world_T_target)
    composed: pycolmap.Rigid3d = world_T_source * source_T_target
    assert np.allclose(composed.translation, world_T_target.translation, atol=1e-14)
    assert np.allclose(quaternion_wxyz(composed), quaternion_wxyz(world_T_target), atol=1e-14)


def test_relative_rotation_degrees_matches_the_axis_angle_magnitude() -> None:
    """The angle between two poses equals the axis-angle magnitude of their relative rotation."""
    world_T_source: pycolmap.Rigid3d = pycolmap.Rigid3d()
    world_T_target: pycolmap.Rigid3d = rigid3d_from_axis_angle_degrees(
        axis_xyz=np.array([0.0, 0.0, 1.0]), angle_degrees=42.5, translation_xyz=np.zeros(3)
    )
    assert relative_rotation_degrees(world_T_source, world_T_target) == pytest.approx(42.5, abs=1e-12)
