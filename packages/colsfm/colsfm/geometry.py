"""SE(3) helpers on top of `pycolmap.Rigid3d`, in cuSFM's conventions.

Three rotation conventions meet in this pipeline and mixing them is the easiest
way to get a plausible-looking but wrong reconstruction:

| Where | Encoding |
|---|---|
| cuSFM `frames_meta.json` | axis-angle, unit axis + angle in **degrees** |
| COLMAP `images.txt` | quaternion `QW QX QY QZ`, canonicalised to `QW >= 0` |
| cuSFM TUM pose files | quaternion `qx qy qz qw`, no canonicalisation |
| `pycolmap.Rotation3d` | quaternion `xyzw` |

Directions follow the `dst_T_src` naming rule: `camera_to_world` in the metadata
is `world_T_cam`, `sensor_to_vehicle_transform` is `vehicle_T_cam`, and COLMAP's
`cam_from_world` is `cam_T_world`, the inverse of the first.

References: `docs/spec/export.md` §2, §3.4 and §4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NamedTuple, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

Vector3: TypeAlias = Float64[ndarray, "3"]
"""A 3-vector: a translation in metres or a rotation axis."""

Matrix3: TypeAlias = Float64[ndarray, "3 3"]
"""A rotation matrix."""

QuaternionWXYZ: TypeAlias = Float64[ndarray, "4"]
"""A quaternion in COLMAP's `w x y z` order."""

QuaternionXYZW: TypeAlias = Float64[ndarray, "4"]
"""A quaternion in Eigen / scipy / TUM `x y z w` order."""

TUM_DECIMALS: Final[int] = 16
"""`std::fixed` precision cuSFM writes TUM columns with (export.md §4.3)."""

MICROSECONDS_PER_SECOND: Final[float] = 1e6
"""Timestamps are integer microseconds everywhere except inside a TUM line."""

MILLIMETRES_PER_METRE: Final[float] = 1000.0
"""Everything is stored in metres; extrinsic and trajectory errors are reported in mm."""

DEFAULT_AXIS: Final[Vector3] = np.array([1.0, 0.0, 0.0])
"""Axis reported for a zero rotation, where the true axis is undefined."""


@dataclass(frozen=True, slots=True)
class AxisAngleDegrees:
    """A rigid transform in cuSFM's `RigidTransform3d` encoding."""

    axis_xyz: Vector3
    """Unit rotation axis, dimensionless."""
    angle_degrees: float
    """Rotation about `axis_xyz`, in degrees."""
    translation_xyz: Vector3
    """Translation in metres."""


class TumPose(NamedTuple):
    """One line of a TUM trajectory file.

    `typing.NamedTuple` refuses an explicit `__slots__` on Python 3.12; the tuple
    base already prevents attribute drift.
    """

    timestamp_microseconds: int
    """Integer microseconds since the Unix epoch, carried losslessly."""
    world_T_body: pycolmap.Rigid3d
    """Pose of the body (vehicle or camera) in the world frame."""


def rigid3d_from_axis_angle_degrees(*, axis_xyz: Vector3, angle_degrees: float, translation_xyz: Vector3) -> pycolmap.Rigid3d:
    """Build a `Rigid3d` from cuSFM's axis-angle-in-degrees encoding.

    Args:
        axis_xyz: Float64 rotation axis with shape `[3]`; normalised internally.
        angle_degrees: Rotation angle in degrees.
        translation_xyz: Float64 translation in metres with shape `[3]`.

    Returns:
        The equivalent rigid transform.
    """
    norm: float = float(np.linalg.norm(axis_xyz))
    unit_axis_xyz: Vector3 = DEFAULT_AXIS if norm == 0.0 else axis_xyz / norm
    rotation_vector: Vector3 = unit_axis_xyz * np.deg2rad(angle_degrees)
    quaternion_xyzw: QuaternionXYZW = Rotation.from_rotvec(rotation_vector).as_quat()
    return pycolmap.Rigid3d(pycolmap.Rotation3d(quaternion_xyzw), np.asarray(translation_xyz, dtype=np.float64))


def rigid3d_from_matrix(rotation_matrix: Matrix3, translation_xyz: Vector3) -> pycolmap.Rigid3d:
    """Build a `Rigid3d` from a rotation matrix and a translation.

    `pycolmap.Rotation3d` takes a quaternion, so every caller that holds a matrix
    would otherwise spell the same `Rotation.from_matrix(...).as_quat()` detour.

    Args:
        rotation_matrix: Float64 rotation matrix with shape `[3, 3]`.
        translation_xyz: Float64 translation in metres with shape `[3]`.

    Returns:
        The equivalent rigid transform.
    """
    quaternion_xyzw: QuaternionXYZW = Rotation.from_matrix(np.asarray(rotation_matrix, dtype=np.float64)).as_quat()
    return pycolmap.Rigid3d(pycolmap.Rotation3d(quaternion_xyzw), np.asarray(translation_xyz, dtype=np.float64))


def axis_angle_degrees_from_rigid3d(pose: pycolmap.Rigid3d) -> AxisAngleDegrees:
    """Convert a `Rigid3d` back to cuSFM's axis-angle-in-degrees encoding.

    Args:
        pose: Rigid transform.

    Returns:
        Unit axis, angle in degrees and translation in metres. A zero rotation
        reports `DEFAULT_AXIS` with a zero angle, since its axis is undefined.
    """
    rotation_vector: Vector3 = Rotation.from_quat(pose.rotation.quat).as_rotvec()
    angle_radians: float = float(np.linalg.norm(rotation_vector))
    axis_xyz: Vector3 = DEFAULT_AXIS if angle_radians == 0.0 else rotation_vector / angle_radians
    return AxisAngleDegrees(
        axis_xyz=axis_xyz,
        angle_degrees=float(np.rad2deg(angle_radians)),
        translation_xyz=np.asarray(pose.translation, dtype=np.float64),
    )


def quaternion_wxyz(pose: pycolmap.Rigid3d) -> QuaternionWXYZ:
    """Return the rotation as a normalised `w x y z` quaternion with `w >= 0`.

    This is COLMAP `images.txt` order plus the sign canonicalisation cuSFM
    applies there (export.md §3.4 step 4). TUM output must not use this.

    Args:
        pose: Rigid transform.

    Returns:
        Float64 quaternion with shape `[4]` in `w x y z` order.
    """
    quaternion_xyzw: QuaternionXYZW = np.asarray(pose.rotation.quat, dtype=np.float64)
    quaternion_xyzw = quaternion_xyzw / float(np.linalg.norm(quaternion_xyzw))
    if quaternion_xyzw[3] < 0.0:
        quaternion_xyzw = -quaternion_xyzw
    return np.array([quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]])


def relative_pose(world_T_source: pycolmap.Rigid3d, world_T_target: pycolmap.Rigid3d) -> pycolmap.Rigid3d:
    """Compose the relative transform between two world poses.

    Args:
        world_T_source: Source body's pose in the world frame.
        world_T_target: Target body's pose in the world frame.

    Returns:
        `source_T_target`, which maps target-frame coordinates into the source frame.
    """
    return world_T_source.inverse() * world_T_target


def relative_rotation_degrees(world_T_source: pycolmap.Rigid3d, world_T_target: pycolmap.Rigid3d) -> float:
    """Angle between two poses' rotations, in degrees.

    Equivalent to `acos((trace(R_source^T R_target) - 1) / 2)`, the form
    `feature_extractor_main` uses for its keyframe rotation gate.

    Args:
        world_T_source: Source body's pose in the world frame.
        world_T_target: Target body's pose in the world frame.

    Returns:
        The rotation angle in degrees, in `[0, 180]`.
    """
    return float(np.rad2deg(world_T_source.rotation.angle_to(world_T_target.rotation)))


def format_tum_line(timestamp_microseconds: int, world_T_body: pycolmap.Rigid3d) -> str:
    """Format one TUM line exactly as cuSFM's `PoseSerializer::TumRecord` does.

    Columns are `timestamp tx ty tz qx qy qz qw`, single-space separated, every
    field `std::fixed` with 16 decimals. The timestamp is `microseconds * 1e-6`
    — a multiplication by the reciprocal, not a division by `1e6`; the two differ
    by one ULP on some values and the blob always matches the multiply
    (export.md §4.3). The quaternion is written raw, with no `w >= 0` flip.

    Args:
        timestamp_microseconds: Integer microseconds since the Unix epoch.
        world_T_body: Pose of the vehicle or camera in the world frame.

    Returns:
        The line, without a trailing newline.
    """
    seconds: float = float(timestamp_microseconds) * 1e-06
    quaternion_xyzw: QuaternionXYZW = np.asarray(world_T_body.rotation.quat, dtype=np.float64)
    translation_xyz: Vector3 = np.asarray(world_T_body.translation, dtype=np.float64)
    fields: list[float] = [seconds, *translation_xyz.tolist(), *quaternion_xyzw.tolist()]
    return " ".join(f"{value:.{TUM_DECIMALS}f}" for value in fields)


def parse_tum_line(line: str) -> TumPose:
    """Parse one TUM line back into integer microseconds and a `Rigid3d`.

    Rounds the timestamp instead of truncating it. cuSFM truncates, which turns
    `1707938736402913` into `...912` and silently drops keyframes downstream
    (export.md §5.4); this port refuses to inherit that.

    Args:
        line: One whitespace-separated `timestamp tx ty tz qx qy qz qw` row.

    Returns:
        The timestamp in microseconds and the world pose of the body.

    Raises:
        ValueError: When the line does not have eight numeric columns.
    """
    fields: list[str] = line.split()
    if len(fields) != 8:
        raise ValueError(f"Expected 8 TUM columns, got {len(fields)}: {line!r}")
    values: list[float] = [float(field) for field in fields]
    timestamp_microseconds: int = round(values[0] * MICROSECONDS_PER_SECOND)
    translation_xyz: Vector3 = np.array(values[1:4])
    quaternion_xyzw: QuaternionXYZW = np.array(values[4:8])
    return TumPose(
        timestamp_microseconds=timestamp_microseconds,
        world_T_body=pycolmap.Rigid3d(pycolmap.Rotation3d(quaternion_xyzw), translation_xyz),
    )
