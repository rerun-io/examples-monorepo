"""Trajectory, calibration, and baked pixel orientation.

Images and geometry are baked into gravity-upright camera frames. Acceleration
and gyroscope Scalars deliberately remain in the original device frame: they
are frame-agnostic numeric channels rather than spatial vector archetypes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64
from scipy.spatial.transform import Rotation, Slerp

from arkitscenes_download.ingest.clock import path_timestamps

ORIENTATION_MAX_SPREAD_RAD: float = float(np.deg2rad(20.0))
ORIENTATION_MIN_BOUNDARY_DISTANCE_RAD: float = float(np.deg2rad(10.0))


@dataclass(frozen=True, slots=True)
class Trajectory:
    """World-from-rig poses on the uptime clock."""

    timestamps: Float64[np.ndarray, "n"]
    """ARKit uptime timestamps."""
    translation: Float64[np.ndarray, "n 3"]
    """World-frame rig translations in metres."""
    quaternion_xyzw: Float64[np.ndarray, "n 4"]
    """World-from-rig rotations in xyzw order."""


@dataclass(frozen=True, slots=True)
class Intrinsics:
    """Timestamped pinhole calibrations."""

    timestamps: Float64[np.ndarray, "n"]
    """ARKit uptime timestamps."""
    matrices: Float64[np.ndarray, "n 3 3"]
    """Image-from-camera matrices."""


@dataclass(frozen=True, slots=True)
class DeviceFrameFit:
    """Fixed original-camera-from-CoreMotion-device rotation fit."""

    quaternion_xyzw: Float64[np.ndarray, "4"]
    """Fitted rotation in xyzw order."""
    rms_residual_rad: float
    """Root-mean-square angular gravity residual."""


def load_trajectory(path: Path) -> Trajectory:
    """Read camera-from-world poses and invert them to world-from-camera."""
    rows: Float64[np.ndarray, "n 7"] = np.loadtxt(path, dtype=np.float64)
    camera_from_world: Rotation = Rotation.from_rotvec(rows[:, 1:4])
    world_from_camera: Rotation = camera_from_world.inv()
    translation: Float64[np.ndarray, "n 3"] = -world_from_camera.apply(rows[:, 4:7])
    return Trajectory(rows[:, 0], translation, world_from_camera.as_quat())


def _interpolate_pose(trajectory: Trajectory, query: Float64[np.ndarray, "m"]) -> tuple[Float64[np.ndarray, "m 3"], Rotation]:
    """Interpolate translations and rotations at query timestamps."""
    translations: Float64[np.ndarray, "m 3"] = np.column_stack(
        [np.interp(query, trajectory.timestamps, trajectory.translation[:, axis]) for axis in range(3)]
    )
    rotations: Rotation = Slerp(trajectory.timestamps, Rotation.from_quat(trajectory.quaternion_xyzw))(query)
    return translations, rotations


def resample_trajectory(trajectory: Trajectory, timestamps: Float64[np.ndarray, "m"]) -> Trajectory:
    """Interpolate onto every requested timestamp, holding endpoint poses."""
    selected: Float64[np.ndarray, "m"] = np.clip(timestamps, trajectory.timestamps[0], trajectory.timestamps[-1])
    translations: Float64[np.ndarray, "m 3"]
    rotations: Rotation
    translations, rotations = _interpolate_pose(trajectory, selected)
    return Trajectory(timestamps, translations, rotations.as_quat())


def sky_angles(world_from_camera_xyzw: Float64[np.ndarray, "n 4"]) -> Float64[np.ndarray, "n"]:
    """Compute clockwise image angles from image-up to projected world +Z."""
    camera_from_world: Rotation = Rotation.from_quat(world_from_camera_xyzw).inv()
    sky_xyz: Float64[np.ndarray, "n 3"] = camera_from_world.apply(np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    return np.arctan2(sky_xyz[:, 0], -sky_xyz[:, 1])


def measured_orientation_quarter_turns(angles_rad: Float64[np.ndarray, "n"]) -> int:
    """Round the circular mean gravity angle to counter-clockwise turns."""
    angle: float = float(np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad))))
    return int(round(angle / (np.pi / 2.0))) % 4


def orientation_ambiguity(angles_rad: Float64[np.ndarray, "n"]) -> tuple[bool, float, float]:
    """Return ambiguity, circular spread, and distance to a 45-degree boundary."""
    sin_mean: float = float(np.mean(np.sin(angles_rad)))
    cos_mean: float = float(np.mean(np.cos(angles_rad)))
    resultant: float = float(np.hypot(sin_mean, cos_mean))
    spread: float = float(np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12)))))
    angle: float = float(np.arctan2(sin_mean, cos_mean))
    boundary_distance: float = abs((angle % (np.pi / 2.0)) - np.pi / 4.0)
    ambiguous: bool = bool(spread > ORIENTATION_MAX_SPREAD_RAD or boundary_distance < ORIENTATION_MIN_BOUNDARY_DISTANCE_RAD)
    return ambiguous, spread, boundary_distance


def fit_original_camera_from_device(
    trajectory: Trajectory, accel_timestamps: Float64[np.ndarray, "m"], accel_ms2: Float64[np.ndarray, "m 3"]
) -> DeviceFrameFit:
    """Fit the fixed device-to-original-camera rotation from gravity vectors."""
    query: Float64[np.ndarray, "m"] = np.clip(accel_timestamps, trajectory.timestamps[0], trajectory.timestamps[-1])
    rotations: Rotation
    _, rotations = _interpolate_pose(trajectory, query)
    gravity_camera: Float64[np.ndarray, "m 3"] = rotations.inv().apply(np.asarray([0.0, 0.0, -1.0], dtype=np.float64))
    gravity_device: Float64[np.ndarray, "m 3"] = -accel_ms2 / np.linalg.norm(accel_ms2, axis=1, keepdims=True)
    fit_result = Rotation.align_vectors(gravity_camera, gravity_device)
    fit: Rotation = fit_result[0]
    predicted: Float64[np.ndarray, "m 3"] = fit.apply(gravity_device)
    dots: Float64[np.ndarray, "m"] = np.clip(np.sum(predicted * gravity_camera, axis=1), -1.0, 1.0)
    residual: float = float(np.sqrt(np.mean(np.square(np.arccos(dots)))))
    return DeviceFrameFit(fit.as_quat(), residual)


def rotate_pixels(pixels_xy: Float64[np.ndarray, "... 2"], resolution_wh: tuple[int, int], quarter_turns: int) -> Float64[np.ndarray, "... 2"]:
    """Map continuous pixel centers through ``np.rot90`` semantics.

    Args:
        pixels_xy: Float64 pixel coordinates with shape ``(..., 2)``.
        resolution_wh: Original image width and height.
        quarter_turns: Counter-clockwise quarter turns, reduced modulo four.
    """
    width, height = resolution_wh
    pixels: Float64[np.ndarray, "... 2"] = np.asarray(pixels_xy, dtype=np.float64)
    x: Float64[np.ndarray, "..."] = pixels[..., 0]
    y: Float64[np.ndarray, "..."] = pixels[..., 1]
    match quarter_turns % 4:
        case 0:
            return pixels.copy()
        case 1:
            return np.stack((y, (width - 1.0) - x), axis=-1)
        case 2:
            return np.stack(((width - 1.0) - x, (height - 1.0) - y), axis=-1)
        case 3:
            return np.stack(((height - 1.0) - y, x), axis=-1)
    raise AssertionError("unreachable")


def rotate_resolution(resolution_wh: tuple[int, int], quarter_turns: int) -> tuple[int, int]:
    """Return image dimensions after counter-clockwise quarter turns."""
    return resolution_wh if quarter_turns % 2 == 0 else (resolution_wh[1], resolution_wh[0])


def bake_camera_orientation(camera_from_world: Rotation, quarter_turns: int) -> Rotation:
    """Rotate an OpenCV RDF camera frame consistently with ``np.rot90`` pixels."""
    optical_rotation: Rotation = Rotation.from_euler("z", -quarter_turns * np.pi / 2.0)
    return optical_rotation * camera_from_world


def rotate_trajectory(trajectory: Trajectory, quarter_turns: int) -> Trajectory:
    """Bake pixel rotation into world-from-camera poses without moving centers."""
    world_from_camera: Rotation = Rotation.from_quat(trajectory.quaternion_xyzw)
    camera_from_world_baked: Rotation = bake_camera_orientation(world_from_camera.inv(), quarter_turns)
    return Trajectory(trajectory.timestamps, trajectory.translation, camera_from_world_baked.inv().as_quat())


def rotate_intrinsics_sequence(intrinsics: Intrinsics, resolution_wh: tuple[int, int], quarter_turns: int) -> tuple[Intrinsics, tuple[int, int]]:
    """Bake the same pixel rotation into every timestamped calibration."""
    centers: Float64[np.ndarray, "n 2"] = rotate_pixels(intrinsics.matrices[:, :2, 2], resolution_wh, quarter_turns)
    focal_lengths: Float64[np.ndarray, "n 2"] = intrinsics.matrices[:, (0, 1), (0, 1)]
    if quarter_turns % 2:
        focal_lengths = focal_lengths[:, ::-1]
    resolution: tuple[int, int] = rotate_resolution(resolution_wh, quarter_turns)
    matrices: Float64[np.ndarray, "n 3 3"] = np.zeros_like(intrinsics.matrices, dtype=np.float64)
    matrices[:, 0, 0] = focal_lengths[:, 0]
    matrices[:, 1, 1] = focal_lengths[:, 1]
    matrices[:, :2, 2] = centers
    matrices[:, 2, 2] = 1.0
    return Intrinsics(intrinsics.timestamps, matrices), resolution


def interpolate_sky_angles(trajectory: Trajectory, timestamps: Float64[np.ndarray, "m"]) -> Float64[np.ndarray, "m"]:
    """Interpolate sky orientation at every requested time, clamping endpoints."""
    query_times: Float64[np.ndarray, "m"] = np.clip(timestamps, trajectory.timestamps[0], trajectory.timestamps[-1])
    rotations: Rotation
    _, rotations = _interpolate_pose(trajectory, query_times)
    return sky_angles(rotations.as_quat())


def load_intrinsics(paths: list[Path]) -> Intrinsics:
    """Read pincam files into timestamped calibration matrices."""
    matrices: list[np.ndarray] = []
    for path in paths:
        values: Float64[np.ndarray, "6"] = np.asarray([float(value) for value in path.read_text().split()], dtype=np.float64)
        matrix: Float64[np.ndarray, "3 3"] = np.asarray([[values[2], 0.0, values[4]], [0.0, values[3], values[5]], [0.0, 0.0, 1.0]], dtype=np.float64)
        matrices.append(matrix)
    timestamps_array: Float64[np.ndarray, "n"] = path_timestamps(paths)
    order: np.ndarray = np.argsort(timestamps_array)
    return Intrinsics(timestamps_array[order], np.asarray(matrices, dtype=np.float64)[order])


def scale_intrinsics(intrinsics: Intrinsics, scale: float) -> Intrinsics:
    """Scale focal lengths and principal points without changing homogeneous K."""
    matrices: Float64[np.ndarray, "n 3 3"] = intrinsics.matrices.copy()
    matrices[:, :2, :] *= scale
    return Intrinsics(intrinsics.timestamps, matrices)
