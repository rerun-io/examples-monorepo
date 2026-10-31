"""SelfCap raw CSV, video-clock, calibration, and session metadata readers.

These readers preserve the ingest boundary; planning and logging live in selfcap.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from serde import serde

from dataforge.logging_toolkit import ImuChannel
from dataforge.quest_joint_poses import read_quest_joint_poses
from dataforge.records import read_csv_columns, read_json

BROWN_CONRADY_COEFFICIENTS: int = 14
"""OpenCV's 14-coefficient Brown-Conrady distortion layout."""

@serde
@dataclass(frozen=True, slots=True)
class Size:
    width: int
    """Array/capture width in pixels."""
    height: int
    """Array/capture height in pixels."""


@serde
@dataclass(frozen=True, slots=True)
class Sensor:
    pixel_array_size: Size
    """Full sensor array before centre crop."""


@serde
@dataclass(frozen=True, slots=True)
class Lens:
    focal_length_x: float | None
    """Horizontal focal in pixels."""
    focal_length_y: float | None
    """Vertical focal in pixels."""
    principal_point_x: float | None
    """Horizontal principal point in sensor pixels."""
    principal_point_y: float | None
    """Vertical principal point in sensor pixels."""
    available: bool
    """Whether factory intrinsics are valid."""


@serde
@dataclass(frozen=True, slots=True)
class Distortion:
    coefficients: tuple[float, ...] | None = None
    """OAK Brown-Conrady vector; absent for Quest."""


@serde
@dataclass(frozen=True, slots=True)
class Intrinsic:
    camera_id: str
    """Physical camera id, including Quest 50/51."""
    positional_layout: str
    """Factory slot name."""
    lens_intrinsics: Lens
    """Raw factory lens values."""
    sensor_info: Sensor
    """Sensor crop metadata."""
    capture_resolution: Size
    """Captured image dimensions."""
    distortion: Distortion
    """Full OAK distortion; Quest has none."""


@serde
@dataclass(frozen=True, slots=True)
class FactoryExtrinsic:
    type: str
    """Factory direction label; matrices follow scratch's device-specific convention."""
    R: Float64[ndarray, "3 3"]
    """Raw factory rotation."""
    T: Float64[ndarray, "3"]
    """Translation: OAK centimetres, Quest metres."""
    to: str = ""
    """Physical destination camera id."""


@serde
@dataclass(frozen=True, slots=True)
class FactoryCalibration:
    device_id: str
    """Full factory device identity."""
    intrinsics: tuple[Intrinsic, ...]
    """Available and unavailable camera slots."""
    extrinsics: tuple[FactoryExtrinsic, ...] = ()
    """Factory transforms, never estimated session calibration."""


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """One camera's intrinsics as the raw calibration states them."""

    width: int
    """Image width in pixels, from ``capture_resolution`` (the encoded video's width)."""
    height: int
    """Image height in pixels, from ``capture_resolution``."""
    focal_length_xy: Float64[ndarray, "2"]
    """``(fx, fy)`` in pixels."""
    principal_point_xy: Float64[ndarray, "2"]
    """``(cx, cy)`` in pixels."""
    distortion_coefficients: Float64[ndarray, "14"] | None
    """Brown-Conrady coefficients, or None when the raw calibration declares none."""



@dataclass(frozen=True, slots=True)
class HeadPoses:
    """The Quest's left-eye track in its own Y-up world frame."""

    times_ns: Int64[ndarray, "n_samples"]
    """Sample times on the episode-relative ``video_time`` clock, in nanoseconds."""
    translations_xyz: Float32[ndarray, "n_samples 3"]
    """Left-eye positions in metres."""
    quaternions_xyzw: Float32[ndarray, "n_samples 4"]
    """Left-eye orientations as xyzw quaternions (Rerun's component layout)."""



@dataclass(frozen=True, slots=True)
class QuestHands:
    """Raw Quest positions mapped into COCO-133, without inferred confidence."""

    times_ns: Int64[ndarray, "n_samples"]
    """Unmodified episode-relative hand timestamps in nanoseconds."""
    positions_xyz: Float32[ndarray, "n_samples 133 3"]
    """World positions in metres; absent hand samples and non-hand joints are NaN."""

    def __post_init__(self) -> None:
        if len(self.times_ns) != len(self.positions_xyz):
            raise ValueError("hand positions and timestamps must have equal lengths")



def read_quest_hands(quest_dir: Path) -> QuestHands:
    """Read 26-joint CSVs (xyz or xyz+xyzw per joint) on their raw clock.

    The upstream CSV layout is timestamp followed by 26 joint records in Quest
    order. An all-zero row is the CSV's "hand not tracked" and becomes NaN; no jump,
    freeze, or view-cone filtering is applied at ingest (that is curation).
    Missing files are warned about and leave that hand absent.
    """
    tracks: list[tuple[int, Int64[ndarray, "n"], Float32[ndarray, "n 21 3"]]] = []
    mapping: list[int] = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]
    for side, base in (("left", 91), ("right", 112)):
        source: Path = quest_dir / f"{side}_hand_poses.csv"
        if not os.access(source, os.R_OK):
            print(f"  warning: unreadable Quest hands {source}")
            continue
        raw = read_quest_joint_poses(source)
        times = raw.times_ns
        points = raw.positions[:, mapping]
        tracks.append((base, times, points))
    times_ns: Int64[ndarray, "n_samples"] = np.unique(np.concatenate([track[1] for track in tracks])) if tracks else np.empty(0, dtype=np.int64)
    positions: Float32[ndarray, "n_samples 133 3"] = np.full((len(times_ns), 133, 3), np.nan, dtype=np.float32)
    for base, stamps, points in tracks:
        positions[np.searchsorted(times_ns, stamps), base : base + 21] = points
    return QuestHands(times_ns=times_ns, positions_xyz=positions)



def read_imu_csv(csv_path: Path) -> tuple[ImuChannel, ImuChannel]:
    """Read the OAK IMU csv, keeping only the gyroscope and accelerometer channels.

    Both channels share the csv's single ``ts_ns`` column: the OAK reports one
    fused row per sample instant, unlike RoboCap's two independent tables.

    Args:
        csv_path: An ``ego/imu.csv``; also holds ``mag_*`` and ``rot_*`` columns.

    Returns:
        The gyro (rad/s) and accel (m/s^2) channels at their native ~100 Hz rate.
        TODO(dataforge): the magnetometer and the ``rot_ijk`` device attitude are
        dropped; they want a magnetometer entity and a proper rotation component.
    """
    names = ("gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z")
    columns: pa.Table = read_csv_columns(csv_path, dict.fromkeys(("ts_ns", *names), pa.float64()))
    times_ns: Int64[ndarray, "n_samples"] = np.rint(columns.column("ts_ns").to_numpy()).astype(np.int64)
    gyro_xyz: Float64[ndarray, "n_samples 3"] = np.column_stack([columns.column(name).to_numpy() for name in names[:3]])
    accel_xyz: Float64[ndarray, "n_samples 3"] = np.column_stack([columns.column(name).to_numpy() for name in names[3:]])
    return ImuChannel(times_ns=times_ns, values_xyz=gyro_xyz), ImuChannel(times_ns=times_ns, values_xyz=accel_xyz)



def read_head_poses(csv_path: Path) -> HeadPoses:
    """Read the Quest head-pose csv, keeping the left-eye track.

    Args:
        csv_path: A ``quest/head_pose.csv`` holding both eye poses per row.

    Returns:
        The left-eye poses, which stand in for the Quest rig pose.
        TODO(dataforge): the right eye's offset relative to the left is
        recoverable from ``quest/calibration.json`` (``device_to_left`` /
        ``device_to_right`` / ``left_to_right`` extrinsics); once validated it
        becomes the static ``rig_T_cam`` of both Quest cameras.
    """
    names = ("left_pos_x", "left_pos_y", "left_pos_z", "left_quat_x", "left_quat_y", "left_quat_z", "left_quat_w")
    columns: pa.Table = read_csv_columns(csv_path, dict.fromkeys(("ts_ns", *names), pa.float64()))
    times_ns: Int64[ndarray, "n_samples"] = np.rint(columns.column("ts_ns").to_numpy()).astype(np.int64)
    translations_xyz: Float32[ndarray, "n_samples 3"] = np.column_stack([columns.column(name).to_numpy() for name in names[:3]]).astype(np.float32)
    quaternions_xyzw: Float32[ndarray, "n_samples 4"] = np.column_stack([columns.column(name).to_numpy() for name in names[3:]]).astype(np.float32)
    return HeadPoses(times_ns=times_ns, translations_xyz=translations_xyz, quaternions_xyzw=quaternions_xyzw)



def read_calibration(calibration_path: Path) -> dict[str, CameraCalibration]:
    """Read a device ``calibration.json`` into intrinsics keyed by ``positional_layout``.

    Extrinsics are deliberately not applied here: exo files ship an empty list, and the
    ego/Quest ones are in unvalidated units (the OAK baseline reads as 7.5,
    i.e. centimetres). TODO(dataforge): validate and log them as ``rig_T_cam``.

    Args:
        calibration_path: A ``calibration.json`` from ``ego/``, ``exo/<Device>/``, or ``quest/``.

    Returns:
        ``positional_layout`` → intrinsics for every camera the file describes.
    """
    document: FactoryCalibration = read_json(calibration_path, FactoryCalibration)
    cameras: dict[str, CameraCalibration] = {}
    for entry in document.intrinsics:
        lens = entry.lens_intrinsics
        resolution = entry.capture_resolution
        if lens.focal_length_x is None:
            continue
        raw_coefficients = entry.distortion.coefficients
        distortion: Float64[ndarray, "14"] | None = (
            np.asarray(raw_coefficients, dtype=np.float64) if raw_coefficients is not None and len(raw_coefficients) == BROWN_CONRADY_COEFFICIENTS else None
        )
        cameras[entry.positional_layout] = CameraCalibration(
            width=resolution.width,
            height=resolution.height,
            focal_length_xy=np.asarray([lens.focal_length_x, lens.focal_length_y], dtype=np.float64),
            principal_point_xy=np.asarray([lens.principal_point_x, lens.principal_point_y], dtype=np.float64),
            distortion_coefficients=distortion,
        )
    return cameras



def read_session_task(metadata_path: Path) -> dict[str, str]:
    """Task and collector of a session, or ``{}`` when its metadata is unreadable.

    Every ``cut`` session's ``metadata.json`` is mode 000 as of 2026-08-20, so
    this returns ``{}`` corpus-wide until the pending chmod lands.

    Args:
        metadata_path: Session-level ``metadata.json``.

    Returns:
        The ``task`` and ``collector`` keys that were present and non-empty.
    """
    if not os.access(metadata_path, os.R_OK):
        return {}
    try:
        document: dict = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"  warning: unreadable session metadata {metadata_path}: {error}")
        return {}
    values: dict[str, str] = {}
    for key, source in (("task", "task"), ("collector", "collector_name")):
        value: object = document.get(source)
        if isinstance(value, str) and value:
            values[key] = value
    return values
