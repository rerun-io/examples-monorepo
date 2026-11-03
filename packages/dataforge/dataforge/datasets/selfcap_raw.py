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

from dataforge.logging_toolkit import ImuChannel
from dataforge.oak_mount import OAK_MOUNT_OTHER as OAK_MOUNT_OTHER
from dataforge.oak_mount import OAK_MOUNTS as OAK_MOUNTS
from dataforge.oak_mount import Distortion as Distortion
from dataforge.oak_mount import FactoryCalibration as FactoryCalibration
from dataforge.oak_mount import FactoryExtrinsic as FactoryExtrinsic
from dataforge.oak_mount import Intrinsic as Intrinsic
from dataforge.oak_mount import Lens as Lens
from dataforge.oak_mount import Sensor as Sensor
from dataforge.oak_mount import Size as Size
from dataforge.oak_mount import factory_transform as factory_transform
from dataforge.oak_mount import nominal_oak_mount as nominal_oak_mount
from dataforge.oak_mount import quest_left_T_right as quest_left_T_right
from dataforge.oak_mount import quest_mesh_frame as quest_mesh_frame
from dataforge.quest_joint_poses import QUEST_HAND_SIDES, QuestJointPoses, read_quest_joint_poses
from dataforge.records import read_csv_columns, read_json

BROWN_CONRADY_COEFFICIENTS: int = 14
"""OpenCV's 14-coefficient Brown-Conrady distortion layout."""

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
    available: bool
    """Whether the lens values were measured. The iPhones write ``False`` with a 700 px fallback focal."""



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



COCO133_HAND_BASE: dict[str, int] = {"left": 91, "right": 112}
"""First COCO-133 keypoint id of each hand (21 joints each)."""
COCO133_FROM_OPENXR: list[int] = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]
"""The 21 of 26 OpenXR joints COCO-133 keeps, in COCO order: wrist, then each finger without its metacarpal (palm dropped)."""


def read_quest_joint_hands(quest_dir: Path) -> dict[str, QuestJointPoses]:
    """Both hands' 26-joint CSVs as recorded; an unreadable file is warned about and leaves that hand out."""
    hands: dict[str, QuestJointPoses] = {}
    for side in QUEST_HAND_SIDES:
        source: Path = quest_dir / f"{side}_hand_poses.csv"
        if not os.access(source, os.R_OK):
            print(f"  warning: unreadable Quest hands {source}")
            continue
        hands[side] = read_quest_joint_poses(source)
    return hands


def coco133_hands(hands: dict[str, QuestJointPoses]) -> QuestHands:
    """The COCO-133 view of the Quest hands: 21 joints per hand on the union of both hands' stamps, NaN elsewhere."""
    times_ns: Int64[ndarray, "n_samples"] = (
        np.unique(np.concatenate([hand.times_ns for hand in hands.values()])) if hands else np.empty(0, dtype=np.int64)
    )
    positions: Float32[ndarray, "n_samples 133 3"] = np.full((len(times_ns), 133, 3), np.nan, dtype=np.float32)
    for side, hand in hands.items():
        base = COCO133_HAND_BASE[side]
        positions[np.searchsorted(times_ns, hand.times_ns), base : base + 21] = hand.positions[:, COCO133_FROM_OPENXR]
    return QuestHands(times_ns=times_ns, positions_xyz=positions)


def read_quest_hands(quest_dir: Path) -> QuestHands:
    """The COCO-133 view straight from the CSVs, for bases written before the joint poses were ingested."""
    return coco133_hands(read_quest_joint_hands(quest_dir))


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



def crop_shift_px(intrinsic: Intrinsic) -> float:
    """Rows cut from the top when a camera records a centred, same-width crop of its sensor.

    The Quest sensors are 1280x1280 and the videos 1280x960, so the factory ``cy`` is 160 px too low for the video.
    """
    sensor, capture = intrinsic.sensor_info.pixel_array_size, intrinsic.capture_resolution
    return (sensor.height - capture.height) / 2.0 if sensor.width == capture.width and sensor.height > capture.height else 0.0


def read_calibration(calibration_path: Path) -> dict[str, CameraCalibration]:
    """Read a device ``calibration.json`` into intrinsics keyed by ``positional_layout``, with ``cy`` in video pixels.

    Extrinsics are read separately (``oak_stereo``, ``quest_left_T_right``); exo files ship none.

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
            principal_point_xy=np.asarray([lens.principal_point_x, lens.principal_point_y - crop_shift_px(entry)], dtype=np.float64),
            distortion_coefficients=distortion,
            available=lens.available,
        )
    return cameras


def assumed_phone_lens(width: int, height: int) -> CameraCalibration:
    """An iPhone's lens until calibration fits one: a centred pinhole with no distortion and a 90 deg horizontal field of view
    (focal = width / 2). The phones record no lens values, so this is an assumption, tagged as one."""
    return CameraCalibration(
        width=width, height=height, focal_length_xy=np.array([width / 2, width / 2]), principal_point_xy=np.array([width / 2, height / 2]),
        distortion_coefficients=None, available=False,
    )


def oak_stereo(calibration: FactoryCalibration) -> dict[str, Float64[ndarray, "4 4"]]:
    """RGB-relative OAK stereo transforms (``rgb_T_left``, ``rgb_T_right``); the factory file is in centimetres."""
    ext = {e.type: e for e in calibration.extrinsics}
    if not {"left_to_right", "left_to_rgb"} <= ext.keys():
        return {}
    right, rgb = ext["left_to_right"], ext["left_to_rgb"]
    rgb_T_left = np.linalg.inv(factory_transform(rgb.R, rgb.T * 0.01))
    return {"rgb_T_left": rgb_T_left, "rgb_T_right": rgb_T_left @ factory_transform(right.R, right.T * 0.01)}






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
