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
from dataforge.quest_joint_poses import QUEST_HAND_SIDES, QuestJointPoses, read_quest_joint_poses
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


def factory_transform(rotation: Float64[ndarray, "3 3"], translation: Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """Rigid homogeneous transform from a factory rotation and translation."""
    matrix: Float64[ndarray, "4 4"] = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix


def oak_stereo(calibration: FactoryCalibration) -> dict[str, Float64[ndarray, "4 4"]]:
    """RGB-relative OAK stereo transforms (``rgb_T_left``, ``rgb_T_right``); the factory file is in centimetres."""
    ext = {e.type: e for e in calibration.extrinsics}
    if not {"left_to_right", "left_to_rgb"} <= ext.keys():
        return {}
    right, rgb = ext["left_to_right"], ext["left_to_rgb"]
    rgb_T_left = np.linalg.inv(factory_transform(rgb.R, rgb.T * 0.01))
    return {"rgb_T_left": rgb_T_left, "rgb_T_right": rgb_T_left @ factory_transform(right.R, right.T * 0.01)}


# Prior for the OAK mount on the headset: the RGB camera (RDF) in the Quest left-eye frame, as (healthy, camera-50) pairs.
# The mount belongs to the OAK's bracket, so the prior is keyed by OAK device id. Medians of the fitted mounts in
# selfcap-post-ep1-v2 (2026-10-30, 678 sessions); spread about each prior is p50 0.4-1.0 cm / 0.5-1.5 deg. The camera-50
# column is for headsets whose broken left camera borrows camera 50: their eye track's frame sits about 175 deg about x.
OAK_MOUNT_OTHER: tuple[Float64[ndarray, "4 4"], Float64[ndarray, "4 4"]] = (
    np.array([[0.9996, -0.0131, -0.0230, 0.0509], [0.0212, 0.9161, 0.4004, 0.0408], [0.0158, -0.4008, 0.9160, -0.0513], [0, 0, 0, 1.0]]),
    np.array([[0.9996, -0.0036, -0.0271, 0.0510], [-0.0168, -0.8630, -0.5048, -0.0479], [-0.0216, 0.5051, -0.8628, -0.0223], [0, 0, 0, 1.0]]),
)
"""380 healthy / 59 camera-50 sessions on OAKs C1F3CFD2 and F1FBBCD6, and the fallback for an OAK no fit has seen."""
OAK_MOUNTS: dict[str, tuple[Float64[ndarray, "4 4"], Float64[ndarray, "4 4"]]] = {
    "OAK - 14442C10D1FCD7D200": (  # 214 healthy / 25 camera-50 sessions; sits about 4 cm from the others, tilted 7 deg more
        np.array([[0.9989, -0.0328, 0.0346, 0.0306], [0.0108, 0.8627, 0.5056, 0.0671], [-0.0464, -0.5047, 0.8621, -0.0144], [0, 0, 0, 1.0]]),
        np.array([[0.9995, -0.0069, 0.0324, 0.0333], [0.0123, -0.8297, -0.5581, -0.0728], [0.0307, 0.5581, -0.8292, -0.0414], [0, 0, 0, 1.0]]),
    ),
}


def nominal_oak_mount(oak: FactoryCalibration, quest: FactoryCalibration) -> Float64[ndarray, "4 4"] | None:
    """Prior Quest-left-eye → OAK-RGB mount for this OAK and headset layout, or None for a layout no fit has seen."""
    healthy, camera_50 = OAK_MOUNTS.get(oak.device_id, OAK_MOUNT_OTHER)
    if quest_left_T_right(quest) is not None:
        return healthy
    if any(c.camera_id == "50" for c in quest.intrinsics):
        return camera_50
    return None


def quest_mesh_frame(oak: FactoryCalibration, quest: FactoryCalibration) -> Float64[ndarray, "4 4"] | None:
    """Where the Quest mesh (built in a healthy left-eye frame) sits in this headset's eye-track frame.

    Identity for a healthy headset. A camera-50 headset's eye frame is turned about 175 deg; both OAK priors describe the
    same physical mount, so ``camera50_T_healthy = camera_50 @ inv(healthy)``. None for a layout no fit has seen.
    """
    healthy, camera_50 = OAK_MOUNTS.get(oak.device_id, OAK_MOUNT_OTHER)
    if quest_left_T_right(quest) is not None:
        return np.eye(4)
    if any(c.camera_id == "50" for c in quest.intrinsics):
        return camera_50 @ np.linalg.inv(healthy)
    return None


def quest_left_T_right(calibration: FactoryCalibration) -> Float64[ndarray, "4 4"] | None:
    """The right camera in the left camera's frame, when the headset is healthy: both cameras measured and both
    ``device_to_*`` extrinsics present. Faulty headsets return None; their fix borrows from another session."""
    ext = {e.type: e for e in calibration.extrinsics}
    measured = {c.positional_layout for c in calibration.intrinsics if c.lens_intrinsics.available}
    if not ({"device_to_left", "device_to_right"} <= ext.keys() and {"left", "right"} <= measured):
        return None
    left, right = ext["device_to_left"], ext["device_to_right"]
    return np.linalg.inv(factory_transform(left.R.T, left.T)) @ factory_transform(right.R.T, right.T)



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
