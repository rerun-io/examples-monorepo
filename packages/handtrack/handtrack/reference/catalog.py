"""A6's catalog adapter, preserving float64 transforms and upstream millimetres.

This is the explicit unit boundary of the reference backend; handtrack's shared
pipeline still uses metres. Lens coefficients come from the lens component,
never the nonexistent source k5/k6 columns.
"""
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch

from handtrack.data.catalog import (
    HAND_ROOT,
    PROFILE_COLUMN,
    SIDE_NAMES,
    TIMELINE,
    UMETRACK_LAYOUT,
    bool_rows,
    static_floats,
    static_number,
    static_text,
)
from handtrack.hand.pose import hand_model_from_profile


@dataclass(frozen=True, slots=True)
class CameraSpec:
    width: int
    """Native pixel width."""
    height: int
    """Native pixel height."""
    focal: tuple[float, float]
    """Horizontal and vertical focal lengths."""
    principal: tuple[float, float]
    """Principal point in native pixels."""
    coefficients: list[float]
    """Upstream order: k1..k4, p1, p2, k5, k6."""
    angle: float
    """Source camera roll in degrees."""


@dataclass(frozen=True, slots=True)
class ReferenceLabels:
    cameras: list[CameraSpec]
    """Static camera calibration in cam_00..03 order."""
    camera_to_world: Float64[ndarray, "f 4 4 4"]
    """World from camera, millimetres; zeros on untracked frames."""
    wrists: Float64[ndarray, "f 2 4 4"]
    """World from wrist, millimetres, without right-hand mirroring."""
    joints: Float64[ndarray, "f 2 22"]
    """Joint angles, radians; zeros for missing poses."""
    confidence: Float64[ndarray, "f 2"]
    """Confidence, forced to zero on untracked headset frames."""
    tracked: Bool[ndarray, "f"]
    """Finite, tracked headset pose."""
    times: Int64[ndarray, "f"]
    """Strictly increasing video_time nanoseconds."""
    hand_model: HandModelTorch
    """Profile parsed through the shared typed pyserde reader."""


def _rows(table: pa.Table, column: str, width: int) -> Float64[ndarray, "f w"]:
    """A6 float64 values (the shared training reader intentionally uses float32)."""
    result: Float64[ndarray, "f w"] = np.full((table.num_rows, width), np.nan, dtype=np.float64)
    for row, values in enumerate(table[column].to_pylist()):
        if values:
            result[row] = values[0]
    return result


def _poses(table: pa.Table, prefix: str) -> Float64[ndarray, "f 4 4"]:
    quaternion: Float64[ndarray, "f 4"] = _rows(table, f"{prefix}:Transform3D:quaternion", 4)
    translation: Float64[ndarray, "f 3"] = _rows(table, f"{prefix}:Transform3D:translation", 3)
    valid: Bool[ndarray, "f"] = np.isfinite(quaternion).all(axis=1) & (np.linalg.norm(quaternion, axis=1) > 0)
    poses: Float64[ndarray, "f 4 4"] = np.zeros((table.num_rows, 4, 4), dtype=np.float64)
    poses[:, :3, :3] = np.nan
    poses[valid, :3, :3] = Rotation.from_quat(quaternion[valid]).as_matrix()
    poses[:, :3, 3] = translation * 1000.0
    poses[:, 3, 3] = 1.0
    return poses


def reference_labels(statics: pa.Table, table: pa.Table) -> ReferenceLabels:
    """Convert static and timeline Arrow tables with A6's validity and camera conventions."""
    cameras: list[CameraSpec] = []
    extrinsics: list[Float64[ndarray, "4 4"]] = []
    for camera in UMETRACK_LAYOUT.cameras:
        if static_number(statics, f"{camera}:Transform3D:relation", camera) != 2:
            raise ValueError(f"{camera}: expected ChildFromParent")
        transform: Float64[ndarray, "4 4"] = np.eye(4)
        transform[:3, :3] = np.asarray(static_floats(statics, f"{camera}:Transform3D:mat3x3", camera)).reshape(3, 3).T
        transform[:3, 3] = np.asarray(static_floats(statics, f"{camera}:Transform3D:translation", camera)) * 1000.0
        extrinsics.append(np.linalg.inv(transform))
        pinhole: str = f"{camera}/pinhole"
        k: Float64[ndarray, "3 3"] = np.asarray(static_floats(statics, f"{pinhole}:Pinhole:image_from_camera", camera)).reshape(3, 3).T
        resolution: list[float] = static_floats(statics, f"{pinhole}:Pinhole:resolution", camera)
        lens: list[float] = static_floats(statics, f"{pinhole}:simplecv.components.DistortionCoefficients", camera)
        if len(lens) != 8 or static_text(statics, f"{pinhole}:simplecv.components.DistortionModel", camera) != "kannala_brandt":
            raise ValueError(f"{camera}: expected eight-coefficient Fisheye62")
        coefficients: list[float] = [lens[i] for i in (0, 1, 2, 3, 6, 7, 4, 5)]
        for index, name in enumerate(("k1", "k2", "k3", "k4", "p1", "p2")):
            if static_number(statics, f"{pinhole}:{name}", camera) != coefficients[index]:
                raise ValueError(f"{camera}: lens {name} disagrees with source")
        cameras.append(CameraSpec(int(resolution[0]), int(resolution[1]), (float(k[0, 0]), float(k[1, 1])),
                                  (float(k[0, 2]), float(k[1, 2])), coefficients, static_number(statics, f"{pinhole}:source_camera_angle_deg", camera)))
    times: Int64[ndarray, "f"] = np.asarray(table[TIMELINE].combine_chunks().to_numpy(zero_copy_only=False)).view(np.int64)
    if not len(times) or not np.all(np.diff(times) > 0):
        raise ValueError("Expected nonempty strictly increasing video_time")
    rig: Float64[ndarray, "f 4 4"] = _poses(table, UMETRACK_LAYOUT.rig)
    tracked: Bool[ndarray, "f"] = np.isfinite(rig).all(axis=(1, 2)) & ~bool_rows(table[f"{UMETRACK_LAYOUT.rig}:untracked"])[0]
    transforms: Float64[ndarray, "f 4 4 4"] = np.zeros((len(times), 4, 4, 4), dtype=np.float64)
    transforms[tracked] = rig[tracked, None] @ np.stack(extrinsics)[None]
    confidence: Float64[ndarray, "f 2"] = np.zeros((len(times), 2), dtype=np.float64)
    joints: Float64[ndarray, "f 2 22"] = np.zeros((len(times), 2, 22), dtype=np.float64)
    wrists: Float64[ndarray, "f 2 4 4"] = np.zeros((len(times), 2, 4, 4), dtype=np.float64)
    for hand, side in enumerate(SIDE_NAMES):
        prefix: str = f"{HAND_ROOT}/{side}"
        angles: Float64[ndarray, "f 22"] = _rows(table, f"{prefix}/joint_angles:joint_angles", 22)
        wrist: Float64[ndarray, "f 4 4"] = _poses(table, f"{prefix}/wrist")
        present: Bool[ndarray, "f"] = np.isfinite(angles).all(axis=1) & np.isfinite(wrist).all(axis=(1, 2))
        confidence[:, hand] = np.nan_to_num(_rows(table, f"{prefix}/confidence:Scalars:scalars", 1)[:, 0], nan=0.0)
        if np.any((confidence[:, hand] > 0) & ~present):
            raise ValueError(f"{side}: positive confidence without a finite pose")
        confidence[~tracked, hand] = 0.0
        joints[present, hand] = angles[present]
        wrists[present, hand] = wrist[present]
    return ReferenceLabels(cameras, transforms, wrists, joints, confidence, tracked, times,
                           hand_model_from_profile(static_text(statics, PROFILE_COLUMN, "hand profile")))
