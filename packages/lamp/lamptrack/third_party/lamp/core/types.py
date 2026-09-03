# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared data types for the LAMP pipeline."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray

# Golden-ratio hue step for deterministic, visually distinct track colors.
_GOLDEN_RATIO_CONJ: float = 0.6180339887498949


def color_from_id(person_id: int) -> tuple[float, float, float, float]:
    """Deterministic, visually-distinct RGBA color for a track id."""
    hue = (person_id * _GOLDEN_RATIO_CONJ) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (float(r), float(g), float(b), 1.0)


@dataclass(slots=True)
class Detection2D:
    """A 2D person detection (bbox + keypoints) for one camera at one timestamp."""

    box_xyxy: Float32[ndarray, "4"]  # [xmin, ymin, xmax, ymax]
    box_score: float  # 0..1
    keypoints: Float32[ndarray, "17 3"]  # COCO order, [x, y, score]
    cam_idx: int
    timestamp_ns: int
    has_keypoints: bool = True  # False if keypoints are placeholder zeros
    det_crop: UInt8[ndarray, "h w 3"] | None = None  # resized model crop
    det_crop_raw: UInt8[ndarray, "h w 3"] | None = None  # original-resolution crop
    track_id: int | None = None  # filled in by LampTracker after assignment

    def iou(self, other: Detection2D) -> float:
        """Axis-aligned bounding-box IoU."""
        return box_iou_xyxy(self.box_xyxy, other.box_xyxy)

    def giou(self, other: Detection2D) -> float:
        """Generalized IoU, normalized to [0, 1]."""
        a, b = self.box_xyxy, other.box_xyxy
        inter = _box_intersection_area(a, b)
        union = _box_area(a) + _box_area(b) - inter
        encl = _box_enclosing_area(a, b)
        normal_iou = inter / max(union, 1e-6)
        return float((normal_iou - (encl - union) / max(encl, 1e-6) + 1.0) / 2.0)


def _box_area(box: Float32[ndarray, "4"]) -> float:
    """Area of an xyxy box; 0 for degenerate (max < min) boxes."""
    w = max(0.0, float(box[2] - box[0]))
    h = max(0.0, float(box[3] - box[1]))
    return w * h


def _box_intersection_area(a: Float32[ndarray, "4"], b: Float32[ndarray, "4"]) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_enclosing_area(a: Float32[ndarray, "4"], b: Float32[ndarray, "4"]) -> float:
    """Area of the smallest axis-aligned box containing both `a` and `b`."""
    x1, y1 = min(float(a[0]), float(b[0])), min(float(a[1]), float(b[1]))
    x2, y2 = max(float(a[2]), float(b[2])), max(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou_xyxy(a: Float32[ndarray, "4"], b: Float32[ndarray, "4"]) -> float:
    """Axis-aligned bounding-box IoU on raw `(4,)` xyxy arrays."""
    inter = _box_intersection_area(a, b)
    if inter == 0.0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return float(inter / max(union, 1e-6))


@dataclass(slots=True)
class Skeleton:
    """A SMPL 3D person pose in the world frame."""

    kp_world: Float32[ndarray, "24 3"]
    kp_score: Float32[ndarray, "24"]
    T_world_pelvis: Float32[ndarray, "4 4"]
    shape: Float32[ndarray, "betas"] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    joints_rot_mat: Float32[ndarray, "joints 3 3"] = field(
        default_factory=lambda: np.zeros((0, 3, 3), dtype=np.float32)
    )
    verts_w: Float32[ndarray, "vertices 3"] = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )


@dataclass(slots=True)
class PersonState:
    """Per-timestamp record for a tracked person."""

    detection2ds: list[Detection2D]
    skeleton: Skeleton | None = None
    num_fuses: int = 1  # how many obs were fused into this state


@dataclass
class Person:
    """A persistent person track across frames."""

    id: int
    color: tuple[float, float, float, float] | None = None
    ts_to_states: dict[int, PersonState] = field(default_factory=dict[int, PersonState])
    last_obs_ts: int = -1
    last_lifted_ts: int = -1
    active: bool = True
    inactive_ts: int = -1
    uncertainty: float = -1.0
    num_obs_3d: float = 0.0
    shape_estimate: Float32[ndarray, "betas"] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    shape_num_updates: int = 0
    shape_locked: bool = False

    def __post_init__(self) -> None:
        """Assign a deterministic HSV-from-id color when `color is None`."""
        if self.color is None:
            self.color = color_from_id(self.id)


@dataclass(slots=True)
class LampResult:
    """The pipeline's per-frame output: every tracked person at a timestamp."""

    timestamp_ns: int
    people: dict[int, Person]


@dataclass(slots=True)
class Frameset:
    """A multi-camera capture at a single timestamp."""

    timestamp_ns: int  # capture timestamp
    images: dict[int, UInt8[ndarray, "h w channels"]]


class CameraOrientation(IntEnum):
    """Per-camera image orientation hint passed to the detector."""

    UPRIGHT = 0
    ROTATE_90_CW = 1  # Aria SLAM cameras


# SMPL 24-joint topology — `(parent_idx, child_idx)` pairs used by the
# renderer. Indices follow the canonical SMPL joint order:
#   0 Pelvis, 1 L_Hip, 2 R_Hip, 3 Spine1,
#   4 L_Knee, 5 R_Knee, 6 Spine2,
#   7 L_Ankle, 8 R_Ankle, 9 Spine3,
#   10 L_Foot, 11 R_Foot, 12 Neck,
#   13 L_Collar, 14 R_Collar, 15 Head,
#   16 L_Shoulder, 17 R_Shoulder, 18 L_Elbow, 19 R_Elbow,
#   20 L_Wrist, 21 R_Wrist, 22 L_Hand, 23 R_Hand.
# Derived directly from the SMPL `kintree_table` parents.
SMPL_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    # Spine
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    # Shoulders / clavicles
    (9, 13),
    (9, 14),
    # Left arm
    (13, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    # Right arm
    (14, 17),
    (17, 19),
    (19, 21),
    (21, 23),
    # Left leg
    (0, 1),
    (1, 4),
    (4, 7),
    (7, 10),
    # Right leg
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),
)
