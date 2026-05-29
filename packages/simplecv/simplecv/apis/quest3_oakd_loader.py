from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Sequence
from csv import DictReader
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int, Int64
from numpy import ndarray
from rerun import AnnotationInfo, ClassDescription
from serde import coerce, from_dict, serde
from serde import field as serde_field

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.skeleton.assembly_hands import _ASM2COCO, _ASM2COCO_R
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_IDS, COCO_133_LINKS
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from simplecv.umetrack_temp.generic_hand_model_numpy import LANDMARK

# ---- Quest body skeleton metadata ----

_QUEST_BODY_JOINT_NAMES: tuple[str, ...] = (
    "root",
    "hips",
    "spine_lower",
    "spine_middle",
    "spine_upper",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_scapula",
    "left_arm_upper",
    "left_arm_lower",
    "left_hand_wrist_twist",
    "right_shoulder",
    "right_scapula",
    "right_arm_upper",
    "right_arm_lower",
    "right_hand_wrist_twist",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot_ankle_twist",
    "left_foot_ankle",
    "left_foot_subtalar",
    "left_foot_transverse",
    "left_foot_ball",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot_ankle_twist",
    "right_foot_ankle",
    "right_foot_subtalar",
    "right_foot_transverse",
    "right_foot_ball",
)

_QUEST_BODY_NAME_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(_QUEST_BODY_JOINT_NAMES)}

_QUEST_BODY_LINK_NAMES: tuple[tuple[str, str], ...] = (
    ("root", "hips"),
    ("hips", "spine_lower"),
    ("spine_lower", "spine_middle"),
    ("spine_middle", "spine_upper"),
    ("spine_upper", "chest"),
    ("chest", "neck"),
    ("neck", "head"),
    ("chest", "left_shoulder"),
    ("left_shoulder", "left_scapula"),
    ("left_scapula", "left_arm_upper"),
    ("left_arm_upper", "left_arm_lower"),
    ("left_arm_lower", "left_hand_wrist_twist"),
    ("chest", "right_shoulder"),
    ("right_shoulder", "right_scapula"),
    ("right_scapula", "right_arm_upper"),
    ("right_arm_upper", "right_arm_lower"),
    ("right_arm_lower", "right_hand_wrist_twist"),
    ("hips", "left_upper_leg"),
    ("left_upper_leg", "left_lower_leg"),
    ("left_lower_leg", "left_foot_ankle_twist"),
    ("left_foot_ankle_twist", "left_foot_ankle"),
    ("left_foot_ankle", "left_foot_subtalar"),
    ("left_foot_subtalar", "left_foot_transverse"),
    ("left_foot_transverse", "left_foot_ball"),
    ("hips", "right_upper_leg"),
    ("right_upper_leg", "right_lower_leg"),
    ("right_lower_leg", "right_foot_ankle_twist"),
    ("right_foot_ankle_twist", "right_foot_ankle"),
    ("right_foot_ankle", "right_foot_subtalar"),
    ("right_foot_subtalar", "right_foot_transverse"),
    ("right_foot_transverse", "right_foot_ball"),
)

_QUEST_BODY_LINKS: tuple[tuple[int, int], ...] = tuple(
    (_QUEST_BODY_NAME_TO_IDX[src], _QUEST_BODY_NAME_TO_IDX[dst]) for src, dst in _QUEST_BODY_LINK_NAMES
)

_QUEST_BODY_KEYPOINT_IDS: list[int] = list(range(len(_QUEST_BODY_JOINT_NAMES)))
_QUEST_BODY_CLASS_ID: int = 1

# Map a subset of Quest body joints into COCO-133 IDs so body logging can share the same stream
_QUEST_BODY_TO_COCO_ID: dict[str, int] = {
    # Torso/shoulders: use scapula joints for proper lateral spread
    "left_scapula": 5,
    "right_scapula": 6,
    # Arms
    "left_arm_lower": 7,  # elbow
    "right_arm_lower": 8,  # elbow
    "left_hand_wrist_twist": 9,  # wrist
    "right_hand_wrist_twist": 10,  # wrist
    # Legs
    "left_upper_leg": 11,  # hip
    "right_upper_leg": 12,  # hip
    "left_lower_leg": 13,  # knee
    "right_lower_leg": 14,  # knee
    "left_foot_ankle": 15,  # ankle
    "right_foot_ankle": 16,  # ankle
}

# Wrist joint indices in the Quest body skeleton for 6DOF extraction
LEFT_WRIST_BODY_IDX: int = 12  # left_hand_wrist_twist
RIGHT_WRIST_BODY_IDX: int = 17  # right_hand_wrist_twist


class QuestHandLandmark(IntEnum):
    """Quest 3 + Oak-D hand keypoint indices following the CSV column ordering."""

    PALM = 0
    WRIST = 1
    THUMB_METACARPAL = 2
    THUMB_PROXIMAL = 3
    THUMB_DISTAL = 4
    THUMB_TIP = 5
    INDEX_METACARPAL = 6
    INDEX_PROXIMAL = 7
    INDEX_INTERMEDIATE = 8
    INDEX_DISTAL = 9
    INDEX_TIP = 10
    MIDDLE_METACARPAL = 11
    MIDDLE_PROXIMAL = 12
    MIDDLE_INTERMEDIATE = 13
    MIDDLE_DISTAL = 14
    MIDDLE_TIP = 15
    RING_METACARPAL = 16
    RING_PROXIMAL = 17
    RING_INTERMEDIATE = 18
    RING_DISTAL = 19
    RING_TIP = 20
    LITTLE_METACARPAL = 21
    LITTLE_PROXIMAL = 22
    LITTLE_INTERMEDIATE = 23
    LITTLE_DISTAL = 24
    LITTLE_TIP = 25


_QUEST_HAND_LANDMARK_PREFIXES: tuple[str, ...] = (
    "palm",
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
)


class QuestHandSide(IntEnum):
    """Hands available in the Quest CSV exports."""

    LEFT = 0
    RIGHT = 1

    @property
    def label(self) -> str:
        return "left" if self is QuestHandSide.LEFT else "right"

    @property
    def entity_suffix(self) -> str:
        return f"{self.label}_hand"


QUEST_HAND_LANDMARK_COUNT: int = len(QuestHandLandmark)
UME_HAND_LANDMARK_COUNT: int = len(LANDMARK)
LANDMARK_TO_QUEST_INDEX: Int64[ndarray, "n_ume_kpts"] = np.array(
    [
        QuestHandLandmark.THUMB_TIP.value,
        QuestHandLandmark.INDEX_TIP.value,
        QuestHandLandmark.MIDDLE_TIP.value,
        QuestHandLandmark.RING_TIP.value,
        QuestHandLandmark.LITTLE_TIP.value,
        QuestHandLandmark.WRIST.value,
        QuestHandLandmark.THUMB_PROXIMAL.value,
        QuestHandLandmark.THUMB_DISTAL.value,
        QuestHandLandmark.INDEX_PROXIMAL.value,
        QuestHandLandmark.INDEX_INTERMEDIATE.value,
        QuestHandLandmark.INDEX_DISTAL.value,
        QuestHandLandmark.MIDDLE_PROXIMAL.value,
        QuestHandLandmark.MIDDLE_INTERMEDIATE.value,
        QuestHandLandmark.MIDDLE_DISTAL.value,
        QuestHandLandmark.RING_PROXIMAL.value,
        QuestHandLandmark.RING_INTERMEDIATE.value,
        QuestHandLandmark.RING_DISTAL.value,
        QuestHandLandmark.LITTLE_PROXIMAL.value,
        QuestHandLandmark.LITTLE_INTERMEDIATE.value,
        QuestHandLandmark.LITTLE_DISTAL.value,
        QuestHandLandmark.PALM.value,
    ],
    dtype=np.int64,
)


@serde(type_check=coerce)
class QuestHandDictRow:
    """Raw CSV row for Quest 3 + Oak-D hand poses."""

    ts_ns: int = serde_field(rename="ts_ns")
    """Capture timestamp measured in nanoseconds from recording start."""

    palm_x: float
    """Palm center X coordinate in meters."""
    palm_y: float
    """Palm center Y coordinate in meters."""
    palm_z: float
    """Palm center Z coordinate in meters."""

    wrist_x: float
    """Wrist joint X coordinate in meters."""
    wrist_y: float
    """Wrist joint Y coordinate in meters."""
    wrist_z: float
    """Wrist joint Z coordinate in meters."""

    thumb_metacarpal_x: float
    """Thumb metacarpal joint X coordinate in meters."""
    thumb_metacarpal_y: float
    """Thumb metacarpal joint Y coordinate in meters."""
    thumb_metacarpal_z: float
    """Thumb metacarpal joint Z coordinate in meters."""

    thumb_proximal_x: float
    """Thumb proximal joint X coordinate in meters."""
    thumb_proximal_y: float
    """Thumb proximal joint Y coordinate in meters."""
    thumb_proximal_z: float
    """Thumb proximal joint Z coordinate in meters."""

    thumb_distal_x: float
    """Thumb distal joint X coordinate in meters."""
    thumb_distal_y: float
    """Thumb distal joint Y coordinate in meters."""
    thumb_distal_z: float
    """Thumb distal joint Z coordinate in meters."""

    thumb_tip_x: float
    """Thumb fingertip X coordinate in meters."""
    thumb_tip_y: float
    """Thumb fingertip Y coordinate in meters."""
    thumb_tip_z: float
    """Thumb fingertip Z coordinate in meters."""

    index_metacarpal_x: float
    """Index metacarpal joint X coordinate in meters."""
    index_metacarpal_y: float
    """Index metacarpal joint Y coordinate in meters."""
    index_metacarpal_z: float
    """Index metacarpal joint Z coordinate in meters."""

    index_proximal_x: float
    """Index proximal joint X coordinate in meters."""
    index_proximal_y: float
    """Index proximal joint Y coordinate in meters."""
    index_proximal_z: float
    """Index proximal joint Z coordinate in meters."""

    index_intermediate_x: float
    """Index intermediate joint X coordinate in meters."""
    index_intermediate_y: float
    """Index intermediate joint Y coordinate in meters."""
    index_intermediate_z: float
    """Index intermediate joint Z coordinate in meters."""

    index_distal_x: float
    """Index distal joint X coordinate in meters."""
    index_distal_y: float
    """Index distal joint Y coordinate in meters."""
    index_distal_z: float
    """Index distal joint Z coordinate in meters."""

    index_tip_x: float
    """Index fingertip X coordinate in meters."""
    index_tip_y: float
    """Index fingertip Y coordinate in meters."""
    index_tip_z: float
    """Index fingertip Z coordinate in meters."""

    middle_metacarpal_x: float
    """Middle metacarpal joint X coordinate in meters."""
    middle_metacarpal_y: float
    """Middle metacarpal joint Y coordinate in meters."""
    middle_metacarpal_z: float
    """Middle metacarpal joint Z coordinate in meters."""

    middle_proximal_x: float
    """Middle proximal joint X coordinate in meters."""
    middle_proximal_y: float
    """Middle proximal joint Y coordinate in meters."""
    middle_proximal_z: float
    """Middle proximal joint Z coordinate in meters."""

    middle_intermediate_x: float
    """Middle intermediate joint X coordinate in meters."""
    middle_intermediate_y: float
    """Middle intermediate joint Y coordinate in meters."""
    middle_intermediate_z: float
    """Middle intermediate joint Z coordinate in meters."""

    middle_distal_x: float
    """Middle distal joint X coordinate in meters."""
    middle_distal_y: float
    """Middle distal joint Y coordinate in meters."""
    middle_distal_z: float
    """Middle distal joint Z coordinate in meters."""

    middle_tip_x: float
    """Middle fingertip X coordinate in meters."""
    middle_tip_y: float
    """Middle fingertip Y coordinate in meters."""
    middle_tip_z: float
    """Middle fingertip Z coordinate in meters."""

    ring_metacarpal_x: float
    """Ring metacarpal joint X coordinate in meters."""
    ring_metacarpal_y: float
    """Ring metacarpal joint Y coordinate in meters."""
    ring_metacarpal_z: float
    """Ring metacarpal joint Z coordinate in meters."""

    ring_proximal_x: float
    """Ring proximal joint X coordinate in meters."""
    ring_proximal_y: float
    """Ring proximal joint Y coordinate in meters."""
    ring_proximal_z: float
    """Ring proximal joint Z coordinate in meters."""

    ring_intermediate_x: float
    """Ring intermediate joint X coordinate in meters."""
    ring_intermediate_y: float
    """Ring intermediate joint Y coordinate in meters."""
    ring_intermediate_z: float
    """Ring intermediate joint Z coordinate in meters."""

    ring_distal_x: float
    """Ring distal joint X coordinate in meters."""
    ring_distal_y: float
    """Ring distal joint Y coordinate in meters."""
    ring_distal_z: float
    """Ring distal joint Z coordinate in meters."""

    ring_tip_x: float
    """Ring fingertip X coordinate in meters."""
    ring_tip_y: float
    """Ring fingertip Y coordinate in meters."""
    ring_tip_z: float
    """Ring fingertip Z coordinate in meters."""

    little_metacarpal_x: float
    """Little metacarpal joint X coordinate in meters."""
    little_metacarpal_y: float
    """Little metacarpal joint Y coordinate in meters."""
    little_metacarpal_z: float
    """Little metacarpal joint Z coordinate in meters."""

    little_proximal_x: float
    """Little proximal joint X coordinate in meters."""
    little_proximal_y: float
    """Little proximal joint Y coordinate in meters."""
    little_proximal_z: float
    """Little proximal joint Z coordinate in meters."""

    little_intermediate_x: float
    """Little intermediate joint X coordinate in meters."""
    little_intermediate_y: float
    """Little intermediate joint Y coordinate in meters."""
    little_intermediate_z: float
    """Little intermediate joint Z coordinate in meters."""

    little_distal_x: float
    """Little distal joint X coordinate in meters."""
    little_distal_y: float
    """Little distal joint Y coordinate in meters."""
    little_distal_z: float
    """Little distal joint Z coordinate in meters."""

    little_tip_x: float
    """Little fingertip X coordinate in meters."""
    little_tip_y: float
    """Little fingertip Y coordinate in meters."""
    little_tip_z: float
    """Little fingertip Z coordinate in meters."""


@serde(type_check=coerce)
class QuestBodyDictRow:
    """Raw CSV row for Quest 3 + Oak-D full-body poses (32 joints)."""

    ts_ns: int = serde_field(rename="ts_ns")
    """Capture timestamp measured in nanoseconds from recording start."""

    # Spine / torso
    root_x: float
    root_y: float
    root_z: float
    root_qx: float
    root_qy: float
    root_qz: float
    root_qw: float

    hips_x: float
    hips_y: float
    hips_z: float
    hips_qx: float
    hips_qy: float
    hips_qz: float
    hips_qw: float

    spine_lower_x: float
    spine_lower_y: float
    spine_lower_z: float
    spine_lower_qx: float
    spine_lower_qy: float
    spine_lower_qz: float
    spine_lower_qw: float

    spine_middle_x: float
    spine_middle_y: float
    spine_middle_z: float
    spine_middle_qx: float
    spine_middle_qy: float
    spine_middle_qz: float
    spine_middle_qw: float

    spine_upper_x: float
    spine_upper_y: float
    spine_upper_z: float
    spine_upper_qx: float
    spine_upper_qy: float
    spine_upper_qz: float
    spine_upper_qw: float

    chest_x: float
    chest_y: float
    chest_z: float
    chest_qx: float
    chest_qy: float
    chest_qz: float
    chest_qw: float

    neck_x: float
    neck_y: float
    neck_z: float
    neck_qx: float
    neck_qy: float
    neck_qz: float
    neck_qw: float

    head_x: float
    head_y: float
    head_z: float
    head_qx: float
    head_qy: float
    head_qz: float
    head_qw: float

    # Left arm chain
    left_shoulder_x: float
    left_shoulder_y: float
    left_shoulder_z: float
    left_shoulder_qx: float
    left_shoulder_qy: float
    left_shoulder_qz: float
    left_shoulder_qw: float

    left_scapula_x: float
    left_scapula_y: float
    left_scapula_z: float
    left_scapula_qx: float
    left_scapula_qy: float
    left_scapula_qz: float
    left_scapula_qw: float

    left_arm_upper_x: float
    left_arm_upper_y: float
    left_arm_upper_z: float
    left_arm_upper_qx: float
    left_arm_upper_qy: float
    left_arm_upper_qz: float
    left_arm_upper_qw: float

    left_arm_lower_x: float
    left_arm_lower_y: float
    left_arm_lower_z: float
    left_arm_lower_qx: float
    left_arm_lower_qy: float
    left_arm_lower_qz: float
    left_arm_lower_qw: float

    left_hand_wrist_twist_x: float
    left_hand_wrist_twist_y: float
    left_hand_wrist_twist_z: float
    left_hand_wrist_twist_qx: float
    left_hand_wrist_twist_qy: float
    left_hand_wrist_twist_qz: float
    left_hand_wrist_twist_qw: float

    # Right arm chain
    right_shoulder_x: float
    right_shoulder_y: float
    right_shoulder_z: float
    right_shoulder_qx: float
    right_shoulder_qy: float
    right_shoulder_qz: float
    right_shoulder_qw: float

    right_scapula_x: float
    right_scapula_y: float
    right_scapula_z: float
    right_scapula_qx: float
    right_scapula_qy: float
    right_scapula_qz: float
    right_scapula_qw: float

    right_arm_upper_x: float
    right_arm_upper_y: float
    right_arm_upper_z: float
    right_arm_upper_qx: float
    right_arm_upper_qy: float
    right_arm_upper_qz: float
    right_arm_upper_qw: float

    right_arm_lower_x: float
    right_arm_lower_y: float
    right_arm_lower_z: float
    right_arm_lower_qx: float
    right_arm_lower_qy: float
    right_arm_lower_qz: float
    right_arm_lower_qw: float

    right_hand_wrist_twist_x: float
    right_hand_wrist_twist_y: float
    right_hand_wrist_twist_z: float
    right_hand_wrist_twist_qx: float
    right_hand_wrist_twist_qy: float
    right_hand_wrist_twist_qz: float
    right_hand_wrist_twist_qw: float

    # Left leg chain
    left_upper_leg_x: float
    left_upper_leg_y: float
    left_upper_leg_z: float
    left_upper_leg_qx: float
    left_upper_leg_qy: float
    left_upper_leg_qz: float
    left_upper_leg_qw: float

    left_lower_leg_x: float
    left_lower_leg_y: float
    left_lower_leg_z: float
    left_lower_leg_qx: float
    left_lower_leg_qy: float
    left_lower_leg_qz: float
    left_lower_leg_qw: float

    left_foot_ankle_twist_x: float
    left_foot_ankle_twist_y: float
    left_foot_ankle_twist_z: float
    left_foot_ankle_twist_qx: float
    left_foot_ankle_twist_qy: float
    left_foot_ankle_twist_qz: float
    left_foot_ankle_twist_qw: float

    left_foot_ankle_x: float
    left_foot_ankle_y: float
    left_foot_ankle_z: float
    left_foot_ankle_qx: float
    left_foot_ankle_qy: float
    left_foot_ankle_qz: float
    left_foot_ankle_qw: float

    left_foot_subtalar_x: float
    left_foot_subtalar_y: float
    left_foot_subtalar_z: float
    left_foot_subtalar_qx: float
    left_foot_subtalar_qy: float
    left_foot_subtalar_qz: float
    left_foot_subtalar_qw: float

    left_foot_transverse_x: float
    left_foot_transverse_y: float
    left_foot_transverse_z: float
    left_foot_transverse_qx: float
    left_foot_transverse_qy: float
    left_foot_transverse_qz: float
    left_foot_transverse_qw: float

    left_foot_ball_x: float
    left_foot_ball_y: float
    left_foot_ball_z: float
    left_foot_ball_qx: float
    left_foot_ball_qy: float
    left_foot_ball_qz: float
    left_foot_ball_qw: float

    # Right leg chain
    right_upper_leg_x: float
    right_upper_leg_y: float
    right_upper_leg_z: float
    right_upper_leg_qx: float
    right_upper_leg_qy: float
    right_upper_leg_qz: float
    right_upper_leg_qw: float

    right_lower_leg_x: float
    right_lower_leg_y: float
    right_lower_leg_z: float
    right_lower_leg_qx: float
    right_lower_leg_qy: float
    right_lower_leg_qz: float
    right_lower_leg_qw: float

    right_foot_ankle_twist_x: float
    right_foot_ankle_twist_y: float
    right_foot_ankle_twist_z: float
    right_foot_ankle_twist_qx: float
    right_foot_ankle_twist_qy: float
    right_foot_ankle_twist_qz: float
    right_foot_ankle_twist_qw: float

    right_foot_ankle_x: float
    right_foot_ankle_y: float
    right_foot_ankle_z: float
    right_foot_ankle_qx: float
    right_foot_ankle_qy: float
    right_foot_ankle_qz: float
    right_foot_ankle_qw: float

    right_foot_subtalar_x: float
    right_foot_subtalar_y: float
    right_foot_subtalar_z: float
    right_foot_subtalar_qx: float
    right_foot_subtalar_qy: float
    right_foot_subtalar_qz: float
    right_foot_subtalar_qw: float

    right_foot_transverse_x: float
    right_foot_transverse_y: float
    right_foot_transverse_z: float
    right_foot_transverse_qx: float
    right_foot_transverse_qy: float
    right_foot_transverse_qz: float
    right_foot_transverse_qw: float

    right_foot_ball_x: float
    right_foot_ball_y: float
    right_foot_ball_z: float
    right_foot_ball_qx: float
    right_foot_ball_qy: float
    right_foot_ball_qz: float
    right_foot_ball_qw: float

@dataclass
class QuestHandPoseSample:
    """Single Quest hand pose sample containing timestamp and 3D landmarks."""

    timestamp_ns: int
    """Relative timestamp, in nanoseconds from the recording start."""
    keypoints_m: Float32[ndarray, "n_quest_kpts=26 3"]
    """3D keypoints expressed in meters within the Quest coordinate frame."""


@dataclass
class QuestHandPoseSequence:
    """Sequence of Quest hand pose samples parsed from the Quest CSV export."""

    timestamps_ns: Int64[ndarray, "n_frames"]
    """Monotonic capture timestamps for each frame, measured in nanoseconds."""
    keypoints_m: Float32[ndarray, "n_frames n_quest_kpts=26 3"]
    """Per-frame 3D keypoint coordinates in meters."""

    def __len__(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def __iter__(self) -> Iterator[QuestHandPoseSample]:
        for frame_idx in range(len(self)):
            keypoints_frame: Float32[ndarray, "n_quest_kpts=26 3"] = self.keypoints_m[frame_idx]
            yield QuestHandPoseSample(
                timestamp_ns=int(self.timestamps_ns[frame_idx]),
                keypoints_m=keypoints_frame,
            )


@dataclass
class QuestBodyPoseSample:
    """Single Quest full-body pose sample containing timestamped joint data."""

    timestamp_ns: int
    """Relative timestamp, in nanoseconds from the recording start."""
    joint_positions_m: Float32[ndarray, "n_body_joints=32 3"]
    """Per-joint world-space positions expressed in meters."""
    joint_rotations_xyzw: Float32[ndarray, "n_body_joints=32 4"]
    """Per-joint world-space orientations as ``(x, y, z, w)`` quaternions."""


@dataclass
class QuestBodyPoseSequence:
    """Sequence of Quest full-body poses parsed from the Quest CSV export."""

    timestamps_ns: Int64[ndarray, "n_frames"]
    """Monotonic capture timestamps for each frame, measured in nanoseconds."""
    joint_positions_m: Float32[ndarray, "n_frames n_body_joints=32 3"]
    """Stack of joint positions per frame, measured in meters."""
    joint_rotations_xyzw: Float32[ndarray, "n_frames n_body_joints=32 4"]
    """Stack of joint quaternions per frame, stored as ``(x, y, z, w)``."""

    def __len__(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def __iter__(self) -> Iterator[QuestBodyPoseSample]:
        for frame_idx in range(len(self)):
            yield QuestBodyPoseSample(
                timestamp_ns=int(self.timestamps_ns[frame_idx]),
                joint_positions_m=self.joint_positions_m[frame_idx],
                joint_rotations_xyzw=self.joint_rotations_xyzw[frame_idx],
            )


@dataclass
class Quest3VisualizeConfig:
    """Structured configuration for running the Quest3/Oak-D visualization CLI."""

    rr_config: RerunTyroConfig
    """Command-line options for spawning and configuring the Rerun viewer."""
    data_dir: Path
    """Path to the root directory of the Quest3 Oak-D dataset."""


def _select_common_video_timestamps(
    *,
    left_timestamps: Int[ndarray, "n_left"],
    right_timestamps: Int[ndarray, "n_right"],
) -> Int64[ndarray, "n_common"]:
    """Intersect two per-frame timestamp arrays and keep the shared prefix.

    The Quest left/right eye MP4s occasionally disagree by a handful of frames.
    Because we currently downsample all tracker data to the video cadence, we
    trim to the overlapping prefix and warn the caller so the data loss is
    explicit.

    Args:
        left_timestamps: Nanosecond timestamps emitted by the left-eye MP4.
        right_timestamps: Nanosecond timestamps emitted by the right-eye MP4.

    Returns:
        Int64[np.ndarray, "n_common"]: Prefix of ``left_timestamps`` that
        overlaps with ``right_timestamps``. The length equals the minimum frame
        count among both streams.
    """
    left_ts: Int64[ndarray, "n_left"] = np.asarray(left_timestamps, dtype=np.int64)
    right_ts: Int64[ndarray, "n_right"] = np.asarray(right_timestamps, dtype=np.int64)
    if left_ts.size == 0 or right_ts.size == 0:
        raise ValueError("Quest eye videos must contain at least one frame to drive logging.")
    n_common: int = int(min(left_ts.size, right_ts.size))
    if left_ts.size != right_ts.size or not np.array_equal(left_ts[:n_common], right_ts[:n_common]):
        warnings.warn(
            "Quest eye videos have mismatched timestamps; trimming to the overlapping prefix.",
            stacklevel=2,
        )
    return left_ts[:n_common]


def _nearest_sample_indices(
    *,
    source_timestamps_ns: Int64[ndarray, "n_source"],
    target_timestamps_ns: Int64[ndarray, "n_target"],
) -> Int64[ndarray, "n_target"]:
    """Map each ``target`` timestamp to the closest source sample index.

    Args:
        source_timestamps_ns: Monotonic nanosecond timestamps belonging to the
            high-rate tracker stream.
        target_timestamps_ns: Desired nanosecond timestamps, typically the
            MP4-derived ``video_time`` array.

    Returns:
        Int64[np.ndarray, "n_target"]: Index per target timestamp referencing
        the most recent source sample (nearest neighbor to the left).
    """
    if source_timestamps_ns.size == 0:
        raise ValueError("Cannot resample an empty timeline.")
    insertion_points: Int64[ndarray, "n_target"] = (
        np.searchsorted(source_timestamps_ns, target_timestamps_ns, side="right") - 1
    ).astype(np.int64)
    insertion_points[insertion_points < 0] = 0
    insertion_points[insertion_points >= source_timestamps_ns.size] = source_timestamps_ns.size - 1
    return insertion_points


def _resample_head_extrinsics(
    *,
    samples: Sequence[QuestHeadExtrinsicsSample],
    target_timestamps_ns: Int64[ndarray, "n_target"],
) -> list[QuestHeadExtrinsicsSample]:
    """Subsample head extrinsics to match the MP4 frame cadence.

    Args:
        samples: Original tracker samples emitted by Quest at a higher rate.
        target_timestamps_ns: Target nanosecond timestamps (one per video
            frame) that should drive logging.

    Returns:
        list[QuestHeadExtrinsicsSample]: Downsampled extrinsics sharing the
        ``target_timestamps_ns`` cadence so they stay in sync with video logs.
    """
    source_timestamps_ns: Int64[ndarray, "n_source"] = np.asarray(
        [sample.timestamp_ns for sample in samples],
        dtype=np.int64,
    )
    indices: Int64[ndarray, "n_target"] = _nearest_sample_indices(
        source_timestamps_ns=source_timestamps_ns,
        target_timestamps_ns=target_timestamps_ns,
    )
    resampled: list[QuestHeadExtrinsicsSample] = []
    for idx, target_timestamp_ns in zip(indices, target_timestamps_ns, strict=True):
        source_sample: QuestHeadExtrinsicsSample = samples[int(idx)]
        resampled.append(
            QuestHeadExtrinsicsSample(
                timestamp_ns=int(target_timestamp_ns),
                left_extrinsics=source_sample.left_extrinsics,
                right_extrinsics=source_sample.right_extrinsics,
            )
        )
    return resampled


def _resample_hand_sequence(
    *,
    sequence: QuestHandPoseSequence,
    target_timestamps_ns: Int64[ndarray, "n_target"],
) -> QuestHandPoseSequence:
    """Return a new hand sequence with one sample per ``target_timestamps_ns``.

    Args:
        sequence: Raw Quest hand pose sequence containing the original cadence.
        target_timestamps_ns: Target timeline shared with the MP4 logs.

    Returns:
        QuestHandPoseSequence: Copy of ``sequence`` trimmed/resampled to the
        requested timestamps.
    """
    source_timestamps_ns: Int64[ndarray, "n_source"] = sequence.timestamps_ns.astype(np.int64, copy=False)
    indices: Int64[ndarray, "n_target"] = _nearest_sample_indices(
        source_timestamps_ns=source_timestamps_ns,
        target_timestamps_ns=target_timestamps_ns,
    )
    keypoints_resampled: Float32[ndarray, "n_target n_quest_kpts=26 3"] = sequence.keypoints_m[indices]
    resampled_sequence = QuestHandPoseSequence(
        timestamps_ns=target_timestamps_ns.astype(np.int64, copy=False),
        keypoints_m=keypoints_resampled.astype(np.float32, copy=False),
    )
    return resampled_sequence


def load_hand_sequence(csv_path: Path) -> QuestHandPoseSequence:
    """Load Quest 3 hand landmarks from the CSV export."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(encoding="utf-8", newline="") as file:
        reader: DictReader[str] = DictReader(file)
        samples: list[QuestHandPoseSample] = []
        for row_dict in reader:
            if not row_dict or all(value == "" for value in row_dict.values()):
                continue

            sample: QuestHandDictRow = from_dict(QuestHandDictRow, row_dict)
            # convert to numpy array
            xyz_hand_list: list[tuple[float, float, float]] = [
                (getattr(sample, f"{prefix}_x"), getattr(sample, f"{prefix}_y"), getattr(sample, f"{prefix}_z"))
                for prefix in _QUEST_HAND_LANDMARK_PREFIXES
            ]
            xyz_hand: Float32[ndarray, "n_landmarks=26 3"] = np.array(xyz_hand_list, dtype=np.float32)
            timestamp_ns: int = int(sample.ts_ns)

            samples.append(QuestHandPoseSample(timestamp_ns=timestamp_ns, keypoints_m=xyz_hand))

    if not samples:
        raise ValueError(f"CSV file {csv_path} does not contain any pose rows.")

    timestamps_ns: Int64[ndarray, "n_frames"] = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
    keypoints_stack: Float32[ndarray, "n_frames n_quest_kpts=26 3"] = np.stack(
        [sample.keypoints_m for sample in samples],
        axis=0,
    )

    sequence = QuestHandPoseSequence(
        timestamps_ns=timestamps_ns,
        keypoints_m=keypoints_stack,
    )
    sequence.timestamps_ns = sequence.timestamps_ns - int(sequence.timestamps_ns[0])

    zero_mask: Bool[ndarray, "n_frames n_quest_kpts=26"] = np.asarray(
        np.isclose(sequence.keypoints_m, 0.0, atol=1e-6).all(axis=-1),
        dtype=bool,
    )
    sequence.keypoints_m[zero_mask] = np.nan
    return sequence


def load_body_sequence(csv_path: Path) -> QuestBodyPoseSequence:
    """Load Quest 3 full-body joints from CSV, zero-offset timestamps, and NaN-out zeros."""

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(encoding="utf-8", newline="") as file:
        reader: DictReader[str] = DictReader(file)
        samples: list[QuestBodyPoseSample] = []
        for row_dict in reader:
            if not row_dict or all(value == "" for value in row_dict.values()):
                continue

            sample_row: QuestBodyDictRow = from_dict(QuestBodyDictRow, row_dict)
            timestamp_ns: int = int(sample_row.ts_ns)

            positions_list: list[tuple[float, float, float]] = []
            rotations_list: list[tuple[float, float, float, float]] = []
            for joint_name in _QUEST_BODY_JOINT_NAMES:
                pos: tuple[float, float, float] = (
                    getattr(sample_row, f"{joint_name}_x"),
                    getattr(sample_row, f"{joint_name}_y"),
                    getattr(sample_row, f"{joint_name}_z"),
                )
                rot: tuple[float, float, float, float] = (
                    getattr(sample_row, f"{joint_name}_qx"),
                    getattr(sample_row, f"{joint_name}_qy"),
                    getattr(sample_row, f"{joint_name}_qz"),
                    getattr(sample_row, f"{joint_name}_qw"),
                )
                positions_list.append(pos)
                rotations_list.append(rot)

            joint_positions_m: Float32[ndarray, "n_body_joints=32 3"] = np.array(positions_list, dtype=np.float32)
            joint_rotations_xyzw: Float32[ndarray, "n_body_joints=32 4"] = np.array(
                rotations_list,
                dtype=np.float32,
            )
            samples.append(
                QuestBodyPoseSample(
                    timestamp_ns=timestamp_ns,
                    joint_positions_m=joint_positions_m,
                    joint_rotations_xyzw=joint_rotations_xyzw,
                )
            )

    if not samples:
        raise ValueError(f"CSV file {csv_path} does not contain any pose rows.")

    timestamps_ns: Int64[ndarray, "n_frames"] = np.asarray(
        [sample.timestamp_ns for sample in samples],
        dtype=np.int64,
    )
    joint_positions_stack: Float32[ndarray, "n_frames n_body_joints=32 3"] = np.stack(
        [sample.joint_positions_m for sample in samples],
        axis=0,
    )
    joint_rotations_stack: Float32[ndarray, "n_frames n_body_joints=32 4"] = np.stack(
        [sample.joint_rotations_xyzw for sample in samples],
        axis=0,
    )

    timestamps_ns = timestamps_ns - int(timestamps_ns[0])

    zero_mask_body: Bool[ndarray, "n_frames n_body_joints=32"] = np.asarray(
        np.isclose(joint_positions_stack, 0.0, atol=1e-6).all(axis=-1),
        dtype=bool,
    )
    joint_positions_stack[zero_mask_body] = np.nan

    return QuestBodyPoseSequence(
        timestamps_ns=timestamps_ns,
        joint_positions_m=joint_positions_stack,
        joint_rotations_xyzw=joint_rotations_stack,
    )


def _resample_body_sequence(
    *,
    sequence: QuestBodyPoseSequence,
    target_timestamps_ns: Int64[ndarray, "n_target"],
) -> QuestBodyPoseSequence:
    source_timestamps_ns: Int64[ndarray, "n_source"] = sequence.timestamps_ns.astype(np.int64, copy=False)
    indices: Int64[ndarray, "n_target"] = _nearest_sample_indices(
        source_timestamps_ns=source_timestamps_ns,
        target_timestamps_ns=target_timestamps_ns,
    )

    positions_resampled: Float32[ndarray, "n_target n_body_joints=32 3"] = sequence.joint_positions_m[indices]
    rotations_resampled: Float32[ndarray, "n_target n_body_joints=32 4"] = sequence.joint_rotations_xyzw[indices]

    return QuestBodyPoseSequence(
        timestamps_ns=target_timestamps_ns.astype(np.int64, copy=False),
        joint_positions_m=positions_resampled.astype(np.float32, copy=False),
        joint_rotations_xyzw=rotations_resampled.astype(np.float32, copy=False),
    )


def extract_wrist_6dof(
    body_sequence: QuestBodyPoseSequence,
    left_hand_sequence: QuestHandPoseSequence,
    right_hand_sequence: QuestHandPoseSequence,
) -> tuple[
    Float32[ndarray, "n 3"],
    Float32[ndarray, "n 4"],
    Float32[ndarray, "n 3"],
    Float32[ndarray, "n 4"],
]:
    """Extract wrist position and rotation from hand and body sequences.

    Position is taken from the hand CSV (anatomically correct wrist joint),
    rotation is taken from the body CSV (only source with quaternion data).

    Args:
        body_sequence: Resampled Quest body pose sequence for rotation.
        left_hand_sequence: Resampled left hand sequence for position.
        right_hand_sequence: Resampled right hand sequence for position.

    Returns:
        Tuple of (left_pos, left_quat, right_pos, right_quat) where:
        - left_pos: Left wrist positions in meters, shape (n, 3)
        - left_quat: Left wrist quaternions (xyzw), shape (n, 4)
        - right_pos: Right wrist positions in meters, shape (n, 3)
        - right_quat: Right wrist quaternions (xyzw), shape (n, 4)
    """
    # Position from hand CSV (wrist joint = index 1)
    wrist_idx: int = QuestHandLandmark.WRIST
    left_pos: Float32[ndarray, "n 3"] = left_hand_sequence.keypoints_m[:, wrist_idx, :]
    right_pos: Float32[ndarray, "n 3"] = right_hand_sequence.keypoints_m[:, wrist_idx, :]

    # Rotation from body CSV
    left_quat: Float32[ndarray, "n 4"] = body_sequence.joint_rotations_xyzw[:, LEFT_WRIST_BODY_IDX, :]
    right_quat: Float32[ndarray, "n 4"] = body_sequence.joint_rotations_xyzw[:, RIGHT_WRIST_BODY_IDX, :]

    return left_pos, left_quat, right_pos, right_quat


def log_wrist_6dof_batched(
    timestamps_s: Float32[ndarray, "n"] | Float64[ndarray, "n"],
    left_pos: Float32[ndarray, "n 3"],
    left_quat: Float32[ndarray, "n 4"],
    right_pos: Float32[ndarray, "n 3"],
    right_quat: Float32[ndarray, "n 4"],
    timeline: str = "video_time",
    axis_length: float = 0.05,
) -> None:
    """Log wrist 6DOF transforms to Rerun as Transform3D entities.

    Args:
        timestamps_s: Timestamps in seconds for each frame.
        left_pos: Left wrist positions in meters, shape (n, 3).
        left_quat: Left wrist quaternions (xyzw), shape (n, 4).
        right_pos: Right wrist positions in meters, shape (n, 3).
        right_quat: Right wrist quaternions (xyzw), shape (n, 4).
        timeline: Name of the Rerun timeline to log on.
        axis_length: Length of the RGB coordinate axes in meters (default 5cm).
    """
    for path, pos, quat in [
        ("world/gt/left_wrist", left_pos, left_quat),
        ("world/gt/right_wrist", right_pos, right_quat),
    ]:
        rr.send_columns(
            path,
            indexes=[rr.TimeColumn(timeline, duration=timestamps_s)],
            columns=rr.Transform3D.columns(
                translation=pos,
                quaternion=quat,
            ),
        )
        # Log axis visualization (RGB arrows showing orientation)
        rr.log(path, rr.TransformAxes3D(axis_length=axis_length), static=True)


@dataclass(slots=True)
class QuestHeadExtrinsicsSample:
    """Quest head pose extrinsics accompanied by capture timestamp."""

    timestamp_ns: int
    """Relative timestamp, measured in nanoseconds from recording start."""
    left_extrinsics: Extrinsics
    """Camera-to-world pose describing the left-eye tracking camera."""
    right_extrinsics: Extrinsics
    """Camera-to-world pose describing the right-eye tracking camera."""


@dataclass(slots=True)
class QuestHeadExtrinsicsSequence:
    """Batched Quest head pose extrinsics for efficient processing.

    This is a vectorized alternative to `list[QuestHeadExtrinsicsSample]` that avoids
    per-sample object creation overhead. All arrays share the same first dimension `n_frames`.
    """

    timestamps_ns: Int64[ndarray, "n_frames"]
    """Relative timestamps in nanoseconds from recording start."""

    left_world_T_cam: Float32[ndarray, "n_frames 4 4"]
    """World-to-camera homogeneous transforms for the left eye camera."""

    left_cam_T_world: Float32[ndarray, "n_frames 4 4"]
    """Camera-to-world homogeneous transforms for the left eye camera."""

    right_world_T_cam: Float32[ndarray, "n_frames 4 4"]
    """World-to-camera homogeneous transforms for the right eye camera."""

    right_cam_T_world: Float32[ndarray, "n_frames 4 4"]
    """Camera-to-world homogeneous transforms for the right eye camera."""

    def __len__(self) -> int:
        return len(self.timestamps_ns)

    def resample(self, target_timestamps_ns: Int64[ndarray, "n_target"]) -> "QuestHeadExtrinsicsSequence":
        """Resample this sequence to match target timestamps using nearest-neighbor.

        Args:
            target_timestamps_ns: Target nanosecond timestamps to resample to.

        Returns:
            New QuestHeadExtrinsicsSequence with samples at the target timestamps.
        """
        indices: Int64[ndarray, "n_target"] = _nearest_sample_indices(
            source_timestamps_ns=self.timestamps_ns,
            target_timestamps_ns=target_timestamps_ns,
        )
        return QuestHeadExtrinsicsSequence(
            timestamps_ns=target_timestamps_ns.copy(),
            left_world_T_cam=self.left_world_T_cam[indices],
            left_cam_T_world=self.left_cam_T_world[indices],
            right_world_T_cam=self.right_world_T_cam[indices],
            right_cam_T_world=self.right_cam_T_world[indices],
        )


def _log_annotation_context() -> None:
    coco_description: ClassDescription = ClassDescription(
        info=AnnotationInfo(id=0, label="COCO Wholebody", color=(0, 0, 255)),
        keypoint_annotations=[AnnotationInfo(id=kpt_id, label=COCO_133_ID2NAME[kpt_id]) for kpt_id in COCO_133_IDS],
        keypoint_connections=COCO_133_LINKS,
    )

    rr.log(
        "/",
        rr.AnnotationContext(
            [
                coco_description,
            ]
        ),
        static=True,
    )


def _log_coco133_annotations(
    *,
    left_sequence: QuestHandPoseSequence,
    right_sequence: QuestHandPoseSequence,
    body_sequence: QuestBodyPoseSequence,
    head_extrinsics: QuestHeadExtrinsicsSequence,
    left_intrinsics: Intrinsics,
    right_intrinsics: Intrinsics,
    quest_left_cam_path: Path,
    quest_right_cam_path: Path,
    timeline: str,
    log_2d_keypoints: bool = False,
) -> None:
    """Log COCO-133 joints by fusing Quest hands, body, and head extrinsics.

    Args:
        log_2d_keypoints: If True, project 3D keypoints to Quest camera image planes.
            Slower but useful for debugging alignment.

    Uses batched operations and rr.send_columns for ~10x speedup over per-frame logging.
    """
    from einops import rearrange

    left_keypoints: Float32[ndarray, "n_frames_left 21 3"] = left_sequence.keypoints_m[:, LANDMARK_TO_QUEST_INDEX]
    right_keypoints: Float32[ndarray, "n_frames_right 21 3"] = right_sequence.keypoints_m[:, LANDMARK_TO_QUEST_INDEX]
    frame_count: int = min(
        len(head_extrinsics), left_keypoints.shape[0], right_keypoints.shape[0], len(body_sequence)
    )
    if frame_count == 0:
        return

    # === PHASE 1: Batch assemble all COCO-133 frames ===
    # Pre-allocate output arrays
    xyz_stack: Float32[ndarray, "n_frames 133 3"] = np.full((frame_count, 133, 3), np.nan, dtype=np.float32)
    conf_stack: Float32[ndarray, "n_frames 133"] = np.zeros((frame_count, 133), dtype=np.float32)

    # Batch hand keypoints into COCO-133 format
    # Left hand assembly
    for asm_id, coco_ids in _ASM2COCO.items():
        for cid in coco_ids:
            xyz_stack[:, cid, :] = left_keypoints[:frame_count, asm_id, :]
            conf_stack[:, cid] = 1.0

    # Left thumb base interpolation (wrist + CMC) / 2
    left_wrists: Float32[ndarray, "n 3"] = left_keypoints[:frame_count, 5, :]
    left_thumb_cmcs: Float32[ndarray, "n 3"] = left_keypoints[:frame_count, 6, :]
    left_thumb_valid: Bool[ndarray, "n"] = ~(np.isnan(left_wrists).any(axis=1) | np.isnan(left_thumb_cmcs).any(axis=1))
    xyz_stack[left_thumb_valid, 92, :] = (left_wrists[left_thumb_valid] + left_thumb_cmcs[left_thumb_valid]) * 0.5
    conf_stack[left_thumb_valid, 92] = 1.0

    # Right hand assembly
    for asm_id, coco_ids in _ASM2COCO_R.items():
        for cid in coco_ids:
            xyz_stack[:, cid, :] = right_keypoints[:frame_count, asm_id, :]
            conf_stack[:, cid] = 1.0

    # Right thumb base interpolation
    right_wrists: Float32[ndarray, "n 3"] = right_keypoints[:frame_count, 5, :]
    right_thumb_cmcs: Float32[ndarray, "n 3"] = right_keypoints[:frame_count, 6, :]
    right_thumb_valid: Bool[ndarray, "n"] = ~(
        np.isnan(right_wrists).any(axis=1) | np.isnan(right_thumb_cmcs).any(axis=1)
    )
    xyz_stack[right_thumb_valid, 113, :] = (right_wrists[right_thumb_valid] + right_thumb_cmcs[right_thumb_valid]) * 0.5
    conf_stack[right_thumb_valid, 113] = 1.0

    # === PHASE 2: Batch body joint overlay with shoulder widening ===
    body_positions: Float32[ndarray, "n 32 3"] = body_sequence.joint_positions_m[:frame_count].copy()

    # Get shoulder and hip positions
    left_sh_idx: int = _QUEST_BODY_NAME_TO_IDX["left_shoulder"]
    right_sh_idx: int = _QUEST_BODY_NAME_TO_IDX["right_shoulder"]
    left_hip_idx: int = _QUEST_BODY_NAME_TO_IDX["left_upper_leg"]
    right_hip_idx: int = _QUEST_BODY_NAME_TO_IDX["right_upper_leg"]

    left_sh: Float32[ndarray, "n 3"] = body_positions[:, left_sh_idx, :]
    right_sh: Float32[ndarray, "n 3"] = body_positions[:, right_sh_idx, :]
    left_hip: Float32[ndarray, "n 3"] = body_positions[:, left_hip_idx, :]
    right_hip: Float32[ndarray, "n 3"] = body_positions[:, right_hip_idx, :]

    # Compute widths
    sh_dir: Float32[ndarray, "n 3"] = right_sh - left_sh
    sh_width: Float32[ndarray, "n"] = np.linalg.norm(sh_dir, axis=1).astype(np.float32)
    hip_dir: Float32[ndarray, "n 3"] = right_hip - left_hip
    hip_width: Float32[ndarray, "n"] = np.linalg.norm(hip_dir, axis=1).astype(np.float32)

    # Vectorized shoulder widening
    valid_widths: Bool[ndarray, "n"] = hip_width > 1e-6
    safe_sh_width: Float32[ndarray, "n"] = np.where(sh_width < 1e-6, hip_width, sh_width)
    dir_unit: Float32[ndarray, "n 3"] = np.where(
        (sh_width < 1e-6)[:, None],
        hip_dir / np.maximum(hip_width[:, None], 1e-6),
        sh_dir / np.maximum(safe_sh_width[:, None], 1e-6),
    ).astype(np.float32)

    shoulder_center: Float32[ndarray, "n 3"] = 0.5 * (left_sh + right_sh)
    desired_half: Float32[ndarray, "n"] = 0.5 * hip_width

    new_left_sh: Float32[ndarray, "n 3"] = shoulder_center - dir_unit * desired_half[:, None]
    new_right_sh: Float32[ndarray, "n 3"] = shoulder_center + dir_unit * desired_half[:, None]

    delta_left: Float32[ndarray, "n 3"] = new_left_sh - left_sh
    delta_right: Float32[ndarray, "n 3"] = new_right_sh - right_sh

    # Apply shoulder adjustments where valid
    body_positions[valid_widths, left_sh_idx, :] = new_left_sh[valid_widths]
    body_positions[valid_widths, right_sh_idx, :] = new_right_sh[valid_widths]

    # Propagate to arm chains
    left_chain: list[int] = [
        _QUEST_BODY_NAME_TO_IDX["left_arm_upper"],
        _QUEST_BODY_NAME_TO_IDX["left_arm_lower"],
        _QUEST_BODY_NAME_TO_IDX["left_hand_wrist_twist"],
    ]
    right_chain: list[int] = [
        _QUEST_BODY_NAME_TO_IDX["right_arm_upper"],
        _QUEST_BODY_NAME_TO_IDX["right_arm_lower"],
        _QUEST_BODY_NAME_TO_IDX["right_hand_wrist_twist"],
    ]

    for idx in left_chain:
        body_positions[valid_widths, idx, :] += delta_left[valid_widths]
    for idx in right_chain:
        body_positions[valid_widths, idx, :] += delta_right[valid_widths]

    # Overlay body joints into COCO frame (only where hand confidence is 0)
    for joint_name, coco_id in _QUEST_BODY_TO_COCO_ID.items():
        joint_idx: int = _QUEST_BODY_NAME_TO_IDX[joint_name]
        body_xyz: Float32[ndarray, "n 3"] = body_positions[:, joint_idx, :]
        body_valid: Bool[ndarray, "n"] = ~np.isnan(body_xyz).any(axis=1)
        hand_missing: Bool[ndarray, "n"] = (conf_stack[:, coco_id] == 0.0) | np.isnan(conf_stack[:, coco_id])
        fill_mask: Bool[ndarray, "n"] = body_valid & hand_missing
        xyz_stack[fill_mask, coco_id, :] = body_xyz[fill_mask]
        conf_stack[fill_mask, coco_id] = 1.0

    # Clean up NaN confidences
    invalid_mask: Bool[ndarray, "n 133"] = np.isnan(xyz_stack).any(axis=2)
    conf_stack[invalid_mask] = 0.0

    # === PHASE 3: Batch log 3D keypoints with rr.send_columns ===
    coco_entity_path: Path = Path("/world/gt/coco133_xyz")
    timestamps_seconds: Float64[ndarray, "n"] = head_extrinsics.timestamps_ns[:frame_count].astype(np.float64) * 1e-9
    n_keypoints: int = 133

    # Log static metadata once
    rr.log(
        str(coco_entity_path),
        Points3DWithConfidence.from_fields(
            class_ids=0,
            keypoint_ids=COCO_133_IDS,
            show_labels=False,
        ),
        static=True,
    )

    # Flatten and send batched data
    positions_flat: Float32[ndarray, "n_total 3"] = rearrange(
        xyz_stack, "n_frames kpts dim -> (n_frames kpts) dim"
    ).astype(np.float32)
    confidences_flat: Float32[ndarray, "n_total"] = rearrange(
        conf_stack, "n_frames kpts -> (n_frames kpts)"
    ).astype(np.float32)
    keypoint_lengths: Int[ndarray, "n_frames"] = np.full(frame_count, n_keypoints, dtype=np.int32)

    rr.send_columns(
        str(coco_entity_path),
        indexes=[rr.TimeColumn(timeline, duration=timestamps_seconds)],
        columns=[
            *Points3DWithConfidence.columns(
                positions=positions_flat,
                confidences=confidences_flat,
            ).partition(keypoint_lengths),
        ],
    )

    # === PHASE 4: Batch project and log 2D keypoints (optional) ===
    if not log_2d_keypoints:
        return

    # Pre-compute K matrices
    left_k: Float32[ndarray, "3 3"] = left_intrinsics.k_matrix.astype(np.float32)
    right_k: Float32[ndarray, "3 3"] = right_intrinsics.k_matrix.astype(np.float32)
    left_width: float = float(left_intrinsics.width)
    left_height: float = float(left_intrinsics.height)
    right_width: float = float(right_intrinsics.width)
    right_height: float = float(right_intrinsics.height)

    # Get cam_T_world for all frames
    left_cam_T_world: Float32[ndarray, "n 4 4"] = head_extrinsics.left_cam_T_world[:frame_count]
    right_cam_T_world: Float32[ndarray, "n 4 4"] = head_extrinsics.right_cam_T_world[:frame_count]

    # Project all keypoints for all frames
    # For each camera, batch project across all frames
    for cam_path, cam_T_world_batch, k_mat, width, height in [
        (quest_left_cam_path, left_cam_T_world, left_k, left_width, left_height),
        (quest_right_cam_path, right_cam_T_world, right_k, right_width, right_height),
    ]:
        # Allocate UV output
        uv_stack: Float32[ndarray, "n 133 2"] = np.full((frame_count, n_keypoints, 2), np.nan, dtype=np.float32)
        uv_conf_stack: Float32[ndarray, "n 133"] = np.zeros((frame_count, n_keypoints), dtype=np.float32)

        # Project each frame (vectorized inner loop)
        for frame_idx in range(frame_count):
            uv, uv_conf = _project_keypoints_to_uv_fast(
                positions=xyz_stack[frame_idx],
                confidences=conf_stack[frame_idx],
                cam_T_world=cam_T_world_batch[frame_idx],
                k_matrix=k_mat,
                width=width,
                height=height,
            )
            uv_stack[frame_idx] = uv
            uv_conf_stack[frame_idx] = uv_conf

        # Log static metadata
        rr.log(
            str(cam_path / "pinhole" / "coco133_uv"),
            Points2DWithConfidence.from_fields(
                class_ids=0,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
            ),
            static=True,
        )

        # Batch send 2D keypoints
        uv_positions_flat: Float32[ndarray, "n_total 2"] = rearrange(
            uv_stack, "n_frames kpts dim -> (n_frames kpts) dim"
        ).astype(np.float32)
        uv_conf_flat: Float32[ndarray, "n_total"] = rearrange(
            uv_conf_stack, "n_frames kpts -> (n_frames kpts)"
        ).astype(np.float32)

        rr.send_columns(
            str(cam_path / "pinhole" / "coco133_uv"),
            indexes=[rr.TimeColumn(timeline, duration=timestamps_seconds)],
            columns=[
                *Points2DWithConfidence.columns(
                    positions=uv_positions_flat,
                    confidences=uv_conf_flat,
                ).partition(keypoint_lengths),
            ],
        )


def _project_keypoints_to_uv_fast(
    *,
    positions: Float32[ndarray, "n_kpts 3"],
    confidences: Float32[ndarray, "n_kpts"],
    cam_T_world: Float32[ndarray, "4 4"],
    k_matrix: Float32[ndarray, "3 3"],
    width: float,
    height: float,
) -> tuple[Float32[ndarray, "n_kpts 2"], Float32[ndarray, "n_kpts"]]:
    """Project 3D keypoints to 2D UV coordinates without creating Extrinsics objects.

    This is a fast version of _log_projected_keypoints that takes raw matrices
    instead of PinholeParameters to avoid the expensive Extrinsics creation.

    Args:
        positions: 3D keypoint positions in world coordinates.
        confidences: Confidence values for each keypoint.
        cam_T_world: Camera-to-world transformation matrix.
        k_matrix: Camera intrinsics matrix.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tuple of (uv_positions, uv_confidences) for valid keypoints.
    """
    uv_positions: Float32[ndarray, "n_kpts 2"] = np.full((positions.shape[0], 2), np.nan, dtype=np.float32)
    uv_confidences: Float32[ndarray, "n_kpts"] = np.zeros_like(confidences, dtype=np.float32)

    valid_mask: Bool[ndarray, "n_kpts"] = (~np.isnan(positions).any(axis=1)) & (confidences > 0.0)
    valid_indices_all: Int[ndarray, "n_valid"] = np.flatnonzero(valid_mask)

    if valid_indices_all.size == 0:
        return uv_positions, uv_confidences

    # Build projection matrix: P = K @ cam_T_world[:3, :]
    projection_matrix: Float32[ndarray, "3 4"] = k_matrix @ cam_T_world[:3, :]

    # Transform valid positions to homogeneous coordinates
    xyz_valid: Float32[ndarray, "n_valid 3"] = positions[valid_indices_all]
    xyz_hom: Float32[ndarray, "n_valid 4"] = np.concatenate(
        [xyz_valid, np.ones((valid_indices_all.size, 1), dtype=np.float32)],
        axis=1,
    )

    # Project to image coordinates
    uvw: Float32[ndarray, "n_valid 3"] = (projection_matrix @ xyz_hom.T).T
    uv_raw: Float32[ndarray, "n_valid 2"] = uvw[:, :2] / uvw[:, 2:3]

    # Transform to camera coordinates for depth check
    xyz_cam: Float32[ndarray, "n_valid 4"] = (cam_T_world @ xyz_hom.T).T
    depth: Float32[ndarray, "n_valid"] = xyz_cam[:, 2]

    # Apply bounds and depth filtering
    bounds_mask: Bool[ndarray, "n_valid"] = (
        (uv_raw[:, 0] >= 0.0) & (uv_raw[:, 0] <= width) &
        (uv_raw[:, 1] >= 0.0) & (uv_raw[:, 1] <= height)
    )
    positive_depth: Bool[ndarray, "n_valid"] = depth > 0.0
    final_mask: Bool[ndarray, "n_valid"] = bounds_mask & positive_depth

    if np.any(final_mask):
        final_indices: Int[ndarray, "k"] = valid_indices_all[final_mask]
        uv_positions[final_indices] = uv_raw[final_mask]
        uv_confidences[final_indices] = confidences[final_indices]

    return uv_positions, uv_confidences



@serde(type_check=coerce)
class QuestHeadPoseDictRow:
    """Raw CSV row containing Quest head pose information for both controllers."""

    ts_ns: int = serde_field(rename="ts_ns")
    """Capture timestamp of the sample, measured in nanoseconds."""

    left_pos_x: float
    """Left tracker X coordinate in meters."""
    left_pos_y: float
    """Left tracker Y coordinate in meters."""
    left_pos_z: float
    """Left tracker Z coordinate in meters."""
    left_quat_x: float
    """Left tracker quaternion X component."""
    left_quat_y: float
    """Left tracker quaternion Y component."""
    left_quat_z: float
    """Left tracker quaternion Z component."""
    left_quat_w: float
    """Left tracker quaternion W component."""

    right_pos_x: float
    """Right tracker X coordinate in meters."""
    right_pos_y: float
    """Right tracker Y coordinate in meters."""
    right_pos_z: float
    """Right tracker Z coordinate in meters."""
    right_quat_x: float
    """Right tracker quaternion X component."""
    right_quat_y: float
    """Right tracker quaternion Y component."""
    right_quat_z: float
    """Right tracker quaternion Z component."""
    right_quat_w: float
    """Right tracker quaternion W component."""


def _quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: Float32[ndarray, "4"]) -> Float32[ndarray, "3 3"]:
    """Convert an (x, y, z, w) quaternion into a world-from-camera rotation matrix."""
    quat_f64: Float64[ndarray, "4"] = quaternion_xyzw.astype(np.float64)
    norm: float = float(np.linalg.norm(quat_f64))
    if norm == 0.0:
        raise ValueError("Encountered zero-norm quaternion while building Quest head pose extrinsics.")
    quat_unit: Float64[ndarray, "4"] = quat_f64 / norm
    x: float = float(quat_unit[0])
    y: float = float(quat_unit[1])
    z: float = float(quat_unit[2])
    w: float = float(quat_unit[3])

    xx: float = x * x
    yy: float = y * y
    zz: float = z * z
    xy: float = x * y
    xz: float = x * z
    yz: float = y * z
    wx: float = w * x
    wy: float = w * y
    wz: float = w * z

    rotation_matrix: Float32[ndarray, "3 3"] = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return rotation_matrix


def load_head_sequence(head_csv_path: Path) -> list[QuestHeadExtrinsicsSample]:
    """Parse Quest head pose CSV rows into timestamped camera extrinsics."""
    with head_csv_path.open(encoding="utf-8", newline="") as file:
        reader: DictReader[str] = DictReader(file)
        samples: list[QuestHeadExtrinsicsSample] = []
        for row_dict in reader:
            if not row_dict or all(value == "" for value in row_dict.values()):
                continue
            quest_row: QuestHeadPoseDictRow = from_dict(QuestHeadPoseDictRow, row_dict)
            position_m: Float32[ndarray, "3"] = np.array(
                [quest_row.left_pos_x, quest_row.left_pos_y, quest_row.left_pos_z],
                dtype=np.float32,
            )
            rotation_xyzw: Float32[ndarray, "4"] = np.array(
                [quest_row.left_quat_x, quest_row.left_quat_y, quest_row.left_quat_z, quest_row.left_quat_w],
                dtype=np.float32,
            )
            world_R_cam: Float32[ndarray, "3 3"] = _quaternion_xyzw_to_rotation_matrix(rotation_xyzw)
            world_t_cam: Float32[ndarray, "3"] = position_m
            left_extrinsics: Extrinsics = Extrinsics(world_R_cam=world_R_cam, world_t_cam=world_t_cam)

            right_position_m: Float32[ndarray, "3"] = np.array(
                [quest_row.right_pos_x, quest_row.right_pos_y, quest_row.right_pos_z],
                dtype=np.float32,
            )
            right_rotation_xyzw: Float32[ndarray, "4"] = np.array(
                [quest_row.right_quat_x, quest_row.right_quat_y, quest_row.right_quat_z, quest_row.right_quat_w],
                dtype=np.float32,
            )
            right_world_R_cam: Float32[ndarray, "3 3"] = _quaternion_xyzw_to_rotation_matrix(right_rotation_xyzw)
            right_world_t_cam: Float32[ndarray, "3"] = right_position_m
            right_extrinsics: Extrinsics = Extrinsics(world_R_cam=right_world_R_cam, world_t_cam=right_world_t_cam)

            timestamp_ns: int = int(quest_row.ts_ns)
            sample: QuestHeadExtrinsicsSample = QuestHeadExtrinsicsSample(
                timestamp_ns=timestamp_ns,
                left_extrinsics=left_extrinsics,
                right_extrinsics=right_extrinsics,
            )
            samples.append(sample)

    if not samples:
        raise ValueError(f"CSV file {head_csv_path} does not contain any pose rows.")

    return samples


def _quaternion_xyzw_to_rotation_matrix_batched(
    quat_xyzw: Float32[ndarray, "n 4"],
) -> Float32[ndarray, "n 3 3"]:
    """Convert batched quaternions (x, y, z, w) to rotation matrices.

    Args:
        quat_xyzw: Batched quaternions with shape (n, 4) in xyzw order.

    Returns:
        Batched rotation matrices with shape (n, 3, 3).
    """
    x: Float32[ndarray, "n"] = quat_xyzw[:, 0]
    y: Float32[ndarray, "n"] = quat_xyzw[:, 1]
    z: Float32[ndarray, "n"] = quat_xyzw[:, 2]
    w: Float32[ndarray, "n"] = quat_xyzw[:, 3]

    n: int = len(x)
    R: Float32[ndarray, "n 3 3"] = np.zeros((n, 3, 3), dtype=np.float32)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def _se3_inverse_batched(
    T: Float32[ndarray, "n 4 4"],
) -> Float32[ndarray, "n 4 4"]:
    """Compute the inverse of batched SE(3) transformation matrices analytically.

    For an SE(3) matrix T = [R | t; 0 | 1], the inverse is [R^T | -R^T @ t; 0 | 1].
    This is ~100x faster than np.linalg.inv() for batched matrices.

    Args:
        T: Batched 4x4 homogeneous transformation matrices.

    Returns:
        Batched inverse transformation matrices.
    """
    R: Float32[ndarray, "n 3 3"] = T[:, :3, :3]
    t: Float32[ndarray, "n 3"] = T[:, :3, 3]

    R_T: Float32[ndarray, "n 3 3"] = np.swapaxes(R, -2, -1)
    t_inv: Float32[ndarray, "n 3"] = -np.einsum("nij,nj->ni", R_T, t)

    result: Float32[ndarray, "n 4 4"] = np.zeros_like(T)
    result[:, :3, :3] = R_T
    result[:, :3, 3] = t_inv
    result[:, 3, 3] = 1.0

    return result


def load_head_sequence_batched(head_csv_path: Path) -> QuestHeadExtrinsicsSequence:
    """Parse Quest head pose CSV into a batched extrinsics sequence.

    This is ~1000x faster than `load_head_sequence()` for large CSVs by avoiding
    per-row object creation and using vectorized numpy operations.

    Args:
        head_csv_path: Path to the head_pose.csv file.

    Returns:
        QuestHeadExtrinsicsSequence containing all head poses as batched arrays.
    """
    # Read CSV into list of dicts
    with head_csv_path.open(encoding="utf-8", newline="") as file:
        reader: DictReader[str] = DictReader(file)
        rows: list[dict[str, str]] = [
            row for row in reader if row and any(v != "" for v in row.values())
        ]

    if not rows:
        raise ValueError(f"CSV file {head_csv_path} does not contain any pose rows.")

    n: int = len(rows)

    # Pre-allocate arrays
    timestamps_ns: Int64[ndarray, "n"] = np.zeros(n, dtype=np.int64)
    left_positions: Float32[ndarray, "n 3"] = np.zeros((n, 3), dtype=np.float32)
    left_quats: Float32[ndarray, "n 4"] = np.zeros((n, 4), dtype=np.float32)
    right_positions: Float32[ndarray, "n 3"] = np.zeros((n, 3), dtype=np.float32)
    right_quats: Float32[ndarray, "n 4"] = np.zeros((n, 4), dtype=np.float32)

    # Extract data into arrays (still need loop for dict access, but no object creation)
    for i, row in enumerate(rows):
        timestamps_ns[i] = int(row["ts_ns"])
        left_positions[i] = [float(row["left_pos_x"]), float(row["left_pos_y"]), float(row["left_pos_z"])]
        left_quats[i] = [
            float(row["left_quat_x"]),
            float(row["left_quat_y"]),
            float(row["left_quat_z"]),
            float(row["left_quat_w"]),
        ]
        right_positions[i] = [float(row["right_pos_x"]), float(row["right_pos_y"]), float(row["right_pos_z"])]
        right_quats[i] = [
            float(row["right_quat_x"]),
            float(row["right_quat_y"]),
            float(row["right_quat_z"]),
            float(row["right_quat_w"]),
        ]

    # Vectorized quaternion to rotation matrix
    left_R: Float32[ndarray, "n 3 3"] = _quaternion_xyzw_to_rotation_matrix_batched(left_quats)
    right_R: Float32[ndarray, "n 3 3"] = _quaternion_xyzw_to_rotation_matrix_batched(right_quats)

    # Build world_T_cam matrices
    left_world_T_cam: Float32[ndarray, "n 4 4"] = np.zeros((n, 4, 4), dtype=np.float32)
    left_world_T_cam[:, :3, :3] = left_R
    left_world_T_cam[:, :3, 3] = left_positions
    left_world_T_cam[:, 3, 3] = 1.0

    right_world_T_cam: Float32[ndarray, "n 4 4"] = np.zeros((n, 4, 4), dtype=np.float32)
    right_world_T_cam[:, :3, :3] = right_R
    right_world_T_cam[:, :3, 3] = right_positions
    right_world_T_cam[:, 3, 3] = 1.0

    # Analytical SE(3) inverse (vectorized)
    left_cam_T_world: Float32[ndarray, "n 4 4"] = _se3_inverse_batched(left_world_T_cam)
    right_cam_T_world: Float32[ndarray, "n 4 4"] = _se3_inverse_batched(right_world_T_cam)

    return QuestHeadExtrinsicsSequence(
        timestamps_ns=timestamps_ns,
        left_world_T_cam=left_world_T_cam,
        left_cam_T_world=left_cam_T_world,
        right_world_T_cam=right_world_T_cam,
        right_cam_T_world=right_cam_T_world,
    )


@serde
class QuestLensIntrinsics:
    """Pinhole camera intrinsics exported by the Quest device."""

    focal_length_x: float
    """Focal length along the X axis expressed in pixels."""
    focal_length_y: float
    """Focal length along the Y axis expressed in pixels."""
    principal_point_x: float
    """Principal point horizontal offset in pixels."""
    principal_point_y: float
    """Principal point vertical offset in pixels."""
    skew: int
    """Skew coefficient coupling the X and Y axes."""
    available: bool
    """Flag indicating whether calibrated intrinsics are available."""


@serde(type_check=coerce)
class QuestCaptureResolution:
    """Image resolution used for Quest capture."""

    width: int
    """Capture width in pixels."""
    height: int
    """Capture height in pixels."""


@serde(type_check=coerce)
class QuestCameraIntrinsicsDocument:
    """Quest camera intrinsics JSON document. Keep only important bits."""

    lens_intrinsics: QuestLensIntrinsics
    """Calibrated lens intrinsics expressed in pixel units."""
    capture_resolution: QuestCaptureResolution
    """Image resolution active during capture."""
    positional_layout: str | None = None
    """Optional tag describing whether the camera is layed out on the left or right eye."""


@serde(type_check=coerce)
class QuestCombinedCalibrationDocument:
    """Quest calibration document containing intrinsics for multiple sensors."""

    intrinsics: list[QuestCameraIntrinsicsDocument]
    """Per-camera intrinsics blocks present in the calibration JSON."""


def load_camera_intrinsics(intrinsics_path: Path, *, positional_layout: str | None = None) -> Intrinsics:
    """Load Quest camera intrinsics metadata from a JSON file."""
    if not intrinsics_path.exists():
        raise FileNotFoundError(intrinsics_path)

    with intrinsics_path.open(encoding="utf-8") as file:
        raw_intrinsics: dict[str, object] = json.load(file)

    def _build_intrinsics(doc: QuestCameraIntrinsicsDocument) -> Intrinsics:
        return Intrinsics(
            camera_conventions="RDF",
            fl_x=doc.lens_intrinsics.focal_length_x,
            fl_y=doc.lens_intrinsics.focal_length_y,
            cx=doc.lens_intrinsics.principal_point_x,
            cy=doc.lens_intrinsics.principal_point_y,
            width=doc.capture_resolution.width,
            height=doc.capture_resolution.height,
        )

    if "intrinsics" in raw_intrinsics:
        calibration_doc: QuestCombinedCalibrationDocument = from_dict(QuestCombinedCalibrationDocument, raw_intrinsics)
        if positional_layout is None:
            raise ValueError(
                "Combined calibration JSON detected. Please provide positional_layout (e.g. 'left' or 'right')."
            )

        matching_docs: list[QuestCameraIntrinsicsDocument] = [
            doc for doc in calibration_doc.intrinsics if doc.positional_layout == positional_layout
        ]
        if not matching_docs:
            available_layouts: set[str] = {
                doc.positional_layout for doc in calibration_doc.intrinsics if doc.positional_layout is not None
            }
            raise ValueError(
                f"Could not find intrinsics for layout '{positional_layout}' in {intrinsics_path}. "
                f"Available layouts: {sorted(available_layouts)}"
            )
        return _build_intrinsics(matching_docs[0])

    intrinsics_doc: QuestCameraIntrinsicsDocument = from_dict(QuestCameraIntrinsicsDocument, raw_intrinsics)
    if positional_layout is not None and intrinsics_doc.positional_layout not in (None, positional_layout):
        raise ValueError(
            f"Requested layout '{positional_layout}' mismatches JSON layout '{intrinsics_doc.positional_layout}'."
        )
    return _build_intrinsics(intrinsics_doc)


def _log_head_cameras(
    extrinsics_seq: QuestHeadExtrinsicsSequence,
    *,
    left_intrinsics: Intrinsics,
    right_intrinsics: Intrinsics,
    left_cam_path: Path,
    right_cam_path: Path,
    timeline: str = "video_time",
) -> None:
    """Log Quest head cameras over time using batched send_columns API.

    This replaces the row-by-row logging with a single batched operation,
    achieving ~100x speedup for large sequences.
    """

    # Log static intrinsics once (the pinhole projection doesn't change)
    # Using a shared archetype for both cameras since intrinsics are logged separately
    from simplecv.rerun_custom_types import PinholeWithDistortion

    # Create dummy PinholeParameters just for intrinsics logging (extrinsics don't matter for static pinhole)
    left_dummy_extrinsics: Extrinsics = Extrinsics(
        world_R_cam=extrinsics_seq.left_world_T_cam[0, :3, :3],
        world_t_cam=extrinsics_seq.left_world_T_cam[0, :3, 3],
    )
    right_dummy_extrinsics: Extrinsics = Extrinsics(
        world_R_cam=extrinsics_seq.right_world_T_cam[0, :3, :3],
        world_t_cam=extrinsics_seq.right_world_T_cam[0, :3, 3],
    )

    left_camera_params: PinholeParameters = PinholeParameters(
        name="quest_left_eye",
        extrinsics=left_dummy_extrinsics,
        intrinsics=left_intrinsics,
    )
    right_camera_params: PinholeParameters = PinholeParameters(
        name="quest_right_eye",
        extrinsics=right_dummy_extrinsics,
        intrinsics=right_intrinsics,
    )

    # Log static pinhole intrinsics
    rr.log(
        f"{left_cam_path}/pinhole",
        PinholeWithDistortion.from_camera(left_camera_params, image_plane_distance=0.05),
        static=True,
    )
    rr.log(
        f"{right_cam_path}/pinhole",
        PinholeWithDistortion.from_camera(right_camera_params, image_plane_distance=0.05),
        static=True,
    )

    # === Extract cam_T_world (inverse of world_T_cam) ===
    # log_pinhole uses cam_T_world with from_parent=True (ChildFromParent)
    # So we need to invert world_T_cam to cam_T_world and specify ChildFromParent
    
    # Left camera: compute cam_T_world from world_T_cam
    left_world_R_cam: Float32[ndarray, "n 3 3"] = extrinsics_seq.left_world_T_cam[:, :3, :3]
    left_world_t_cam: Float32[ndarray, "n 3"] = extrinsics_seq.left_world_T_cam[:, :3, 3]
    left_cam_R_world: Float32[ndarray, "n 3 3"] = np.transpose(left_world_R_cam, axes=(0, 2, 1))
    left_cam_t_world: Float32[ndarray, "n 3"] = -np.einsum("nij,nj->ni", left_cam_R_world, left_world_t_cam)

    # Right camera: compute cam_T_world from world_T_cam
    right_world_R_cam: Float32[ndarray, "n 3 3"] = extrinsics_seq.right_world_T_cam[:, :3, :3]
    right_world_t_cam: Float32[ndarray, "n 3"] = extrinsics_seq.right_world_T_cam[:, :3, 3]
    right_cam_R_world: Float32[ndarray, "n 3 3"] = np.transpose(right_world_R_cam, axes=(0, 2, 1))
    right_cam_t_world: Float32[ndarray, "n 3"] = -np.einsum("nij,nj->ni", right_cam_R_world, right_world_t_cam)

    # Convert timestamps to TimeColumn
    timestamps_seconds: Float64[ndarray, "n"] = extrinsics_seq.timestamps_ns.astype(np.float64) * 1e-9
    n_frames: int = len(timestamps_seconds)

    # Relation must be broadcast to match number of frames
    relation_array: list[rr.components.TransformRelation] = [
        rr.components.TransformRelation.ChildFromParent
    ] * n_frames

    # Batch log left camera transforms using send_columns
    rr.send_columns(
        str(left_cam_path),
        indexes=[rr.TimeColumn(timeline, duration=timestamps_seconds)],
        columns=rr.Transform3D.columns(
            translation=left_cam_t_world,
            mat3x3=left_cam_R_world,
            relation=relation_array,
        ),
    )

    # Batch log right camera transforms using send_columns
    rr.send_columns(
        str(right_cam_path),
        indexes=[rr.TimeColumn(timeline, duration=timestamps_seconds)],
        columns=rr.Transform3D.columns(
            translation=right_cam_t_world,
            mat3x3=right_cam_R_world,
            relation=relation_array,
        ),
    )


def load_and_log_quest_data(
    config: Quest3VisualizeConfig,
    *,
    timeline: str = "video_time",
    log_2d_keypoints: bool = False,
) -> tuple[list[Path], Int64[ndarray, "n_frames"], Float32[ndarray, "n_frames 4 4"]]:
    """Ingest Quest3+OAK data, log resampled tracks, and return pinhole roots.

    Args:
        config: Filesystem + viewer configuration describing where Quest data
            lives and how to initialize Rerun.
        timeline: Logical timeline used for all emitted logs. Defaults to
            ``"video_time"`` which matches our MP4 timestamps.
        log_2d_keypoints: If True, project 3D keypoints to camera image planes.
            Slower but useful for debugging alignment. Default False.

    Returns:
        list[Path]: ``/world/ego/quest3_left/right`` pinhole entity roots so
        callers can embed them inside a blueprint.
        Int64[ndarray, "n_frames"]: Timestamps for each video frame in nanoseconds.
        Float32[ndarray, "n_frames 4 4"]: Batched left eye world_T_cam transforms
            for aligning other cameras to the Quest frame.
    """
    data_root: Path = config.data_dir
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    rr.log("/", rr.ViewCoordinates.RUB, static=True)

    left_csv: Path = data_root / "quest" / "left_hand_poses.csv"
    right_csv: Path = data_root / "quest" / "right_hand_poses.csv"
    head_csv: Path = data_root / "quest" / "head_pose.csv"
    body_csv: Path = data_root / "quest" / "body_poses.csv"
    calibration_json: Path = data_root / "quest" / "calibration.json"
    left_video_path: Path = data_root / "quest" / "left.mp4"
    right_video_path: Path = data_root / "quest" / "right.mp4"
    if not left_csv.exists():
        raise FileNotFoundError(left_csv)
    if not right_csv.exists():
        raise FileNotFoundError(right_csv)
    if not head_csv.exists():
        raise FileNotFoundError(head_csv)
    if not calibration_json.exists():
        raise FileNotFoundError(calibration_json)
    if not body_csv.exists():
        raise FileNotFoundError(body_csv)
    if not left_video_path.exists():
        raise FileNotFoundError(left_video_path)
    if not right_video_path.exists():
        raise FileNotFoundError(right_video_path)

    head_extrinsics_raw: QuestHeadExtrinsicsSequence = load_head_sequence_batched(head_csv)
    body_sequence_raw: QuestBodyPoseSequence = load_body_sequence(body_csv)
    left_intrinsics: Intrinsics = load_camera_intrinsics(calibration_json, positional_layout="left")
    right_intrinsics: Intrinsics = load_camera_intrinsics(calibration_json, positional_layout="right")

    quest_left_cam_path: Path = Path("/world/ego/quest3_left")
    quest_right_cam_path: Path = Path("/world/ego/quest3_right")

    _log_annotation_context()

    _left_video_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        video_source=left_video_path,
        video_log_path=quest_left_cam_path / "pinhole" / "video",
        timeline=timeline,
    )
    _right_video_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        video_source=right_video_path,
        video_log_path=quest_right_cam_path / "pinhole" / "video",
        timeline=timeline,
    )

    quest_video_timestamps_ns: Int64[ndarray, "n_frames"] = _select_common_video_timestamps(
        left_timestamps=_left_video_timestamps_ns,
        right_timestamps=_right_video_timestamps_ns,
    )

    # Resample head extrinsics to video frame rate using efficient batched method
    head_extrinsics: QuestHeadExtrinsicsSequence = head_extrinsics_raw.resample(quest_video_timestamps_ns)

    body_sequence: QuestBodyPoseSequence = _resample_body_sequence(
        sequence=body_sequence_raw,
        target_timestamps_ns=quest_video_timestamps_ns,
    )

    sequence_map: dict[QuestHandSide, QuestHandPoseSequence] = {
        side: load_hand_sequence(csv_path)
        for side, csv_path in (
            (QuestHandSide.LEFT, left_csv),
            (QuestHandSide.RIGHT, right_csv),
        )
    }
    if QuestHandSide.LEFT not in sequence_map or QuestHandSide.RIGHT not in sequence_map:
        raise ValueError("Both left and right hand pose sequences are required for COCO-133 logging.")

    sequence_map[QuestHandSide.LEFT] = _resample_hand_sequence(
        sequence=sequence_map[QuestHandSide.LEFT],
        target_timestamps_ns=quest_video_timestamps_ns,
    )
    sequence_map[QuestHandSide.RIGHT] = _resample_hand_sequence(
        sequence=sequence_map[QuestHandSide.RIGHT],
        target_timestamps_ns=quest_video_timestamps_ns,
    )

    _log_head_cameras(
        head_extrinsics,
        left_intrinsics=left_intrinsics,
        right_intrinsics=right_intrinsics,
        left_cam_path=quest_left_cam_path,
        right_cam_path=quest_right_cam_path,
        timeline=timeline,
    )

    _log_coco133_annotations(
        left_sequence=sequence_map[QuestHandSide.LEFT],
        right_sequence=sequence_map[QuestHandSide.RIGHT],
        body_sequence=body_sequence,
        head_extrinsics=head_extrinsics,
        left_intrinsics=left_intrinsics,
        right_intrinsics=right_intrinsics,
        quest_left_cam_path=quest_left_cam_path,
        quest_right_cam_path=quest_right_cam_path,
        timeline=timeline,
        log_2d_keypoints=log_2d_keypoints,
    )

    # Log wrist 6DOF transforms
    left_wrist_pos, left_wrist_quat, right_wrist_pos, right_wrist_quat = extract_wrist_6dof(
        body_sequence,
        left_hand_sequence=sequence_map[QuestHandSide.LEFT],
        right_hand_sequence=sequence_map[QuestHandSide.RIGHT],
    )
    timestamps_s: Float64[ndarray, "n"] = quest_video_timestamps_ns.astype(np.float64) * 1e-9
    log_wrist_6dof_batched(
        timestamps_s=timestamps_s,
        left_pos=left_wrist_pos,
        left_quat=left_wrist_quat,
        right_pos=right_wrist_pos,
        right_quat=right_wrist_quat,
        timeline=timeline,
    )

    quest_pinhole_paths: list[Path] = [
        quest_left_cam_path / "pinhole",
        quest_right_cam_path / "pinhole",
    ]
    return quest_pinhole_paths, quest_video_timestamps_ns, head_extrinsics.left_world_T_cam


def main(config: Quest3VisualizeConfig) -> None:
    load_and_log_quest_data(config)
