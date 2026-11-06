"""Keypoint skeleton formats shared by every posekit model.

A skeleton is pure metadata (names + links). Predictions carry a reference to
their skeleton so downstream consumers (Rerun logging, triangulation, format
projection) never have to guess what index 17 means.
"""

from dataclasses import dataclass

from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_LINKS


@dataclass(frozen=True, slots=True)
class KeypointSkeleton:
    """Named keypoint layout with connectivity."""

    name: str
    """Stable format identifier, e.g. ``"coco_17"``."""
    keypoint_names: tuple[str, ...]
    """Per-index keypoint names; ``len(keypoint_names)`` is the keypoint count."""
    links: tuple[tuple[int, int], ...]
    """Keypoint index pairs forming the skeleton edges."""

    @property
    def num_keypoints(self) -> int:
        """Number of keypoints in this skeleton."""
        return len(self.keypoint_names)


COCO_133 = KeypointSkeleton(
    name="coco_133",
    keypoint_names=tuple(COCO_133_ID2NAME[idx] for idx in range(len(COCO_133_ID2NAME))),
    links=tuple((int(a), int(b)) for a, b in COCO_133_LINKS),
)
"""COCO WholeBody: 17 body + 6 foot + 68 face + 42 hand keypoints."""

COCO_17 = KeypointSkeleton(
    name="coco_17",
    keypoint_names=COCO_133.keypoint_names[:17],
    links=tuple(link for link in COCO_133.links if link[0] < 17 and link[1] < 17),
)
"""COCO body-only subset (indices 0-16 of COCO WholeBody)."""

HAND_21 = KeypointSkeleton(
    name="hand_21",
    keypoint_names=tuple(MEDIAPIPE_ID2NAME[idx] for idx in range(len(MEDIAPIPE_ID2NAME))),
    links=tuple((int(a), int(b)) for a, b in MEDIAPIPE_LINKS),
)
"""21-keypoint hand layout shared by MediaPipe and MANO joint regressors."""

SOMA_HAND: KeypointSkeleton = KeypointSkeleton(
    "soma_hand_25",
    ("wrist", "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip")
    + tuple(
        f"{finger}_{joint}"
        for finger in ("index", "middle", "ring", "little")
        for joint in ("metacarpal", "proximal", "intermediate", "distal", "tip")
    ),
    tuple((index - 1 if index not in (1, 5, 10, 15, 20) else 0, index) for index in range(1, 25)),
)


MHR_70: KeypointSkeleton = KeypointSkeleton(
    "mhr_70",
    (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_hip",
        "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
        "right_small_toe", "right_heel", "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint", "right_forefinger4",
        "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint", "right_middle_finger4", "right_middle_finger3",
        "right_middle_finger2", "right_middle_finger_third_joint", "right_ring_finger4", "right_ring_finger3", "right_ring_finger2",
        "right_ring_finger_third_joint", "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint",
        "right_wrist", "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint", "left_forefinger4", "left_forefinger3",
        "left_forefinger2", "left_forefinger_third_joint", "left_middle_finger4", "left_middle_finger3", "left_middle_finger2",
        "left_middle_finger_third_joint", "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint",
        "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint", "left_wrist", "left_olecranon",
        "right_olecranon", "left_cubital_fossa", "right_cubital_fossa", "left_acromion", "right_acromion", "neck",
    ),
    (
        (13, 11), (11, 9), (14, 12), (12, 10), (9, 10), (5, 9), (6, 10), (5, 6), (5, 7), (6, 8), (7, 62), (8, 41), (1, 2), (0, 1), (0, 2), (1, 3),
        (2, 4), (3, 5), (4, 6), (13, 15), (13, 16), (13, 17), (14, 18), (14, 19), (14, 20), (62, 45), (45, 44), (44, 43), (43, 42), (62, 49),
        (49, 48), (48, 47), (47, 46), (62, 53), (53, 52), (52, 51), (51, 50), (62, 57), (57, 56), (56, 55), (55, 54), (62, 61), (61, 60), (60, 59),
        (59, 58), (41, 24), (24, 23), (23, 22), (22, 21), (41, 28), (28, 27), (27, 26), (26, 25), (41, 32), (32, 31), (31, 30), (30, 29), (41, 36),
        (36, 35), (35, 34), (34, 33), (41, 40), (40, 39), (39, 38), (38, 37),
    ),
)
"""The first 70 of the 308 MHR keypoints (SAM 3D Body): body and feet, right hand 21-41, left hand 42-62 (wrist last), then arm and neck points.

Names and links are upstream's ``sam_3d_body.metadata.mhr70.pose_info`` (b5c765a), kept here so the skeleton
registry imports without the posekit-sam3d-body feature; ``test_mhr70_matches_upstream`` checks them.
"""


SKELETONS: dict[str, KeypointSkeleton] = {skeleton.name: skeleton for skeleton in (COCO_17, COCO_133, HAND_21, SOMA_HAND, MHR_70)}
"""Registry of built-in skeleton formats keyed by ``KeypointSkeleton.name``."""
