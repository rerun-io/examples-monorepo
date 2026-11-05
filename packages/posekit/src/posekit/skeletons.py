"""Keypoint skeleton formats shared by every posekit model.

A skeleton is pure metadata (names + links). Predictions carry a reference to
their skeleton so downstream consumers (Rerun logging, triangulation, format
projection) never have to guess what index 17 means.
"""

from dataclasses import dataclass

from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_LINKS

from sam3d_body.metadata.mhr70 import MHR70_ID2NAME, MHR70_LINKS


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
    tuple(MHR70_ID2NAME[index] for index in range(len(MHR70_ID2NAME))),
    tuple((int(a), int(b)) for a, b in MHR70_LINKS),
)
"""The first 70 of the 308 MHR keypoints (SAM 3D Body): body and feet, right hand 21-41, left hand 42-62 (wrist last), then arm and neck points."""


SKELETONS: dict[str, KeypointSkeleton] = {skeleton.name: skeleton for skeleton in (COCO_17, COCO_133, HAND_21, SOMA_HAND, MHR_70)}
"""Registry of built-in skeleton formats keyed by ``KeypointSkeleton.name``."""
