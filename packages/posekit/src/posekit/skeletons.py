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

SKELETONS: dict[str, KeypointSkeleton] = {skeleton.name: skeleton for skeleton in (COCO_17, COCO_133, HAND_21)}
"""Registry of built-in skeleton formats keyed by ``KeypointSkeleton.name``."""
