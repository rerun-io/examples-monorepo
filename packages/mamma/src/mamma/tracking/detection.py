"""Per-camera person detections the tracker's identity logic consumes (boxes, scores, CLIP features).

The detector and the CLIP encoder are posekit roles (``TrackerConfig.detector``, ``ClipIdentity``);
see :meth:`mamma.tracking.tracker.MultiViewTracker._detect`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float32
from numpy import ndarray


@dataclass(slots=True)
class PersonDetections:
    """Person detections for one frame, sorted by descending confidence."""

    boxes_xyxy: Float32[ndarray, "n 4"]
    """Pixel-space boxes ``[x1, y1, x2, y2]`` in the detection frame's resolution."""
    scores: Float32[ndarray, "n"]
    """Detector confidence per box."""
    features: Float32[torch.Tensor, "n d"]
    """CLIP appearance embedding of each box (CPU, unnormalized), same order as ``boxes_xyxy``."""

    def __len__(self) -> int:
        return int(self.boxes_xyxy.shape[0])

    @property
    def centers_xy1(self) -> Float32[ndarray, "n 3"]:
        """Homogeneous box centers ``[cx, cy, 1]`` (epipolar reference points)."""
        centers: Float32[ndarray, "n 3"] = np.ones((len(self), 3), dtype=np.float32)
        centers[:, 0] = (self.boxes_xyxy[:, 0] + self.boxes_xyxy[:, 2]) / 2.0
        centers[:, 1] = (self.boxes_xyxy[:, 1] + self.boxes_xyxy[:, 3]) / 2.0
        return centers


def bbox_iou_xyxy(box_a: Float32[ndarray, "4"], box_b: Float32[ndarray, "4"]) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes (original ``_bbox_iou_xyxy``)."""
    ix1: float = max(float(box_a[0]), float(box_b[0]))
    iy1: float = max(float(box_a[1]), float(box_b[1]))
    ix2: float = min(float(box_a[2]), float(box_b[2]))
    iy2: float = min(float(box_a[3]), float(box_b[3]))
    inter: float = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a: float = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b: float = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union: float = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0
