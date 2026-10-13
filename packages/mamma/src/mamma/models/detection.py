"""YOLOv12-X person detector (Ultralytics) in posekit's ``PersonDetector`` role.

Ported from the original ``segmentation/core/pipeline.py``: confidence 0.5, class 0 (person) only, greedy dedup at IoU 0.75
keeping the higher-confidence box. Detection runs sparsely (bootstrap and re-detect ticks), so frames take one host hop.
"""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray
from posekit.models import PersonDetector
from posekit.predictions import BoxDetections
from torch import Tensor


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


class Yolo12xDetector(PersonDetector):
    def __init__(self, weights_path: Path, confidence: float = 0.5, dedup_iou: float = 0.75, device: str = "cuda") -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path), verbose=False)
        self.model.to(device)
        self.confidence, self.dedup_iou = confidence, dedup_iou

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> BoxDetections:
        """Every frame's person boxes, highest confidence first within a frame."""
        boxes: list[Float32[ndarray, "4"]] = []
        scores: list[float] = []
        frame_indices: list[int] = []
        for index, frame in enumerate(frames_rgb.cpu().numpy()):
            result = self.model(np.ascontiguousarray(frame[:, :, ::-1]), verbose=False)[0]  # Ultralytics reads BGR arrays
            found: list[tuple[Float32[ndarray, "4"], float]] = [
                (box.xyxy[0].cpu().numpy().astype(np.float32), float(box.conf)) for box in result.boxes if int(box.cls) == 0 and float(box.conf) >= self.confidence
            ]
            order: list[int] = sorted(range(len(found)), key=lambda i: found[i][1], reverse=True)
            while order:
                kept: int = order.pop(0)
                boxes.append(found[kept][0])
                scores.append(found[kept][1])
                frame_indices.append(index)
                order = [j for j in order if bbox_iou_xyxy(found[kept][0], found[j][0]) < self.dedup_iou]
        device: torch.device = frames_rgb.device
        return BoxDetections(
            xyxy=torch.as_tensor(np.stack(boxes) if boxes else np.zeros((0, 4), dtype=np.float32), device=device),
            scores=torch.as_tensor(scores, dtype=torch.float32, device=device),
            frame_indices=torch.as_tensor(frame_indices, dtype=torch.int64, device=device),
        )
