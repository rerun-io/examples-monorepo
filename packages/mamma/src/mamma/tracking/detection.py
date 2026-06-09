"""YOLO person detection, ported from the original ``segmentation/core/pipeline.py``.

Detection runs sparsely (bootstrap tick + periodic re-detects), so frames take
one GPU->CPU hop here; the dense per-tick path (mask tracking) stays on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray


@dataclass(slots=True)
class PersonDetections:
    """Person detections for one frame, sorted by descending confidence."""

    boxes_xyxy: Float32[ndarray, "n 4"]
    """Pixel-space boxes ``[x1, y1, x2, y2]`` in the detection frame's resolution."""
    scores: Float32[ndarray, "n"]
    """YOLO confidence per box."""
    crops: list[UInt8[ndarray, "ch cw 3"]]
    """RGB crops of each box (for CLIP identity), same order as ``boxes_xyxy``."""

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


class PersonDetector:
    """YOLOv12-X person detector with greedy IoU deduplication.

    Mirrors the original defaults: confidence 0.5, class 0 (person) only,
    dedup at IoU 0.75 keeping the higher-confidence box.
    """

    def __init__(
        self,
        weights_path: Path,
        confidence: float = 0.5,
        dedup_iou: float = 0.75,
        device: str = "cuda",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path), verbose=False)
        self.model.to(device)
        self.confidence: float = confidence
        self.dedup_iou: float = dedup_iou

    def detect(self, rgb_chw: UInt8[torch.Tensor, "3 h w"]) -> PersonDetections:
        """Detect persons in one RGB CHW frame (GPU tensor ok; copies to CPU)."""
        rgb_hwc: UInt8[ndarray, "h w 3"] = rgb_chw.permute(1, 2, 0).contiguous().cpu().numpy()
        # ultralytics expects BGR numpy for array inputs.
        results = self.model(rgb_hwc[:, :, ::-1], verbose=False)

        boxes_list: list[Float32[ndarray, "4"]] = []
        scores_list: list[float] = []
        for box in results[0].boxes:
            if int(box.cls) != 0 or float(box.conf) < self.confidence:
                continue
            boxes_list.append(box.xyxy[0].cpu().numpy().astype(np.float32))
            scores_list.append(float(box.conf))

        # Greedy dedup: keep highest-confidence box among IoU > threshold groups.
        order: list[int] = sorted(range(len(boxes_list)), key=lambda i: scores_list[i], reverse=True)
        kept: list[int] = []
        while order:
            i: int = order.pop(0)
            kept.append(i)
            order = [j for j in order if bbox_iou_xyxy(boxes_list[i], boxes_list[j]) < self.dedup_iou]

        boxes: Float32[ndarray, "n 4"] = (
            np.stack([boxes_list[i] for i in kept], axis=0) if kept else np.zeros((0, 4), dtype=np.float32)
        )
        scores: Float32[ndarray, "n"] = np.asarray([scores_list[i] for i in kept], dtype=np.float32)
        height: int = rgb_hwc.shape[0]
        width: int = rgb_hwc.shape[1]
        crops: list[UInt8[ndarray, "ch cw 3"]] = []
        for b in boxes:
            x1: int = max(0, int(b[0]))
            y1: int = max(0, int(b[1]))
            x2: int = min(width, int(b[2]))
            y2: int = min(height, int(b[3]))
            crops.append(np.ascontiguousarray(rgb_hwc[y1:y2, x1:x2]))
        return PersonDetections(boxes_xyxy=boxes, scores=scores, crops=crops)
