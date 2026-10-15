"""YOLOv12-X person detector (Ultralytics) in posekit's ``PersonDetector`` role.

Ported from the original ``segmentation/core/pipeline.py``: confidence 0.5, class 0 (person) only, dedup at IoU 0.75
keeping the higher-confidence box (``batched_nms`` drops IoU > 0.75, the original >= 0.75: only an exact tie differs).
Detection runs sparsely (bootstrap and re-detect ticks), so frames take one host hop.
"""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32, Int64, UInt8
from posekit.models import PersonDetector
from posekit.predictions import BoxDetections
from torch import Tensor
from torchvision.ops import batched_nms


class Yolo12xDetector(PersonDetector):
    def __init__(self, weights_path: Path, confidence: float = 0.5, dedup_iou: float = 0.75, device: str = "cuda") -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path), verbose=False)
        self.model.to(device)
        self.confidence, self.dedup_iou = confidence, dedup_iou

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> BoxDetections:
        """Every frame's person boxes, highest confidence first within a frame."""
        boxes: list[Float32[Tensor, "m 4"]] = []
        scores: list[Float32[Tensor, " m"]] = []
        frame_indices: list[Int64[Tensor, " m"]] = []
        for index, frame in enumerate(frames_rgb.cpu().numpy()):
            found = self.model(np.ascontiguousarray(frame[:, :, ::-1]), verbose=False)[0].boxes  # Ultralytics reads BGR arrays
            person: Tensor = (found.cls == 0) & (found.conf >= self.confidence)
            boxes.append(found.xyxy[person].float())
            scores.append(found.conf[person].float())
            frame_indices.append(torch.full((int(person.sum()),), index, dtype=torch.int64, device=person.device))
        xyxy, conf, frame_index = torch.cat(boxes), torch.cat(scores), torch.cat(frame_indices)
        keep: Int64[Tensor, " k"] = batched_nms(xyxy, conf, frame_index, self.dedup_iou)  # highest confidence first
        keep = keep[torch.argsort(frame_index[keep], stable=True)]
        device: torch.device = frames_rgb.device
        return BoxDetections(xyxy=xyxy[keep].to(device), scores=conf[keep].to(device), frame_indices=frame_index[keep].to(device))
