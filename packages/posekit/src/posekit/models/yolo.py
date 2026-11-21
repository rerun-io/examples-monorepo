"""Ultralytics YOLO person detector (YOLOv12-X by default), the detector mamma's tracker used.

Runs Ultralytics' own pre/postprocessing per frame on a BGR host copy (its letterbox and NMS), keeps the person
class above ``score_thr`` and suppresses boxes overlapping a higher-scored one at ``dedup_iou`` or more, as
mamma's port of the original MAMMA segmentation stage did. Detection runs sparsely (bootstrap, periodic
re-detects), so the per-frame host copy is acceptable; TensorRT is not wired.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision
from jaxtyping import UInt8
from torch import Tensor

from posekit.models.base import PersonDetector
from posekit.predictions import BoxDetections, validate_frames_rgb


@dataclass(frozen=True, slots=True)
class YoloDetectorConfig:
    """Ultralytics YOLO person detector configuration."""

    weights: Path | str = "yolo12x.pt"
    """Checkpoint path, or an Ultralytics model name it resolves (and downloads)."""
    score_thr: float = 0.5
    """Minimum person confidence."""
    dedup_iou: float = 0.75
    """A box overlapping a higher-scored kept box by this IoU or more is dropped."""
    device: str = "cuda"
    """Inference device."""

    def setup(self) -> "YoloDetector":
        """Load the checkpoint and return a ready detector."""
        return YoloDetector(self)


class YoloDetector(PersonDetector):
    """Ultralytics YOLO person detector."""

    def __init__(self, config: YoloDetectorConfig) -> None:
        from ultralytics import YOLO

        self.config: YoloDetectorConfig = config
        self.model = YOLO(str(config.weights), verbose=False)
        self.model.to(config.device)

    @torch.inference_mode()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> BoxDetections:
        """Detect persons across a frame batch; boxes per frame come sorted by descending score."""
        validate_frames_rgb(frames_rgb)
        device: torch.device = frames_rgb.device
        boxes: list[Tensor] = []
        scores: list[Tensor] = []
        frame_indices: list[Tensor] = []
        for index, frame in enumerate(frames_rgb.cpu().numpy()):
            # Ultralytics reads numpy arrays as BGR; one image gives one Results (loosely typed upstream).
            results: Any = self.model(frame[:, :, ::-1], verbose=False)
            result: Any = results[0]
            assert result.boxes is not None, "YOLO detection returns boxes"
            keep: Tensor = (result.boxes.cls == 0) & (result.boxes.conf >= self.config.score_thr)
            xyxy: Tensor = result.boxes.xyxy[keep].float()
            conf: Tensor = result.boxes.conf[keep].float()
            # torchvision's nms suppresses IoU > threshold; mamma's dedup dropped IoU >= threshold (equal only on exact ties).
            kept: Tensor = torchvision.ops.nms(xyxy, conf, self.config.dedup_iou)
            boxes.append(xyxy[kept].to(device))
            scores.append(conf[kept].to(device))
            frame_indices.append(torch.full((int(kept.shape[0]),), index, dtype=torch.long, device=device))
        if not boxes:
            return BoxDetections.empty(device)
        return BoxDetections(xyxy=torch.cat(boxes), scores=torch.cat(scores), frame_indices=torch.cat(frame_indices))
