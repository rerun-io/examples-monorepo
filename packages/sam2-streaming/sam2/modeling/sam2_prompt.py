from __future__ import annotations

import torch


class SAM2Prompt:
    def __init__(
        self,
        obj_id: int,
        points_coords: torch.Tensor | None = None,
        points_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        masks_logits: torch.Tensor | None = None,
        is_normalized: bool = False,
    ):
        if (
            points_coords is None
            and points_labels is None
            and boxes is None
            and masks_logits is None
        ):
            raise ValueError(
                "At least one of points_coords, points_labels, boxes, or masks_logits must be provided"
            )

        if points_coords is not None and points_labels is None:
            raise ValueError(
                "points_labels must be provided if points_coords is provided"
            )
        
        if points_coords is not None and (points_coords.ndim != 2 or points_coords.shape[1] != 2):
            raise ValueError(f"Expected points_coords to be of shape (N, 2), got {points_coords.shape}")
        
        if points_labels is not None and (points_labels.ndim != 1 or points_labels.shape[0] != points_coords.shape[0]):
            raise ValueError(f"Expected points_labels to be of shape (N,), got {points_labels.shape}")
        
        if boxes is not None and (boxes.ndim != 2 or boxes.shape[1] != 4):
            raise ValueError(f"Expected boxes to be of shape (N, 4), got {boxes.shape}")
        
        if masks_logits is not None and (masks_logits.ndim != 3 or masks_logits.shape[1] != masks_logits.shape[2]):
            raise ValueError(f"Expected masks_logits to be of shape (N, H, W), got {masks_logits.shape}")

        self.obj_id = obj_id
        self.points_coords = points_coords
        self.points_labels = points_labels
        self.boxes = boxes
        self.masks_logits = masks_logits
        self.is_normalized = is_normalized

    def to(self, device: torch.device) -> SAM2Prompt:
        points_coords = (
            self.points_coords.to(device) if self.points_coords is not None else None
        )
        points_labels = (
            self.points_labels.to(device) if self.points_labels is not None else None
        )
        boxes = self.boxes.to(device) if self.boxes is not None else None
        masks_logits = (
            self.masks_logits.to(device) if self.masks_logits is not None else None
        )
        return SAM2Prompt(
            obj_id=self.obj_id,
            points_coords=points_coords,
            points_labels=points_labels,
            boxes=boxes,
            masks_logits=masks_logits,
            is_normalized=self.is_normalized,
        )

    def clone(self) -> SAM2Prompt:
        return SAM2Prompt(
            obj_id=self.obj_id,
            points_coords=self.points_coords.clone() if self.points_coords is not None else None,
            points_labels=self.points_labels.clone() if self.points_labels is not None else None,
            boxes=self.boxes.clone() if self.boxes is not None else None,
            masks_logits=self.masks_logits.clone() if self.masks_logits is not None else None,
            is_normalized=self.is_normalized,
        )