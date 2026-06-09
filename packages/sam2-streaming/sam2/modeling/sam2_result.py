from __future__ import annotations

import torch


class SAM2Result:
    def __init__(
        self,
        masks_logits: torch.Tensor,
        ious: torch.Tensor,
        obj_ptrs: torch.Tensor,
        obj_scores_logits: torch.Tensor,
    ):
        assert (
            masks_logits.ndim == 4
        ), f"Expected masks_logits to be of shape (B, N, H, W), got {masks_logits.shape}"
        assert ious.ndim == 2, f"Expected ious to be of shape (B, N), got {ious.shape}"
        assert (
            obj_ptrs.ndim == 2
        ), f"Expected obj_ptrs to be of shape (B, N), got {obj_ptrs.shape}"
        assert (
            obj_scores_logits.ndim == 2
        ), f"Expected obj_score_logits to be of shape (B, N), got {obj_scores_logits.shape}"

        self.batch_size = masks_logits.shape[0]

        assert (
            ious.shape[0] == self.batch_size
        ), f"Expected ious to have batch size {self.batch_size}, got {ious.shape[0]}"
        assert (
            obj_ptrs.shape[0] == self.batch_size
        ), f"Expected obj_ptrs to have batch size {self.batch_size}, got {obj_ptrs.shape[0]}"
        assert (
            obj_scores_logits.shape[0] == self.batch_size
        ), f"Expected obj_score_logits to have batch size {self.batch_size}, got {obj_scores_logits.shape[0]}"

        self.masks_logits = masks_logits
        self.ious = ious
        self.obj_ptrs = obj_ptrs
        self.obj_score_logits = obj_scores_logits

    def to(self, device: torch.device) -> SAM2Result:
        return SAM2Result(
            masks_logits=self.masks_logits.to(device),
            ious=self.ious.to(device),
            obj_ptrs=self.obj_ptrs.to(device),
            obj_scores_logits=self.obj_score_logits.to(device),
        )

    def clone(self) -> SAM2Result:
        return SAM2Result(
            masks_logits=self.masks_logits.clone(),
            ious=self.ious.clone(),
            obj_ptrs=self.obj_ptrs.clone(),
            obj_scores_logits=self.obj_score_logits.clone(),
        )

    @staticmethod
    def cat(results: list[SAM2Result]) -> SAM2Result:
        return SAM2Result(
            masks_logits=torch.cat([r.masks_logits for r in results], dim=0),
            ious=torch.cat([r.ious for r in results], dim=0),
            obj_ptrs=torch.cat([r.obj_ptrs for r in results], dim=0),
            obj_scores_logits=torch.cat([r.obj_score_logits for r in results], dim=0),
        )

    def __getitem__(self, idx: int) -> SAM2Result:
        return SAM2Result(
            masks_logits=self.masks_logits[idx].unsqueeze(0),
            ious=self.ious[idx].unsqueeze(0),
            obj_ptrs=self.obj_ptrs[idx].unsqueeze(0),
            obj_scores_logits=self.obj_score_logits[idx].unsqueeze(0),
        )

    @property
    def device(self) -> torch.device:
        return self.masks_logits.device

    @property
    def best_mask_logits(self) -> torch.Tensor:
        best_mask_idx = torch.argmax(self.ious, dim=1, keepdim=True)
        batch_indices = torch.arange(
            self.masks_logits.shape[0], device=self.masks_logits.device
        ).unsqueeze(1)
        return self.masks_logits[batch_indices, best_mask_idx]
