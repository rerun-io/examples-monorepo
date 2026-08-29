"""Streaming video instance segmentation over the sam2-streaming fork.

Wraps ``SAM2GenericVideoPredictor`` (our forward-only fork with the forgetful
causal memory bank — the engine behind mamma's tracker) behind the posekit
:class:`~posekit.models.base.VideoSegmenter` role: one memory state per batch
slot (camera/stream), box or point prompts to start tracks, and per-step
detections with GPU masks and stable ``track_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from jaxtyping import Bool, Float, UInt8
from torch import Tensor

from posekit.models.base import SegmentationPrompts, VideoSegmenter
from posekit.predictions import BoxDetections, validate_frames_rgb

Sam2Variant: TypeAlias = Literal["efficienttam-ti-512", "efficienttam-s-512"]

SAM2_VARIANT_CONFIGS: dict[Sam2Variant, str] = {
    "efficienttam-ti-512": "configs/efficienttam/efficienttam_ti_512x512.yaml",
    "efficienttam-s-512": "configs/efficienttam/efficienttam_s_512x512.yaml",
}
# (repo_id, repo_type, filename) per variant; -s is Kineo's pick, -ti is mamma's speed pick.
SAM2_VARIANT_WEIGHTS: dict[Sam2Variant, tuple[str, str, str]] = {
    "efficienttam-ti-512": ("pablovela5620/mamma-streaming-data", "dataset", "weights/efficienttam/efficienttam_ti.pt"),
    "efficienttam-s-512": ("yunyangx/efficient-track-anything", "model", "efficienttam_s_512x512.pt"),
}


@dataclass(frozen=True, slots=True)
class Sam2VideoSegmenterConfig:
    """Streaming SAM2/EfficientTAM video segmenter configuration."""

    variant: Sam2Variant = "efficienttam-ti-512"
    """Model architecture + checkpoint (-ti@512 is mamma's speed pick, -s@512 is Kineo's)."""
    device: str = "cuda"
    """Inference device."""
    memory_window_size: int = 7
    """Sliding window (frames) of non-prompt memories kept per tracked object."""

    def setup(self) -> "Sam2VideoSegmenter":
        """Download the checkpoint, build the predictor, and return a ready segmenter."""
        return Sam2VideoSegmenter(self)


class Sam2VideoSegmenter(VideoSegmenter):
    """Stateful GPU mask tracker; one causal memory state per batch slot."""

    def __init__(self, config: Sam2VideoSegmenterConfig) -> None:
        """Build the streaming predictor and download weights.

        Args:
            config: Variant, device, and memory-window options.
        """
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2_generic_video_predictor
        from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

        self.config: Sam2VideoSegmenterConfig = config
        repo_id, repo_type, filename = SAM2_VARIANT_WEIGHTS[config.variant]
        checkpoint_path: str = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename)
        self.predictor = build_sam2_generic_video_predictor(SAM2_VARIANT_CONFIGS[config.variant], checkpoint_path, device=config.device)
        self._states: list[SAM2GenericVideoPredictorState] | None = None
        self._frame_idx: int = 0

    def reset(self) -> None:
        """Drop all tracked instances and memory state."""
        self._states = None
        self._frame_idx = 0

    @torch.inference_mode()
    def step(self, frames_rgb: UInt8[Tensor, "b h w 3"], prompts: SegmentationPrompts | None = None) -> BoxDetections:
        """Advance the tracker by one timestep.

        Args:
            frames_rgb: One frame per stream/view at the current timestep; the
                batch layout must stay identical across steps.
            prompts: Box/point prompts starting (or re-anchoring) tracks;
                ``track_ids`` selects the identity per prompt row (row order
                is the fallback id). ``None`` propagates existing tracks.

        Returns:
            Detections with ``masks``/``track_ids`` populated; boxes are the
            mask bounding boxes (zeros while an object is fully occluded).

        Raises:
            ValueError: If the batch size changes between steps or a text
                prompt is given (SAM2 has no text head).
        """
        from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank
        from sam2.modeling.sam2_prompt import SAM2Prompt
        from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

        validate_frames_rgb(frames_rgb)
        if prompts is not None and prompts.text is not None:
            raise ValueError("Sam2VideoSegmenter is visually prompted; text prompts are not supported.")
        batch_size: int = int(frames_rgb.shape[0])
        frame_h: int = int(frames_rgb.shape[1])
        frame_w: int = int(frames_rgb.shape[2])
        if self._states is None:
            self._states = [
                SAM2GenericVideoPredictorState.create(
                    video_hw=(frame_h, frame_w),
                    memory_bank=SAM2ForgetfulObjectMemoryBank(memory_window_size=self.config.memory_window_size),
                )
                for _ in range(batch_size)
            ]
        if len(self._states) != batch_size:
            raise ValueError(f"Batch size changed between steps: state has {len(self._states)} slots, got {batch_size} frames.")

        device: torch.device = frames_rgb.device
        xyxy_rows: list[Tensor] = []
        score_rows: list[Tensor] = []
        mask_rows: list[Tensor] = []
        slot_ids: list[int] = []
        track_ids: list[int] = []
        for slot in range(batch_size):
            slot_prompts: list[SAM2Prompt] = self._prompts_for_slot(prompts, slot) if prompts is not None else []
            frame_chw: UInt8[Tensor, "3 h w"] = frames_rgb[slot].permute(2, 0, 1).contiguous()
            results = self.predictor.forward(self._states[slot], self._frame_idx, frame_chw, prompts=slot_prompts, multimask_output=False)
            for obj_id, result in results.items():
                mask: Bool[Tensor, "h w"] = result.best_mask_logits[0, 0] > 0.0
                # Branch-free mask bounding box: argmax of the row/column
                # projections finds the first True; zeros when fully occluded.
                # Keeps the streaming loop free of per-object GPU->CPU syncs.
                cols: Bool[Tensor, "w"] = mask.any(dim=0)
                rows: Bool[Tensor, "h"] = mask.any(dim=1)
                x0: Tensor = cols.int().argmax()
                y0: Tensor = rows.int().argmax()
                x1: Tensor = frame_w - cols.flip(0).int().argmax()
                y1: Tensor = frame_h - rows.flip(0).int().argmax()
                box: Float[Tensor, "4"] = torch.where(cols.any(), torch.stack([x0, y0, x1, y1]).float(), torch.zeros(4, device=device))
                xyxy_rows.append(box)
                score_rows.append(result.ious.max().float())
                mask_rows.append(mask)
                slot_ids.append(slot)
                track_ids.append(int(obj_id))
        self._frame_idx += 1
        if not xyxy_rows:
            return BoxDetections.empty(device, mask_hw=(frame_h, frame_w), with_track_ids=True)
        return BoxDetections(
            xyxy=torch.stack(xyxy_rows),
            scores=torch.stack(score_rows),
            frame_indices=torch.tensor(slot_ids, dtype=torch.long, device=device),
            masks=torch.stack(mask_rows),
            track_ids=torch.tensor(track_ids, dtype=torch.long, device=device),
        )

    def _prompts_for_slot(self, prompts: SegmentationPrompts, slot: int) -> list:
        """Convert the prompt rows targeting one batch slot into SAM2 prompts."""
        from sam2.modeling.sam2_prompt import SAM2Prompt

        rows: list[int] = torch.where(prompts.frame_indices == slot)[0].tolist()
        sam2_prompts: list[SAM2Prompt] = []
        for row in rows:
            track_id: int = int(prompts.track_ids[row]) if prompts.track_ids is not None else row
            boxes: Float[Tensor, "1 4"] | None = None
            points_xy: Float[Tensor, "1 2"] | None = None
            points_labels: Tensor | None = None
            if prompts.boxes_xyxy is not None:
                boxes = prompts.boxes_xyxy[row : row + 1].float()
            if prompts.points_xy is not None:
                points_xy = prompts.points_xy[row : row + 1].float()
                points_labels = torch.ones((1,), dtype=torch.long, device=points_xy.device)
            sam2_prompts.append(SAM2Prompt(obj_id=track_id, points_coords=points_xy, points_labels=points_labels, boxes=boxes))
        return sam2_prompts


__all__ = ("Sam2VideoSegmenter", "Sam2VideoSegmenterConfig")
