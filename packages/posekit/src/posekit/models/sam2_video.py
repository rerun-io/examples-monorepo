"""Streaming video instance segmentation over the sam2-streaming fork.

Wraps ``SAM2GenericVideoPredictor`` (our forward-only fork with the forgetful
causal memory bank — the engine behind mamma's tracker) behind the posekit
:class:`~posekit.models.base.VideoSegmenter` role: one memory state per batch
slot (camera/stream), box or point prompts to start tracks, and per-step
detections with GPU masks and stable ``track_ids``.
"""

from __future__ import annotations

import functools
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


@functools.cache
def cached_predictor(variant: Sam2Variant):
    """Load one SAM2-streaming predictor per variant."""
    return Sam2VideoSegmenterConfig(variant=variant).setup().predictor


@dataclass(frozen=True, slots=True)
class Sam2VideoSegmenterConfig:
    """Streaming SAM2/EfficientTAM video segmenter configuration."""

    variant: Sam2Variant = "efficienttam-ti-512"
    """Model architecture + checkpoint (-ti@512 is mamma's speed pick, -s@512 is Kineo's)."""
    device: str = "cuda"
    """Inference device."""
    memory_window_size: int = 7
    """Sliding window (frames) of non-prompt memories kept per tracked object."""
    memory_on_device: bool = False
    """Keep the memory bank on the inference device instead of host memory (no per-step copies)."""
    bf16: bool = False
    """Run the forward pass under bfloat16 autocast (the fp32 memory attention dominates the step time)."""
    max_conditional_memories: int = -1
    """Prompted-frame memories each frame attends to, the temporally nearest (-1: all). Bounded, the bank also stores
    only those (:class:`~posekit.models.sam2_memory_bank.BoundedForgetfulMemoryBank`), so a tracker re-prompted
    through a long video keeps a constant per-frame cost and memory."""
    single_click_multimask: bool = False
    """A prompt of one click per object takes the best of SAM 2's three masks by predicted IoU (SAM 2's
    ``_use_multimask`` rule: a lone click is ambiguous, shirt or person); off, it takes the single mask."""

    fused_rope: bool = True
    """Rotary encodings in one Triton kernel per tensor (sam2-streaming ``fused_rope``, same masks); needs Triton."""

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
        self.predictor.max_cond_frames_in_attn = config.max_conditional_memories
        if config.fused_rope:
            from sam2.modeling import fused_rope

            # A module switch of the fork, read by every RoPEAttention: on for all predictors in the process.
            fused_rope.ENABLED = fused_rope.triton is not None
        self._states: list[SAM2GenericVideoPredictorState] | None = None
        self._frame_idx: int = 0

    def reset(self) -> None:
        """Drop all tracked instances and memory state."""
        self._states = None
        self._frame_idx = 0

    def drop_tracks(self, slot: int, track_ids: list[int]) -> None:
        """Forget tracks of one batch slot (memories and identity); unknown ids are ignored."""
        from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank

        if self._states is None:
            return
        bank = self._states[slot].memory_bank
        assert isinstance(bank, SAM2ForgetfulObjectMemoryBank)
        for track_id in track_ids:
            bank.known_obj_ids.discard(track_id)
            bank.conditional_memories.pop(track_id, None)
            bank.non_conditional_memories.pop(track_id, None)

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
        from sam2.modeling.sam2_result import SAM2Result
        from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

        from posekit.models.sam2_memory_bank import BoundedForgetfulMemoryBank

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
                    memory_bank=SAM2ForgetfulObjectMemoryBank(
                        memory_window_size=self.config.memory_window_size,
                        storage_device=frames_rgb.device if self.config.memory_on_device else torch.device("cpu"),
                    )
                    if self.config.max_conditional_memories < 0
                    else BoundedForgetfulMemoryBank(self.config.max_conditional_memories, self.config.memory_window_size),
                )
                for _ in range(batch_size)
            ]
        if len(self._states) != batch_size:
            raise ValueError(f"Batch size changed between steps: state has {len(self._states)} slots, got {batch_size} frames.")

        device: torch.device = frames_rgb.device
        slot_prompts: list[list[SAM2Prompt]] = [self._prompts_for_slot(prompts, slot) if prompts is not None else [] for slot in range(batch_size)]
        single_clicks: list[bool] = [
            self.config.single_click_multimask
            and bool(sp)
            and all(p.boxes is None and p.points_coords is not None and int(p.points_coords.shape[0]) == 1 for p in sp)
            for sp in slot_prompts
        ]
        # One predictor pass for every slot and object (grouped batches per module, see forward_batched).
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.bf16):
            per_slot = self.predictor.forward_batched(self._states, self._frame_idx, frames_rgb.permute(0, 3, 1, 2), slot_prompts, single_clicks)
        self._frame_idx += 1
        rows_out: list[tuple[int, int, SAM2Result]] = [
            (slot, obj_id, result) for slot, results in enumerate(per_slot) for obj_id, result in results.items()
        ]
        if not rows_out:
            return BoxDetections.empty(device, mask_hw=(frame_h, frame_w), with_track_ids=True)
        masks: Bool[Tensor, "n h w"] = torch.cat([result.masks_logits[:, 0] for _, _, result in rows_out]) > 0.0
        # Branch-free mask bounding boxes: argmax of the row/column projections finds the first True; zeros when fully
        # occluded. Keeps the streaming loop free of GPU->CPU syncs.
        cols: Bool[Tensor, "n w"] = masks.any(dim=1)
        rows: Bool[Tensor, "n h"] = masks.any(dim=2)
        corners: list[Tensor] = [
            cols.int().argmax(dim=1),
            rows.int().argmax(dim=1),
            frame_w - cols.flip(1).int().argmax(dim=1),
            frame_h - rows.flip(1).int().argmax(dim=1),
        ]
        boxes: Float[Tensor, "n 4"] = torch.stack(corners, dim=1).float()
        return BoxDetections(
            xyxy=torch.where(cols.any(dim=1, keepdim=True), boxes, 0.0),
            scores=torch.cat([result.ious.max(dim=1).values for _, _, result in rows_out]).float(),
            frame_indices=torch.tensor([slot for slot, _, _ in rows_out], dtype=torch.long, device=device),
            masks=masks,
            track_ids=torch.tensor([obj_id for _, obj_id, _ in rows_out], dtype=torch.long, device=device),
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
            masks_logits: Float[Tensor, "1 h w"] | None = None
            if prompts.masks is not None:
                # binary mask -> saturated logits; SAM2 resizes the dense prompt to its 256 px mask grid
                masks_logits = torch.where(prompts.masks[row : row + 1], 10.0, -10.0)
            sam2_prompts.append(
                SAM2Prompt(obj_id=track_id, points_coords=points_xy, points_labels=points_labels, boxes=boxes, masks_logits=masks_logits)
            )
        return sam2_prompts


__all__ = ("Sam2VideoSegmenter", "Sam2VideoSegmenterConfig", "cached_predictor")
