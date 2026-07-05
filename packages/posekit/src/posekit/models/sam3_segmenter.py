"""SAM3 open-vocabulary promptable instance segmentation (text prompts).

Loads ``facebook/sam3`` through transformers as a weights + ``nn.Module``
source (docs/design.md §4) and replaces the CPU image processor with GPU
preprocessing: SAM3's pipeline is a plain non-aspect-preserving bilinear
resize to 1008x1008 plus 0.5/0.5 normalization, so CUDA frames go straight
into the model without host copies. Only the tiny text tokenization runs on
CPU, once per call.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int64, UInt8
from torch import Tensor

from posekit.models.base import PromptableSegmenter, SegmentationPrompts
from posekit.predictions import BoxDetections, validate_frames_rgb


@dataclass(frozen=True, slots=True)
class Sam3SegmenterConfig:
    """SAM3 text-prompted instance segmentation configuration (torch backend)."""

    checkpoint: str = "facebook/sam3"
    """HF checkpoint accepted by ``Sam3Model.from_pretrained``."""
    device: str = "cuda"
    """Inference device."""
    score_threshold: float = 0.5
    """Minimum instance confidence kept in the output."""
    mask_threshold: float = 0.5
    """Probability threshold binarizing the output masks."""

    def setup(self) -> "Sam3Segmenter":
        """Download weights and return a ready segmenter."""
        return Sam3Segmenter(self)


class Sam3Segmenter(PromptableSegmenter):
    """Batched GPU SAM3 segmenter returning detections with instance masks.

    SAM3 is concept-prompted: it segments *every* instance matching the text
    prompt, so box/point prompts are not supported by this implementation
    (SAM2 image mode is the planned box/point ``PromptableSegmenter``).
    """

    def __init__(self, config: Sam3SegmenterConfig) -> None:
        """Load the SAM3 model and tokenizer.

        Args:
            config: Checkpoint, device, and thresholds.
        """
        from transformers.models.sam3 import Sam3Model, Sam3Processor

        self.config: Sam3SegmenterConfig = config
        self.model = Sam3Model.from_pretrained(config.checkpoint).to(config.device).eval()
        self.processor = Sam3Processor.from_pretrained(config.checkpoint)
        size = self.processor.image_processor.size
        self.input_hw: tuple[int, int] = (int(size["height"]), int(size["width"]))

    @torch.inference_mode()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], prompts: SegmentationPrompts) -> BoxDetections:
        """Segment every instance matching the text prompt in each frame.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            prompts: Must carry ``text``; box/point prompts are rejected.

        Returns:
            Detections with binary ``masks`` at frame resolution; ``track_ids``
            stays ``None`` (text prompts carry no identities).

        Raises:
            ValueError: If ``prompts.text`` is missing or box/point prompts are given.
        """
        if prompts.text is None:
            raise ValueError("Sam3Segmenter requires a text prompt (prompts.text).")
        if prompts.boxes_xyxy is not None or prompts.points_xy is not None:
            raise ValueError("Sam3Segmenter is concept-prompted; box/point prompts are not supported.")
        validate_frames_rgb(frames_rgb)
        batch_size: int = int(frames_rgb.shape[0])
        frame_h: int = int(frames_rgb.shape[1])
        frame_w: int = int(frames_rgb.shape[2])

        # GPU replication of Sam3ImageProcessor: bilinear resize to 1008x1008 on
        # [0, 1] floats (antialias matches PIL), then (x - 0.5) / 0.5.
        chw: Float[Tensor, "b 3 h w"] = frames_rgb.permute(0, 3, 1, 2).float() / 255.0
        resized: Float[Tensor, "b 3 in_h in_w"] = F.interpolate(chw, size=self.input_hw, mode="bilinear", align_corners=False, antialias=True)
        pixel_values: Float[Tensor, "b 3 in_h in_w"] = (resized - 0.5) / 0.5

        text_inputs = self.processor.tokenizer([prompts.text], return_tensors="pt").to(pixel_values.device)
        input_ids: Int64[Tensor, "1 t"] = text_inputs["input_ids"]
        attention_mask: Int64[Tensor, "1 t"] = text_inputs["attention_mask"]
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids.expand(batch_size, -1),
            attention_mask=attention_mask.expand(batch_size, -1),
        )
        per_frame: list[dict[str, Tensor]] = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.config.score_threshold,
            mask_threshold=self.config.mask_threshold,
            target_sizes=[(frame_h, frame_w)] * batch_size,
        )
        device: torch.device = frames_rgb.device
        xyxy_rows: list[Tensor] = []
        score_rows: list[Tensor] = []
        index_rows: list[Tensor] = []
        mask_rows: list[Tensor] = []
        for frame_idx, results in enumerate(per_frame):
            num: int = int(results["scores"].shape[0])
            if num == 0:
                continue
            xyxy_rows.append(results["boxes"].to(device=device, dtype=torch.float32))
            score_rows.append(results["scores"].to(device=device, dtype=torch.float32))
            index_rows.append(torch.full((num,), frame_idx, dtype=torch.long, device=device))
            mask_rows.append(results["masks"].to(device=device) > self.config.mask_threshold)
        if not xyxy_rows:
            return BoxDetections.empty(device, mask_hw=(frame_h, frame_w))
        masks: Bool[Tensor, "n h w"] = torch.cat(mask_rows, dim=0)
        return BoxDetections(
            xyxy=torch.cat(xyxy_rows, dim=0),
            scores=torch.cat(score_rows, dim=0),
            frame_indices=torch.cat(index_rows, dim=0),
            masks=masks,
        )


__all__ = ("Sam3Segmenter", "Sam3SegmenterConfig")
