"""SAM3 single-image instance segmentation predictor."""

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray
from serde import serde
from torch import Tensor
from transformers.models.sam3 import Sam3Model, Sam3Processor


class SAM3ResultsDict(TypedDict):
    """Torch-format outputs returned directly by ``Sam3Processor`` post-processing."""

    scores: Float32[Tensor, "n"]
    boxes: Float32[Tensor, "n 4"]
    masks: Float32[Tensor, "n h w"]


@serde()
class SAM3Results:
    scores: Float32[ndarray, "n"]
    """Per-instance confidence scores ``[N]``."""
    boxes: Float32[ndarray, "n 4"]
    """Bounding boxes in XYXY pixel coordinates ``[N, 4]``."""
    masks: Float32[ndarray, "n h w"]
    """Probability masks for each detection ``[N, H, W]`` (float32 in ``[0, 1]``)."""


@dataclass
class SAM3Config:
    """Configuration for loading a SAM3 checkpoint and selecting device."""

    device: Literal["cpu", "cuda"] = "cuda"
    """Computation device passed to the Hugging Face SAM3 model."""
    sam3_checkpoint: str = "facebook/sam3"
    """Model identifier or path accepted by ``Sam3Model.from_pretrained``."""


class SAM3Predictor:
    """Lightweight wrapper around the SAM3 model for single-image inference."""

    def __init__(self, config: SAM3Config):
        self.config = config
        self.sam3_model = Sam3Model.from_pretrained(config.sam3_checkpoint).to(config.device)
        self.sam3_processor = Sam3Processor.from_pretrained(config.sam3_checkpoint)

    def predict_single_image(self, rgb_hw3: UInt8[ndarray, "h w 3"], text: str = "person") -> SAM3Results:
        """Run SAM3 instance segmentation on one RGB image.

        Args:
            rgb_hw3: Input image in RGB order with dtype ``uint8`` and shape ``[H, W, 3]``.
            text: Optional prompt used by SAM3's text-conditioned decoder (default: ``"person"``).

        Returns:
            ``SAM3Results`` with NumPy copies of scores, XYXY boxes, and binary masks.
        """
        inputs = self.sam3_processor(
            images=rgb_hw3,
            text=text,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.sam3_model(**inputs)

        results: SAM3ResultsDict = self.sam3_processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        mask_probs: Float32[ndarray, "n h w"] = results["masks"].detach().cpu().numpy().astype(np.float32, copy=False)

        return SAM3Results(
            scores=results["scores"].detach().cpu().numpy().astype(np.float32, copy=False),
            boxes=results["boxes"].detach().cpu().numpy().astype(np.float32, copy=False),
            masks=mask_probs,
        )
