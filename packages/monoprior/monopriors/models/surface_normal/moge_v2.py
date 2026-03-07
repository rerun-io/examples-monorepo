from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, UInt8
from moge.model.v2 import MoGeModel
from torch import Tensor

from .base_normal_model import BaseNormalPredictor, SurfaceNormalPrediction

MOGE_V2_NORMAL_CHECKPOINTS: dict[str, str] = {
    "vitl": "Ruicheng/moge-2-vitl-normal",
    "vitb": "Ruicheng/moge-2-vitb-normal",
    "vits": "Ruicheng/moge-2-vits-normal",
}
"""Mapping of encoder to HuggingFace checkpoint name (all include normals)."""


class MoGeV2NormalPredictor(BaseNormalPredictor):
    """MoGe v2 predictor producing surface normals.

    Requires a ``-normal`` checkpoint variant.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        encoder: Literal["vits", "vitb", "vitl"] = "vitl",
    ) -> None:
        super().__init__()
        checkpoint: str = MOGE_V2_NORMAL_CHECKPOINTS[encoder]
        print(f"Loading MoGe v2 normal model ({checkpoint})...")
        start: float = timer()
        self.model = MoGeModel.from_pretrained(checkpoint).to(device)
        print(f"MoGe v2 normal model loaded. Time: {timer() - start:.2f}s")
        self.device: Literal["cpu", "cuda"] = device

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> SurfaceNormalPrediction:
        h: int
        w: int
        h, w, _ = rgb.shape
        input_image: Float[torch.Tensor, "3 h w"] = torch.tensor(
            rgb / 255, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)

        output: dict[str, Tensor] = self.model.infer(input_image)
        # v2 normal checkpoint output includes "normal" (H,W,3) and "mask" (H,W)

        normal_hw3: Float[np.ndarray, "h w 3"] = output["normal"].numpy(force=True)
        mask: Float[np.ndarray, "h w"] = output["mask"].numpy(force=True).astype(np.float32)
        confidence_hw1: Float[np.ndarray, "h w 1"] = mask[:, :, np.newaxis]

        return SurfaceNormalPrediction(
            normal_hw3=normal_hw3,
            confidence_hw1=confidence_hw1,
        )
