from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8

from monopriors.models.metric_depth.moge_v2 import MOGE_V2_CHECKPOINTS
from monopriors.third_party.moge.model._inference import InferenceOutput
from monopriors.third_party.moge.model.v2 import MoGeModel

from .base_normal_model import BaseNormalPredictor, SurfaceNormalPrediction

MOGE_V2_NORMAL_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "vitl": MOGE_V2_CHECKPOINTS[("vitl", True)],
    "vitb": MOGE_V2_CHECKPOINTS[("vitb", True)],
    "vits": MOGE_V2_CHECKPOINTS[("vits", True)],
}
"""Mapping of encoder to Hugging Face repository and revision (all include normals)."""


class MoGeV2NormalPredictor(BaseNormalPredictor[MoGeModel]):
    """MoGe v2 predictor producing surface normals.

    Requires a ``-normal`` checkpoint variant.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        encoder: Literal["vits", "vitb", "vitl"] = "vitl",
    ) -> None:
        """Load a pinned normal-capable MoGe v2 checkpoint.

        Args:
            device: Torch device used for inference.
            encoder: DINOv2 encoder size.
        """
        super().__init__()
        checkpoint: tuple[str, str] = MOGE_V2_NORMAL_CHECKPOINTS[encoder]
        repo_id: str = checkpoint[0]
        revision: str = checkpoint[1]
        print(f"Loading MoGe v2 normal model ({repo_id})...")
        start: float = timer()
        self.model: MoGeModel = MoGeModel.from_pretrained(repo_id, revision=revision).to(device)
        print(f"MoGe v2 normal model loaded. Time: {timer() - start:.2f}s")
        self.device: Literal["cpu", "cuda"] = device

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> SurfaceNormalPrediction:
        """Predict surface normals from an RGB image.

        Args:
            rgb: uint8 RGB image shaped ``h w 3``.
            K_33: Optional float camera intrinsics shaped ``3 3``; normals do not use it.

        Returns:
            Surface normals and per-pixel confidence.
        """
        h: int
        w: int
        h, w, _ = rgb.shape
        input_image: Float[torch.Tensor, "3 h w"] = rearrange(torch.from_numpy(rgb), "h w c -> c h w").to(
            device=self.device,
            dtype=torch.float32,
        )
        input_image.div_(255.0)

        output: InferenceOutput = self.model.infer(
            input_image,
            force_projection=False,
            output_heads=("normal", "mask"),
        )
        # v2 normal checkpoint output includes "normal" (H,W,3) and "mask" (H,W)

        normal_hw3: Float[np.ndarray, "h w 3"] = output["normal"].numpy(force=True)
        mask: Float[np.ndarray, "h w"] = output["mask"].numpy(force=True).astype(np.float32)
        confidence_hw1: Float[np.ndarray, "h w 1"] = mask[:, :, np.newaxis]

        return SurfaceNormalPrediction(
            normal_hw3=normal_hw3,
            confidence_hw1=confidence_hw1,
        )
