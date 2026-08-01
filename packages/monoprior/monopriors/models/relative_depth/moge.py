from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8

from monopriors.depth_utils import depth_to_disparity
from monopriors.third_party.moge.model._inference import InferenceOutput
from monopriors.third_party.moge.model.v1 import MoGeModel

from .base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction

MOGE_V1_CHECKPOINT_ID: str = "Ruicheng/moge-vitl"
MOGE_V1_CHECKPOINT_REVISION: str = "979e84da9415762c30e6c0cf8dc0962896c793df"


class MogeV1Predictor(BaseRelativePredictor[MoGeModel]):
    """MoGe v1 predictor producing scale-invariant (relative) depth."""

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
    ) -> None:
        """Load the pinned MoGe v1 checkpoint.

        Args:
            device: Torch device used for inference.
        """
        super().__init__()
        print("Loading MoGe v1 model...")
        start: float = timer()
        self.model: MoGeModel = MoGeModel.from_pretrained(
            MOGE_V1_CHECKPOINT_ID,
            revision=MOGE_V1_CHECKPOINT_REVISION,
        ).to(device)
        print(f"MoGe v1 model loaded. Time: {timer() - start:.2f}s")
        self.device: Literal["cpu", "cuda"] = device

    def __call__(self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None = None) -> RelativeDepthPrediction:
        """Predict scale-invariant depth from an RGB image.

        Args:
            rgb: uint8 RGB image shaped ``h w 3``.
            K_33: Optional float camera intrinsics shaped ``3 3``; MoGe v1 estimates its own intrinsics.

        Returns:
            Relative depth, disparity, confidence, and estimated intrinsics.
        """
        h: int
        w: int
        h, w, _ = rgb.shape
        input_image: Float[torch.Tensor, "3 h w"] = rearrange(torch.from_numpy(rgb), "h w c -> c h w").to(
            device=self.device,
            dtype=torch.float32,
        )
        input_image.div_(255.0)

        output: InferenceOutput = self.model.infer(input_image, force_projection=False)
        # v1 output keys: "points" (H,W,3), "depth" (H,W), "mask" (H,W), "intrinsics" (3,3)
        # All values are scale-invariant. Intrinsics are normalized.

        normalized_k: Float[np.ndarray, "3 3"] = output["intrinsics"].numpy(force=True)
        fx: float = float(normalized_k[0, 0] * w)
        fy: float = float(normalized_k[1, 1] * h)
        cx: float = float(normalized_k[0, 2] * w)
        cy: float = float(normalized_k[1, 2] * h)

        intrinsics: Float[np.ndarray, "3 3"] = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        relative_depth: Float[np.ndarray, "h w"] = output["depth"].numpy(force=True)
        mask: Float[np.ndarray, "h w"] = output["mask"].numpy(force=True).astype(np.float32)

        return RelativeDepthPrediction(
            disparity=depth_to_disparity(relative_depth, focal_length=int(intrinsics[0, 0])),
            depth=relative_depth,
            confidence=mask,
            K_33=intrinsics,
        )
