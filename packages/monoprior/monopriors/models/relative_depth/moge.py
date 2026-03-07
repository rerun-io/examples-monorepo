from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, UInt8
from moge.model.v1 import MoGeModel
from torch import Tensor

from monopriors.depth_utils import depth_to_disparity

from .base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction


class MogeV1Predictor(BaseRelativePredictor):
    """MoGe v1 predictor producing scale-invariant (relative) depth."""

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
    ) -> None:
        super().__init__()
        print("Loading MoGe v1 model...")
        start: float = timer()
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        print(f"MoGe v1 model loaded. Time: {timer() - start:.2f}s")
        self.device: Literal["cpu", "cuda"] = device

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None = None
    ) -> RelativeDepthPrediction:
        h: int
        w: int
        h, w, _ = rgb.shape
        input_image: Float[torch.Tensor, "3 h w"] = torch.tensor(
            rgb / 255, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)

        output: dict[str, Tensor] = self.model.infer(input_image)
        # v1 output keys: "points" (H,W,3), "depth" (H,W), "mask" (H,W), "intrinsics" (3,3)
        # All values are scale-invariant. Intrinsics are normalized.

        normalized_k: Float[np.ndarray, "3 3"] = output["intrinsics"].numpy(force=True)
        fx: float = float(normalized_k[0, 0] * w)
        fy: float = float(normalized_k[1, 1] * h)
        cx: float = float(normalized_k[0, 2] * w)
        cy: float = float(normalized_k[1, 2] * h)

        intrinsics: Float[np.ndarray, "3 3"] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        relative_depth: Float[np.ndarray, "h w"] = output["depth"].numpy(force=True)
        mask: Float[np.ndarray, "h w"] = output["mask"].numpy(force=True).astype(np.float32)

        return RelativeDepthPrediction(
            disparity=depth_to_disparity(relative_depth, focal_length=int(intrinsics[0, 0])),
            depth=relative_depth,
            confidence=mask,
            K_33=intrinsics,
        )
