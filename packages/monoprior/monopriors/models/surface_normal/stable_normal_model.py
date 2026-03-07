from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, UInt8
from PIL import Image

from monopriors.models.surface_normal.base_normal_model import (
    BaseNormalPredictor,
    SurfaceNormalPrediction,
)


class StableNormalPredictor(BaseNormalPredictor):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        model_type: Literal["StableNormal", "StableNormal_turbo"] = "StableNormal",
    ):
        self.device = device
        self.model = torch.hub.load("Stable-X/StableNormal", model_type, trust_repo=True, device=device)

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> SurfaceNormalPrediction:
        rgb_pil: Image.Image = Image.fromarray(rgb)
        # preprocess the input image
        normal_image: Image.Image = self.model(rgb_pil)
        normal_hw3: np.ndarray = np.array(normal_image).astype(np.float32)
        conf_hw1: np.ndarray = np.ones_like(normal_hw3[..., 0:1])  # Dummy confidence map

        normal_pred = SurfaceNormalPrediction(
            normal_hw3=normal_hw3,
            confidence_hw1=conf_hw1,
        )

        return normal_pred
