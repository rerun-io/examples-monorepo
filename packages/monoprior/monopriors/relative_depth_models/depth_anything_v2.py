from timeit import default_timer as timer
from typing import Literal

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Float32, UInt8

from monopriors.depth_utils import disparity_to_depth, estimate_intrinsics
from monopriors.third_party.depth_anything_v2.dpt import DepthAnythingV2

from .base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}
encoder2name: dict[str, str] = {
    "vits": "Small",
    "vitb": "Base",
    "vitl": "Large",
    "vitg": "Giant",
}


class DepthAnythingV2Predictor(BaseRelativePredictor):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        encoder: Literal["vits", "vitb", "vitl"] = "vits",
    ) -> None:
        super().__init__()
        print("Loading DepthAnythingV2 model...")
        start = timer()
        model_name: str = encoder2name[encoder]
        self.model = DepthAnythingV2(**model_configs[encoder])
        filepath: str = hf_hub_download(
            repo_id=f"depth-anything/Depth-Anything-V2-{model_name}",
            filename=f"depth_anything_v2_{encoder}.pth",
            repo_type="model",
        )
        state_dict = torch.load(filepath, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model: DepthAnythingV2 = self.model.to(device).eval()
        print(f"DepthAnythingV2 model loaded. Time: {timer() - start:.2f}s")

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> RelativeDepthPrediction:
        # requires bgr not rgb
        bgr: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        disparity: Float32[np.ndarray, "h w"] = self.model.infer_image(bgr)

        if K_33 is None:
            K_33 = estimate_intrinsics(rgb.shape[0], rgb.shape[1])

        relative_prediction = RelativeDepthPrediction(
            disparity=disparity,
            depth=disparity_to_depth(disparity, focal_length=int(K_33[0, 0])),
            confidence=np.ones_like(disparity),
            K_33=K_33,
        )

        return relative_prediction
