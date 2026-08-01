from timeit import default_timer as timer
from typing import Literal, TypedDict

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, Float32, UInt8
from PIL import Image
from transformers import pipeline

from monopriors.depth_utils import disparity_to_depth, estimate_intrinsics

from .base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction


class DepthDict(TypedDict):
    predicted_depth: Float32[torch.Tensor, "h w"]
    depth: Image.Image


def _parse_depth_pipeline_output(output: object) -> DepthDict:
    """Validate the transformers depth-estimation pipeline output."""

    if not isinstance(output, dict):
        raise TypeError(f"DepthAnything pipeline returned {type(output).__name__}, expected dict.")
    predicted_depth_raw: object = output.get("predicted_depth")
    depth_raw: object = output.get("depth")
    if not isinstance(predicted_depth_raw, torch.Tensor):
        raise TypeError("DepthAnything pipeline output missing tensor 'predicted_depth'.")
    if not isinstance(depth_raw, Image.Image):
        raise TypeError("DepthAnything pipeline output missing PIL 'depth'.")
    predicted_depth: Float32[torch.Tensor, "h w"] = predicted_depth_raw
    depth: Image.Image = depth_raw
    return {"predicted_depth": predicted_depth, "depth": depth}


class DepthAnythingV1Predictor(BaseRelativePredictor[torch.nn.Module]):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
    ) -> None:
        super().__init__()
        print("Loading DepthAnythingV1 model...")
        start = timer()
        self.pipe = pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=device,
        )
        # The base contract exposes the underlying module (the pipeline itself is not device-movable).
        self.model: torch.nn.Module = self.pipe.model
        print(f"DepthAnythingV1 model loaded. Time: {timer() - start:.2f}s")

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> RelativeDepthPrediction:
        h, w, _ = rgb.shape
        # Transformers pipeline doesn't work with numpy/cv2, requires pil
        pil_img: Image.Image = Image.fromarray(rgb)
        depth_dict: DepthDict = _parse_depth_pipeline_output(self.pipe(pil_img))

        # depth is actually disparity here, interpolate to the original size
        disparity_bchw: Float32[torch.Tensor, "1 1 h w"] = (
            torch.nn.functional.interpolate(
                rearrange(depth_dict["predicted_depth"], "h w -> 1 1 h w"),
                (h, w),
                mode="bilinear",
            )
        )

        if K_33 is None:
            K_33 = estimate_intrinsics(H=h, W=w)

        disparity: Float32[np.ndarray, "h w"] = rearrange(
            disparity_bchw, "1 1 h w -> h w"
        ).numpy(force=True)

        relative_prediction = RelativeDepthPrediction(
            disparity=disparity,
            depth=disparity_to_depth(disparity, focal_length=int(K_33[0, 0])),
            confidence=np.ones_like(disparity),
            K_33=K_33,
        )

        return relative_prediction

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        # Move the one existing model in place — never rebuild the pipeline,
        # so no stale copy of the weights can outlive a device switch.
        self.model.to(device)
        self.pipe.device = torch.device(device)
