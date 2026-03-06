from typing import Literal, TypedDict
import torch
import numpy as np
from jaxtyping import Float, UInt8
from timeit import default_timer as timer
from monopriors.depth_utils import estimate_intrinsics, disparity_to_depth
from transformers import pipeline
from PIL import Image
from einops import rearrange
from jaxtyping import Float32
from .base_relative_depth import RelativeDepthPrediction, BaseRelativePredictor


class DepthDict(TypedDict):
    predicted_depth: (
        Float32[torch.Tensor, "1 518 756"] | Float32[torch.Tensor, "1 756 518"]
    )
    depth: Image.Image


class DepthAnythingV1Predictor(BaseRelativePredictor):
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
        print(f"DepthAnythingV1 model loaded. Time: {timer() - start:.2f}s")

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> RelativeDepthPrediction:
        h, w, _ = rgb.shape
        # Transformers pipeline doesn't work with numpy/cv2, requires pil
        pil_img: Image.Image = Image.fromarray(rgb)
        depth_dict: DepthDict = self.pipe(pil_img)

        # depth is actually disparity here, interpolate to the original size
        disparity_bchw: Float32[torch.Tensor, "1 1 h w"] = (
            torch.nn.functional.interpolate(
                rearrange(depth_dict["predicted_depth"], "1 h w -> 1 1 h w"),
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
        # uses pipeline instead of model so not implemented
        self.pipe = pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=device,
        )
