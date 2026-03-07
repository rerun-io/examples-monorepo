from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, UInt8
from moge.model.v1 import MoGeModel
from torch import Tensor

from monopriors.depth_utils import depth_to_disparity

from .base_metric_depth import BaseMetricPredictor, MetricDepthPrediction


class MogeV1Predictor(BaseMetricPredictor):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        encoder: Literal["vits", "vitb", "vitl"] = "vits",
    ) -> None:
        super().__init__()
        print("Loading MoGe model...")
        start = timer()
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        print(f"MoGe model loaded. Time: {timer() - start:.2f}s")
        self.device = device

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None = None
    ) -> MetricDepthPrediction:
        h, w, _ = rgb.shape
        input_image = torch.tensor(rgb / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)

        # Infer
        output: dict[str, Tensor] = self.model.infer(input_image)
        # `output` has keys "points", "depth", "mask" and "intrinsics",
        # The maps are in the same size as the input image.
        # {
        #     "points": (H, W, 3),    # scale-invariant point map in OpenCV camera coordinate system (x right, y down, z forward)
        #     "depth": (H, W),        # scale-invariant depth map
        #     "mask": (H, W),         # a binary mask for valid pixels.
        #     "intrinsics": (3, 3),   # normalized camera intrinsics
        # }
        # For more usage details, see the `MoGeModel.infer` docstring.
        normalized_k: Float[np.ndarray, "3 3"] = output["intrinsics"].numpy(force=True)
        fx = float(normalized_k[0, 0] * w)
        fy = float(normalized_k[1, 1] * h)
        cx = float(normalized_k[0, 2] * w)
        cy = float(normalized_k[1, 2] * h)

        K_33: Float[np.ndarray, "3 3"] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        points: Float[np.ndarray, "h w 3"] = output["points"].numpy(force=True)
        relative_depth: Float[np.ndarray, "h w"] = output["depth"].numpy(force=True)
        mask: Float[np.ndarray, "h w"] = output["mask"].numpy(force=True).astype(np.float32)

        relative_prediction = RelativeDepthPrediction(
            disparity=depth_to_disparity(relative_depth, focal_length=int(K_33[0, 0])),
            depth=relative_depth,
            confidence=mask,
            K_33=K_33,
        )

        return relative_prediction
