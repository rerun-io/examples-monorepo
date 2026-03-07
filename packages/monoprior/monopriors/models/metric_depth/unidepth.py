from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8

from monopriors.models.metric_depth.base_metric_depth import (
    BaseMetricPredictor,
    MetricDepthPrediction,
)


class UniDepthMetricPredictor(BaseMetricPredictor):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        version: Literal["v1", "v2"] = "v2",
        backbone: Literal["vits14", "vitl14"] = "vitl14",
    ):
        super().__init__()
        print("Loading UniDepth model...")
        start = timer()
        self.model = (
            torch.hub.load(
                "lpiccinelli-eth/UniDepth",
                "UniDepth",
                version=version,
                backbone=backbone,
                pretrained=True,
                trust_repo=True,
                force_reload=True,
            )
            .to(device)
            .eval()
        )
        self.model.resolution_level = 0
        print(f"UniDepth model loaded. Time: {timer() - start:.2f}s")

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None,
    ) -> MetricDepthPrediction:
        # Load the RGB image and the normalization will be taken care of by the model
        # rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # C, H, W
        rgb = rearrange(rgb, "h w c -> c h w")
        rgb = torch.from_numpy(rgb)

        if K_33 is None:
            predictions = self.model.infer(rgb)
        else:
            K_33 = torch.from_numpy(K_33)
            predictions = self.model.infer(rgb, K_33)

        depth_b1hw: Float[torch.Tensor, "b 1 h w"] = predictions["depth"]
        K_b33: Float[torch.Tensor, "b 3 3"] = predictions["intrinsics"]
        conf_b1hw: Float[torch.Tensor, "b 1 h w"] = predictions["confidence"]

        assert depth_b1hw.shape[0] == 1, "Batch size must be 1"

        # normalize the confidence to 0-1
        conf_b1hw = (conf_b1hw - conf_b1hw.min()) / (conf_b1hw.max() - conf_b1hw.min())

        # convert to numpy and rearrange
        depth_hw = rearrange(depth_b1hw, "1 1 h w -> h w").numpy(force=True)
        conf_hw = rearrange(conf_b1hw, "1 1 h w -> h w").numpy(force=True)
        # rearrange doesn't work here?
        K_33 = K_b33.squeeze(0).numpy(force=True)

        metric_pred = MetricDepthPrediction(
            depth_meters=depth_hw,
            confidence=conf_hw,
            K_33=K_33,
        )

        return metric_pred
