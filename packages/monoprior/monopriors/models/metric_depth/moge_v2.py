from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, UInt8
from moge.model.v2 import MoGeModel
from torch import Tensor

from .base_metric_depth import BaseMetricPredictor, MetricDepthPrediction

MOGE_V2_CHECKPOINTS: dict[tuple[str, bool], str] = {
    ("vitl", False): "Ruicheng/moge-2-vitl",
    ("vitl", True): "Ruicheng/moge-2-vitl-normal",
    ("vitb", True): "Ruicheng/moge-2-vitb-normal",
    ("vits", True): "Ruicheng/moge-2-vits-normal",
}
"""Mapping of (encoder, with_normals) to HuggingFace checkpoint name."""


class MoGeV2MetricPredictor(BaseMetricPredictor):
    """MoGe v2 predictor producing metric-scale depth.

    Uses the depth-only checkpoint when available, otherwise falls back
    to the depth+normals checkpoint (ignoring the normal output).
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        encoder: Literal["vits", "vitb", "vitl"] = "vitl",
    ) -> None:
        super().__init__()
        # Prefer depth-only checkpoint; fall back to depth+normals
        checkpoint: str = MOGE_V2_CHECKPOINTS.get(
            (encoder, False),
            MOGE_V2_CHECKPOINTS[(encoder, True)],
        )
        print(f"Loading MoGe v2 metric model ({checkpoint})...")
        start: float = timer()
        self.model = MoGeModel.from_pretrained(checkpoint).to(device)
        print(f"MoGe v2 metric model loaded. Time: {timer() - start:.2f}s")
        self.device: Literal["cpu", "cuda"] = device

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MetricDepthPrediction:
        h: int
        w: int
        h, w, _ = rgb.shape
        input_image: Float[torch.Tensor, "3 h w"] = torch.tensor(
            rgb / 255, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)

        output: dict[str, Tensor] = self.model.infer(input_image)
        # v2 output keys: "points" (H,W,3), "depth" (H,W), "mask" (H,W),
        # "intrinsics" (3,3), and optionally "normal" (H,W,3).
        # Depth is metric-scale. Intrinsics are normalized.

        normalized_k: Float[np.ndarray, "3 3"] = output["intrinsics"].numpy(force=True)
        fx: float = float(normalized_k[0, 0] * w)
        fy: float = float(normalized_k[1, 1] * h)
        cx: float = float(normalized_k[0, 2] * w)
        cy: float = float(normalized_k[1, 2] * h)

        intrinsics: Float[np.ndarray, "3 3"] = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )
        depth_meters: Float[np.ndarray, "h w"] = output["depth"].numpy(force=True)
        confidence: Float[np.ndarray, "h w"] = output["mask"].numpy(force=True).astype(np.float32)

        return MetricDepthPrediction(
            depth_meters=depth_meters,
            confidence=confidence,
            K_33=intrinsics,
        )
