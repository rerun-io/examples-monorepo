from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8

from monopriors.third_party.moge.model._inference import InferenceOutput
from monopriors.third_party.moge.model.v2 import MoGeModel

from .base_metric_depth import BaseMetricPredictor, MetricDepthPrediction

MOGE_V2_CHECKPOINTS: dict[tuple[str, bool], tuple[str, str]] = {
    ("vitl", False): ("Ruicheng/moge-2-vitl", "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"),
    ("vitl", True): ("Ruicheng/moge-2-vitl-normal", "b135031bae30b5ac2ae141a0e68717795ce38340"),
    ("vitb", True): ("Ruicheng/moge-2-vitb-normal", "54ad3a693e61907ea4633d13dec6ee682fa09419"),
    ("vits", True): ("Ruicheng/moge-2-vits-normal", "679230677b4d282c6f304189a93e98e14f085902"),
}
"""Mapping of (encoder, with_normals) to Hugging Face repository and revision."""


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
        """Load a pinned MoGe v2 metric checkpoint.

        Args:
            device: Torch device used for inference.
            encoder: DINOv2 encoder size.
        """
        super().__init__()
        # Prefer depth-only checkpoint; fall back to depth+normals
        checkpoint: tuple[str, str] = MOGE_V2_CHECKPOINTS.get(
            (encoder, False),
            MOGE_V2_CHECKPOINTS[(encoder, True)],
        )
        repo_id: str = checkpoint[0]
        revision: str = checkpoint[1]
        print(f"Loading MoGe v2 metric model ({repo_id})...")
        start: float = timer()
        self.model: MoGeModel = MoGeModel.from_pretrained(repo_id, revision=revision).to(device)
        print(f"MoGe v2 metric model loaded. Time: {timer() - start:.2f}s")
        self.device: Literal["cpu", "cuda"] = device

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MetricDepthPrediction:
        """Predict metric depth from an RGB image.

        Args:
            rgb: uint8 RGB image shaped ``h w 3``.
            K_33: Optional float camera intrinsics shaped ``3 3``; MoGe v2 estimates its own intrinsics.

        Returns:
            Metric depth, confidence, and estimated intrinsics.
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
            output_heads=("points", "mask", "scale"),
        )
        # v2 output keys: "points" (H,W,3), "depth" (H,W), "mask" (H,W),
        # "intrinsics" (3,3), and optionally "normal" (H,W,3).
        # Depth is metric-scale. Intrinsics are normalized.

        normalized_k: Float[np.ndarray, "3 3"] = output["intrinsics"].numpy(force=True)
        fx: float = float(normalized_k[0, 0] * w)
        fy: float = float(normalized_k[1, 1] * h)
        cx: float = float(normalized_k[0, 2] * w)
        cy: float = float(normalized_k[1, 2] * h)

        intrinsics: Float[np.ndarray, "3 3"] = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        depth_meters: Float[np.ndarray, "h w"] = output["depth"].numpy(force=True)
        confidence: Float[np.ndarray, "h w"] = output["mask"].numpy(force=True).astype(np.float32)

        return MetricDepthPrediction(
            depth_meters=depth_meters,
            confidence=confidence,
            K_33=intrinsics,
        )
