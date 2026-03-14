from abc import ABC, abstractmethod
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8
from moge.model.v2 import MoGeModel
from torch import Tensor

from monopriors.models.metric_depth import MetricDepthPrediction, get_metric_predictor
from monopriors.models.metric_depth.moge_v2 import MOGE_V2_CHECKPOINTS
from monopriors.models.surface_normal import (
    SurfaceNormalPrediction,
    get_normal_predictor,
)


@dataclass
class OldMonoPriorPrediction:
    depth_b1hw: Float[torch.Tensor, "b 1 h w"]
    normal_b3hw: Float[torch.Tensor, "b 3 h w"]
    K_b33: Float[torch.Tensor, "b 3 3"] | None = None
    depth_conf_b1hw: Float[torch.Tensor, "b 1 h w"] | None = None
    normal_conf_b1hw: Float[torch.Tensor, "b 1 h w"] | None = None

    def to_numpy(
        self,
    ) -> tuple[
        Float[np.ndarray, "b h w 1"],
        Float[np.ndarray, "b h w 3"],
        Float[np.ndarray, "b 3 3"] | None,
        Float[np.ndarray, "b h w 1"] | None,
        Float[np.ndarray, "b h w 1"] | None,
    ]:
        depth_np_bhw1 = rearrange(self.depth_b1hw, "b c h w -> b h w c").numpy(force=True)
        normal_np_bhw3 = rearrange(self.normal_b3hw, "b c h w -> b h w c").numpy(force=True)
        K_np_b33 = self.K_b33.numpy(force=True) if self.K_b33 is not None else None
        depth_conf_np_bhw1 = (
            rearrange(self.depth_conf_b1hw, "b c h w -> b h w c").numpy(force=True)
            if self.depth_conf_b1hw is not None
            else None
        )
        normal_conf_np_bhw1 = (
            rearrange(self.normal_conf_b1hw, "b c h w -> b h w c").numpy(force=True)
            if self.normal_conf_b1hw is not None
            else None
        )
        return (
            depth_np_bhw1,
            normal_np_bhw3,
            K_np_b33,
            depth_conf_np_bhw1,
            normal_conf_np_bhw1,
        )


@dataclass
class MonoPriorPrediction:
    metric_pred: MetricDepthPrediction
    normal_pred: SurfaceNormalPrediction


class MonoPriorModel(ABC):
    def __init__(self) -> None:
        self.device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def __call__(self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None) -> MonoPriorPrediction:
        raise NotImplementedError


class DsineAndUnidepth(MonoPriorModel):
    def __init__(self) -> None:
        super().__init__()
        # Keep this composite aligned with its name by using UniDepth for metric depth.
        self.depth_model = get_metric_predictor("UniDepthMetricPredictor")(device=self.device)
        self.surface_model = get_normal_predictor("DSineNormalPredictor")(device=self.device)

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MonoPriorPrediction:
        metric_pred: MetricDepthPrediction = self.depth_model.__call__(rgb, K_33)
        normal_pred: SurfaceNormalPrediction = self.surface_model(rgb, K_33)

        return MonoPriorPrediction(metric_pred=metric_pred, normal_pred=normal_pred)


class MoGeV2MonoPrior(MonoPriorModel):
    """Composite model using a single MoGe v2 forward pass for both metric depth and normals.

    Unlike ``DsineAndUnidepth`` which loads two separate models, this runs one
    model that produces both outputs simultaneously.
    """

    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl"] = "vitl",
    ) -> None:
        super().__init__()
        checkpoint: str = MOGE_V2_CHECKPOINTS[(encoder, True)]
        print(f"Loading MoGe v2 mono-prior model ({checkpoint})...")
        start: float = timer()
        self.model = MoGeModel.from_pretrained(checkpoint).to(self.device)
        print(f"MoGe v2 mono-prior model loaded. Time: {timer() - start:.2f}s")

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MonoPriorPrediction:
        h: int
        w: int
        h, w, _ = rgb.shape
        input_image: Float[torch.Tensor, "3 h w"] = torch.tensor(
            rgb / 255, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)

        output: dict[str, Tensor] = self.model.infer(input_image)

        # Build intrinsics from normalized values
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
        normal_hw3: Float[np.ndarray, "h w 3"] = output["normal"].numpy(force=True)
        confidence_hw1: Float[np.ndarray, "h w 1"] = confidence[:, :, np.newaxis]

        metric_pred = MetricDepthPrediction(
            depth_meters=depth_meters,
            confidence=confidence,
            K_33=intrinsics,
        )
        normal_pred = SurfaceNormalPrediction(
            normal_hw3=normal_hw3,
            confidence_hw1=confidence_hw1,
        )

        return MonoPriorPrediction(metric_pred=metric_pred, normal_pred=normal_pred)
