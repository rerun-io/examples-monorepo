from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8

from monopriors.models.metric_depth import MetricDepthPrediction, get_metric_predictor
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
        self.depth_model = get_metric_predictor("Metric3DPredictor")(device=self.device)
        self.surface_model = get_normal_predictor("DSineNormalPredictor")(device=self.device)

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MonoPriorPrediction:
        metric_pred: MetricDepthPrediction = self.depth_model.__call__(rgb, K_33)
        normal_pred: SurfaceNormalPrediction = self.surface_model(rgb, K_33)

        return MonoPriorPrediction(metric_pred=metric_pred, normal_pred=normal_pred)
