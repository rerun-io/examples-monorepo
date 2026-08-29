from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8

from monopriors.models.metric_depth import BaseMetricPredictor, MetricDepthPrediction, get_metric_predictor
from monopriors.models.moge_v2.adapters import geometry_to_metric_prediction, normal_to_surface_prediction, rgb_to_cuda_batch
from monopriors.models.moge_v2.export import Encoder
from monopriors.models.moge_v2.predictor import MoGeV2GeometryOutput, MoGeV2TorchPredictor
from monopriors.models.surface_normal import (
    BaseNormalPredictor,
    SurfaceNormalPrediction,
    get_normal_predictor,
)


@dataclass
class OldMonoPriorPrediction:
    """Legacy batched tensor representation of monocular priors."""

    depth_b1hw: Float[torch.Tensor, "b 1 h w"]
    """Predicted metric depth in BCHW layout."""
    normal_b3hw: Float[torch.Tensor, "b 3 h w"]
    """Predicted surface normals in BCHW layout."""
    K_b33: Float[torch.Tensor, "b 3 3"] | None = None
    """Optional camera intrinsics for each batch item."""
    depth_conf_b1hw: Float[torch.Tensor, "b 1 h w"] | None = None
    """Optional per-pixel depth confidence."""
    normal_conf_b1hw: Float[torch.Tensor, "b 1 h w"] | None = None
    """Optional per-pixel normal confidence."""

    def to_numpy(
        self,
    ) -> tuple[
        Float[np.ndarray, "b h w 1"],
        Float[np.ndarray, "b h w 3"],
        Float[np.ndarray, "b 3 3"] | None,
        Float[np.ndarray, "b h w 1"] | None,
        Float[np.ndarray, "b h w 1"] | None,
    ]:
        """Convert legacy Torch predictions to channel-last NumPy arrays.

        Returns:
            Float depth, normal, intrinsics, depth-confidence, and normal-confidence arrays with their documented shapes.
        """
        depth_np_bhw1: Float[np.ndarray, "b h w 1"] = rearrange(self.depth_b1hw, "b c h w -> b h w c").numpy(force=True)
        normal_np_bhw3: Float[np.ndarray, "b h w 3"] = rearrange(self.normal_b3hw, "b c h w -> b h w c").numpy(force=True)
        K_np_b33: Float[np.ndarray, "b 3 3"] | None = self.K_b33.numpy(force=True) if self.K_b33 is not None else None
        depth_conf_np_bhw1: Float[np.ndarray, "b h w 1"] | None = (
            rearrange(self.depth_conf_b1hw, "b c h w -> b h w c").numpy(force=True) if self.depth_conf_b1hw is not None else None
        )
        normal_conf_np_bhw1: Float[np.ndarray, "b h w 1"] | None = (
            rearrange(self.normal_conf_b1hw, "b c h w -> b h w c").numpy(force=True) if self.normal_conf_b1hw is not None else None
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
    """Metric-depth and surface-normal predictions for one image."""

    metric_pred: MetricDepthPrediction
    """Metric depth prediction."""
    normal_pred: SurfaceNormalPrediction
    """Surface-normal prediction."""


class MonoPriorModel(ABC):
    """Interface for a paired metric-depth and surface-normal predictor."""

    def __init__(self) -> None:
        self.device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def __call__(self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None) -> MonoPriorPrediction:
        """Predict metric depth and surface normals for a uint8 RGB image shaped ``h w 3``."""
        raise NotImplementedError


class DsineAndUnidepth(MonoPriorModel):
    """Pair UniDepth metric depth with DSINE surface normals."""

    def __init__(self) -> None:
        super().__init__()
        # Keep this composite aligned with its name by using UniDepth for metric depth.
        self.depth_model: BaseMetricPredictor = get_metric_predictor("UniDepthMetricPredictor")(device=self.device)
        self.surface_model: BaseNormalPredictor = get_normal_predictor("DSineNormalPredictor")(device=self.device)

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MonoPriorPrediction:
        """Predict metric depth and normals with separate models.

        Args:
            rgb: uint8 RGB image shaped ``h w 3``.
            K_33: Optional float camera intrinsics shaped ``3 3``.

        Returns:
            Combined metric-depth and surface-normal prediction.
        """
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
        encoder: Encoder = "vitl",
    ) -> None:
        """Load a pinned normal-capable MoGe v2 checkpoint.

        Args:
            encoder: DINOv2 encoder size.
        """
        super().__init__()
        self.predictor: MoGeV2TorchPredictor = MoGeV2TorchPredictor(encoder=encoder, heads="geometry")

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MonoPriorPrediction:
        """Predict metric depth and normals with one MoGe v2 forward pass.

        Args:
            rgb: uint8 RGB image shaped ``h w 3``.
            K_33: Optional float camera intrinsics shaped ``3 3``; MoGe v2 estimates its own intrinsics.

        Returns:
            Combined metric-depth and surface-normal prediction.
        """
        output: MoGeV2GeometryOutput = self.predictor.predict_geometry(rgb_to_cuda_batch(rgb))
        metric_pred: MetricDepthPrediction = geometry_to_metric_prediction(output)
        normal_pred: SurfaceNormalPrediction = normal_to_surface_prediction(output)
        return MonoPriorPrediction(metric_pred=metric_pred, normal_pred=normal_pred)
