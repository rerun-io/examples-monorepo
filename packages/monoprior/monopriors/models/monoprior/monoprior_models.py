from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, UInt8

from monopriors.models.metric_depth import BaseMetricPredictor, MetricDepthPrediction, UniDepthMetricConfig
from monopriors.models.moge_v2.adapters import geometry_to_metric_prediction, normal_to_surface_prediction, rgb_to_cuda_batch
from monopriors.models.moge_v2.export import Encoder
from monopriors.models.moge_v2.predictor import MoGeV2GeometryOutput, MoGeV2TorchPredictor
from monopriors.models.surface_normal import (
    BaseNormalPredictor,
    SurfaceNormalPrediction,
    get_normal_predictor,
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
        self.depth_model: BaseMetricPredictor = UniDepthMetricConfig().setup(device=self.device)
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
