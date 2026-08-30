from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from jaxtyping import Float, UInt8

from monopriors.models.metric_depth import BaseMetricPredictor, BaseMetricPredictorConfig, MetricDepthPrediction, UniDepthMetricConfig
from monopriors.models.moge_v2.adapters import geometry_to_metric_prediction, normal_to_surface_prediction, rgb_to_cuda_batch
from monopriors.models.moge_v2.export import Encoder
from monopriors.models.moge_v2.predictor import MoGeV2GeometryOutput, MoGeV2TorchPredictor
from monopriors.models.surface_normal import (
    BaseNormalPredictor,
    BaseNormalPredictorConfig,
    DSineNormalConfig,
    SurfaceNormalPrediction,
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

    def __init__(self, device: Literal["cpu", "cuda"]) -> None:
        self.device: Literal["cpu", "cuda"] = device

    @abstractmethod
    def __call__(self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None) -> MonoPriorPrediction:
        """Predict metric depth and surface normals for a uint8 RGB image shaped ``h w 3``."""
        raise NotImplementedError


@dataclass
class MonoPriorModelConfig(ABC):
    """Base configuration for a paired metric-depth and surface-normal model."""

    @abstractmethod
    def setup(self, device: Literal["cpu", "cuda"]) -> MonoPriorModel:
        """Build the configured composite model on one device."""
        raise NotImplementedError


@dataclass
class DsineAndUnidepthConfig(MonoPriorModelConfig):
    """Configuration for separate UniDepth metric-depth and DSINE normal models."""

    metric: BaseMetricPredictorConfig = field(default_factory=UniDepthMetricConfig)
    """Metric-depth predictor configuration."""
    normal: BaseNormalPredictorConfig = field(default_factory=DSineNormalConfig)
    """Surface-normal predictor configuration."""

    def setup(self, device: Literal["cpu", "cuda"]) -> DsineAndUnidepth:
        """Build the configured UniDepth and DSINE composite model."""
        return DsineAndUnidepth(config=self, device=device)


class DsineAndUnidepth(MonoPriorModel):
    """Pair UniDepth metric depth with DSINE surface normals."""

    def __init__(self, config: DsineAndUnidepthConfig, device: Literal["cpu", "cuda"]) -> None:
        super().__init__(device=device)
        self.depth_model: BaseMetricPredictor = config.metric.setup(device=device)
        self.surface_model: BaseNormalPredictor = config.normal.setup(device=device)

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


@dataclass
class MoGeV2MonoPriorConfig(MonoPriorModelConfig):
    """Configuration for joint MoGe v2 metric-depth and normal prediction."""

    encoder: Encoder = "vitl"
    """DINOv2 encoder size."""

    def setup(self, device: Literal["cpu", "cuda"]) -> MoGeV2MonoPrior:
        """Build the configured MoGe v2 composite model."""
        return MoGeV2MonoPrior(device=device, encoder=self.encoder)


class MoGeV2MonoPrior(MonoPriorModel):
    """Composite model using a single MoGe v2 forward pass for both metric depth and normals.

    Unlike ``DsineAndUnidepth`` which loads two separate models, this runs one
    model that produces both outputs simultaneously.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        encoder: Encoder = "vitl",
    ) -> None:
        """Load a pinned normal-capable MoGe v2 checkpoint.

        Args:
            device: Must be ``"cuda"``; MoGe v2 inference is GPU-only.
            encoder: DINOv2 encoder size.

        Raises:
            ValueError: If CPU is requested.
        """
        super().__init__(device=device)
        if device == "cpu":
            raise ValueError("MoGe v2 predictors require CUDA; device='cpu' is unsupported.")
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
