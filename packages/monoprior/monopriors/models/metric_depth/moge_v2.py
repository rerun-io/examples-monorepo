"""Single-image metric-depth adapter for the unified MoGe v2 core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float, UInt8

from monopriors.models.metric_depth.base_metric_depth import BaseMetricPredictor, BaseMetricPredictorConfig, MetricDepthPrediction
from monopriors.models.moge_v2.adapters import geometry_to_metric_prediction, rgb_to_cuda_batch
from monopriors.models.moge_v2.export import Encoder
from monopriors.models.moge_v2.predictor import MoGeV2GeometryOutput, MoGeV2TorchPredictor
from monopriors.third_party.moge.model.v2 import MoGeModel


@dataclass
class MoGeV2MetricConfig(BaseMetricPredictorConfig):
    """Configuration for MoGe v2 metric-depth prediction."""

    encoder: Encoder = "vitl"
    """DINOv2 encoder size."""

    def setup(self, device: Literal["cpu", "cuda"]) -> MoGeV2MetricPredictor:
        """Build the configured MoGe v2 metric predictor on one device."""
        return MoGeV2MetricPredictor(device=device, encoder=self.encoder)


class MoGeV2MetricPredictor(BaseMetricPredictor[MoGeModel]):
    """MoGe v2 metric-depth adapter using the normal-capable checkpoint."""

    def __init__(self, device: Literal["cpu", "cuda"], encoder: Encoder = "vitl") -> None:
        """Load the unified geometry core.

        Args:
            device: Must be ``"cuda"``; MoGe v2 inference is GPU-only.
            encoder: DINOv2 encoder size.

        Raises:
            ValueError: If CPU is requested.
        """
        super().__init__()
        if device == "cpu":
            raise ValueError("MoGe v2 predictors require CUDA; device='cpu' is unsupported.")
        self.predictor: MoGeV2TorchPredictor = MoGeV2TorchPredictor(encoder=encoder, heads="geometry")
        self.model: MoGeModel = self.predictor.model
        self.device: Literal["cpu", "cuda"] = device

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MetricDepthPrediction:
        """Predict metric depth, binary confidence, and pixel intrinsics.

        Args:
            rgb: uint8 NumPy RGB image shaped ``h w 3``.
            K_33: Ignored caller intrinsics; MoGe v2 estimates its own.

        Returns:
            Float32 metric depth, binary confidence, and pixel intrinsics.
        """
        output: MoGeV2GeometryOutput = self.predictor.predict_geometry(rgb_to_cuda_batch(rgb))
        return geometry_to_metric_prediction(output)
