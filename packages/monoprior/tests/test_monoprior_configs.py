from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from jaxtyping import Float, UInt8

from monopriors.models.metric_depth import BaseMetricPredictor, BaseMetricPredictorConfig, MetricDepthPrediction
from monopriors.models.monoprior import DsineAndUnidepthConfig
from monopriors.models.surface_normal import (
    BaseNormalPredictor,
    BaseNormalPredictorConfig,
    SurfaceNormalPrediction,
)


class _MetricPredictor(BaseMetricPredictor):
    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None,
    ) -> MetricDepthPrediction:
        raise AssertionError("Prediction is outside this config-construction test.")


class _NormalPredictor(BaseNormalPredictor):
    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None,
    ) -> SurfaceNormalPrediction:
        raise AssertionError("Prediction is outside this config-construction test.")


@dataclass
class _MetricConfig(BaseMetricPredictorConfig):
    predictor: _MetricPredictor = field(default_factory=_MetricPredictor)
    """Predictor returned by setup."""
    requested_device: Literal["cpu", "cuda"] | None = None
    """Device received by setup."""

    def setup(self, device: Literal["cpu", "cuda"]) -> BaseMetricPredictor:
        self.requested_device = device
        return self.predictor


@dataclass
class _NormalConfig(BaseNormalPredictorConfig):
    predictor: _NormalPredictor = field(default_factory=_NormalPredictor)
    """Predictor returned by setup."""
    requested_device: Literal["cpu", "cuda"] | None = None
    """Device received by setup."""

    def setup(self, device: Literal["cpu", "cuda"]) -> BaseNormalPredictor:
        self.requested_device = device
        return self.predictor


def test_dsine_and_unidepth_config_composes_children_on_requested_device() -> None:
    metric_config: _MetricConfig = _MetricConfig()
    normal_config: _NormalConfig = _NormalConfig()

    model = DsineAndUnidepthConfig(metric=metric_config, normal=normal_config).setup(device="cpu")

    assert model.device == "cpu"
    assert model.depth_model is metric_config.predictor
    assert model.surface_model is normal_config.predictor
    assert metric_config.requested_device == "cpu"
    assert normal_config.requested_device == "cpu"
