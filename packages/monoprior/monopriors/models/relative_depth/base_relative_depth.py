from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np
from jaxtyping import Float32, UInt8

from monopriors.models._protocols import DeviceMovable

ModelT = TypeVar("ModelT", bound=DeviceMovable)


@dataclass
class RelativeDepthPrediction:
    disparity: Float32[np.ndarray, "h w"]
    # relative disparity
    depth: Float32[np.ndarray, "h w"]
    # relative depth
    K_33: Float32[np.ndarray, "3 3"]
    # intrinsics
    confidence: Float32[np.ndarray, "h w"] | None
    # per-pixel confidence in [0, 1]; None when the model has no confidence head
    # (DepthAnything v1/v2, VideoDepthAnything, ZipDepth). MoGe gives a validity mask,
    # UniDepth a normalized confidence.


class BaseRelativePredictor(ABC, Generic[ModelT]):
    model: ModelT

    @abstractmethod
    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float32[np.ndarray, "3 3"] | None
    ) -> RelativeDepthPrediction:
        raise NotImplementedError

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        self.model.to(device)


class BaseVideoRelativePredictor(ABC, Generic[ModelT]):
    model: ModelT

    @abstractmethod
    def __call__(
        self, rgb_frames: UInt8[np.ndarray, "T H W 3"], K_33: Float32[np.ndarray, "3 3"] | None
    ) -> list[RelativeDepthPrediction]:
        raise NotImplementedError

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        self.model.to(device)


@dataclass
class BaseRelativePredictorConfig(ABC):
    """Base configuration for a single-image relative-depth predictor."""

    @abstractmethod
    def setup(self, device: Literal["cpu", "cuda"]) -> BaseRelativePredictor:
        """Build the configured predictor on one device."""
        raise NotImplementedError


@dataclass
class BaseVideoRelativePredictorConfig(ABC):
    """Base configuration for a video relative-depth predictor."""

    @abstractmethod
    def setup(self, device: Literal["cpu", "cuda"]) -> BaseVideoRelativePredictor:
        """Build the configured video predictor on one device."""
        raise NotImplementedError
