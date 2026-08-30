from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np
from jaxtyping import Float32, UInt8

from monopriors.models._protocols import DeviceMovable

ModelT = TypeVar("ModelT", bound=DeviceMovable)


@dataclass(slots=True)
class StereoDepthPrediction:
    """Output of a rectified stereo matcher for the left view."""

    disparity: Float32[np.ndarray, "h w"]
    """Left-view disparity in pixels (the network's native output); ``<= 0`` marks invalid pixels."""
    depth_meters: Float32[np.ndarray, "h w"]
    """``fx * baseline / disparity``, with ``0`` where disparity is invalid."""
    K_33: Float32[np.ndarray, "3 3"]
    """Rectified pinhole intrinsics shared by both views."""
    baseline_m: float
    """Stereo baseline in metres."""


def disparity_to_metric_depth(disparity_hw: Float32[np.ndarray, "h w"], fx: float, baseline_m: float) -> Float32[np.ndarray, "h w"]:
    """Convert disparity to metric depth, leaving invalid (``<= 0``) disparities at ``0`` depth.

    Args:
        disparity_hw: Left-view disparity in pixels, ``Float32[ndarray, "h w"]``.
        fx: Horizontal focal length in pixels.
        baseline_m: Stereo baseline in metres.

    Returns:
        Depth in metres, ``Float32[ndarray, "h w"]``.
    """
    valid_hw: np.ndarray = disparity_hw > 0.0
    depth_hw: Float32[np.ndarray, "h w"] = np.zeros_like(disparity_hw)
    depth_hw[valid_hw] = fx * baseline_m / disparity_hw[valid_hw]
    return depth_hw


class BaseStereoPredictor(ABC, Generic[ModelT]):
    model: ModelT

    @abstractmethod
    def infer_disparity(
        self,
        left_rgb: UInt8[np.ndarray, "h w 3"],
        right_rgb: UInt8[np.ndarray, "h w 3"],
    ) -> Float32[np.ndarray, "h w"]:
        """Predict left-view disparity in pixels for a rectified stereo pair."""
        raise NotImplementedError

    def __call__(
        self,
        left_rgb: UInt8[np.ndarray, "h w 3"],
        right_rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float32[np.ndarray, "3 3"],
        baseline_m: float,
    ) -> StereoDepthPrediction:
        """Predict calibrated left-view disparity and metric depth for a rectified stereo pair."""
        disparity_hw: Float32[np.ndarray, "h w"] = self.infer_disparity(left_rgb, right_rgb)
        depth_hw: Float32[np.ndarray, "h w"] = disparity_to_metric_depth(disparity_hw, float(K_33[0, 0]), baseline_m)
        return StereoDepthPrediction(disparity=disparity_hw, depth_meters=depth_hw, K_33=K_33, baseline_m=baseline_m)

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        self.model.to(device)


@dataclass
class BaseStereoPredictorConfig(ABC):
    """Base configuration for a stereo predictor."""

    @abstractmethod
    def setup(self, device: Literal["cpu", "cuda"]) -> BaseStereoPredictor:
        """Build the configured predictor on one device."""
        raise NotImplementedError
