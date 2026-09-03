"""Base contract for calibrated multi-camera rig depth predictors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor


@dataclass(slots=True)
class RigDepthPrediction:
    """Per-view metric depth prediction at the network input resolution."""

    depth_m: Float32[Tensor, "s h w"]
    """Camera-frame z-depth in metres for each view."""
    confidence: Float32[Tensor, "s h w"]
    """Positive uncalibrated confidence score; larger means more reliable."""
    mask: Float32[Tensor, "s h w"]
    """Predicted non-ambiguity probability in ``[0, 1]``."""
    scale: float
    """Positive scene-level metric scaling factor."""


def unproject(
    prediction: RigDepthPrediction,
    rays: Float32[ndarray, "s h w 3"],
) -> Float32[ndarray, "s h w 3"]:
    """Unproject camera z-depth along unit camera-frame rays.

    Args:
        prediction: Per-view X-Lens output at network resolution.
        rays: OpenCV camera-frame unit rays, ``Float32[ndarray, "s h w 3"]``.

    Returns:
        Camera-frame points, ``Float32[ndarray, "s h w 3"]``. Invalid zero rays
        remain zero.
    """
    depth_m: Float32[ndarray, "s h w"] = prediction.depth_m.detach().cpu().numpy()
    safe_z: Float32[ndarray, "s h w"] = np.maximum(np.abs(rays[..., 2]), np.float32(1e-6))
    points: Float32[ndarray, "s h w 3"] = np.asarray((depth_m / safe_z)[..., None] * rays, dtype=np.float32)
    valid: np.ndarray = (np.linalg.norm(rays, axis=-1) >= 1e-3) & np.isfinite(depth_m)
    points[~valid] = 0.0
    return points


class BaseRigDepthPredictor(ABC):
    """Interface for calibrated rig-to-depth predictors."""

    @abstractmethod
    def __call__(
        self,
        images: UInt8[ndarray, "s h w 3"],
        rays: Float32[ndarray, "s h w 3"],
        cam_types: Int64[ndarray, "s"],
        cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    ) -> RigDepthPrediction:
        """Predict metric z-depth, confidence, and mask for a calibrated rig."""
        raise NotImplementedError


@dataclass
class BaseRigDepthPredictorConfig(ABC):
    """Base configuration for a calibrated rig-depth predictor."""

    @abstractmethod
    def setup(self, device: Literal["cpu", "cuda"]) -> BaseRigDepthPredictor:
        """Build the configured predictor on one device."""
        raise NotImplementedError
