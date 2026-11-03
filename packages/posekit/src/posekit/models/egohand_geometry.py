"""Calibrated OAK camera and explicit anatomical hand correspondences."""

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from egohand_inference.camera.camera_models import CameraModel, load_camera_calibration
from jaxtyping import Float32, Float64
from numpy import ndarray

HandSide: TypeAlias = Literal["left", "right"]
SOMA_HAND21: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24)
"""Wrist, then thumb/index/middle/ring/little: omit non-thumb metacarpals."""


def detector_hand_side(label: int) -> HandSide:
    """Decode the raw NVIDIA class name; identity assignment is a separate stage."""
    if label == 1:
        return "left"
    if label == 2:
        return "right"
    raise ValueError(f"Unknown hand label {label}; expected 1 or 2")


def coco_hand_indices(side: HandSide) -> tuple[int, ...]:
    """Return the matching COCO-133 hand indices in MediaPipe order."""
    start: int = 91 if side == "left" else 112
    return tuple(range(start, start + 21))


def soma_to_hand21(joints: Float32[ndarray, "25 3"]) -> Float32[ndarray, "21 3"]:
    """Select MediaPipe/COCO and catalog Quest-21 joints from SOMA-25."""
    return joints[np.asarray(SOMA_HAND21)]


@dataclass(frozen=True, slots=True)
class OAKCamera:
    """An OAK RGB camera retaining all 14 OpenCV distortion terms."""

    intrinsics: Float64[ndarray, "3 3"]
    """Pixel intrinsics for the original decoded image."""
    distortion: Float64[ndarray, "14"]
    """k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tx,ty."""
    width: int
    """Original image width."""
    height: int
    """Original image height."""
    model: CameraModel = field(init=False, repr=False)
    """Validated upstream camera; also used by the inference crop path."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model",
            load_camera_calibration(
                {
                    "model": "RATIONAL_POLYNOMIAL",
                    "image_width": self.width,
                    "image_height": self.height,
                    "camera_matrix": {"matrix_3x3": self.intrinsics},
                    "distortion_coefficients": self.distortion,
                }
            ),
        )

    def project(self, points: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
        """Project camera-space points through all distortion terms."""
        return self.model.project(points)

    def unproject(self, pixels: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 3"]:
        """Recover unit camera-space rays, rejecting non-invertible pixels."""
        return self.model.unproject(pixels)
