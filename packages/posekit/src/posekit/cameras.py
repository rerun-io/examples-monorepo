"""Calibrated source cameras for models that undo lens distortion (EgoHand's perspective crops).

A pure record: the lens projection itself lives with the model that needs it, so
importing this module pulls in no GPU or warp code.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from jaxtyping import Float64
from numpy import ndarray
from serde import SerdeError, from_dict, serde

LensModel: TypeAlias = Literal["PINHOLE", "RATIONAL_POLYNOMIAL", "FISHEYE624", "KB4"]
"""Lens models EgoHand's camera code projects: no distortion, OpenCV 14-term, Meta fisheye 6-2-4, Kannala-Brandt 4."""
LENS_COEFFICIENTS: dict[LensModel, int] = {"PINHOLE": 0, "RATIONAL_POLYNOMIAL": 14, "FISHEYE624": 12, "KB4": 4}
"""Distortion coefficient count of each lens model."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CalibratedCamera:
    """Intrinsics and lens distortion of the camera that took the frames, in its source pixels."""

    intrinsics: Float64[ndarray, "3 3"]
    """Pixel intrinsics ``K`` of the original decoded image (zero skew)."""
    distortion: Float64[ndarray, "d"]
    """Distortion coefficients in the order of ``lens``: OpenCV ``k1,k2,p1,p2,k3,k4,k5,k6,s1..s4,tx,ty``;
    FISHEYE624 ``k0..k5, p0, p1, s0..s3``; KB4 ``k1..k4``; PINHOLE none."""
    width: int
    """Original image width in pixels."""
    height: int
    """Original image height in pixels."""
    lens: LensModel = "RATIONAL_POLYNOMIAL"
    """Lens model the coefficients belong to."""

    def __post_init__(self) -> None:
        expected: int = LENS_COEFFICIENTS[self.lens]
        if self.distortion.shape != (expected,):
            raise ValueError(f"{self.lens} needs {expected} distortion coefficients, got {self.distortion.shape[0]}.")
        if min(self.width, self.height) <= 0:
            raise ValueError(f"Camera size must be positive, got {self.width}x{self.height}.")


def load_calibrated_camera(path: Path) -> CalibratedCamera:
    """Read a ``CalibratedCamera`` JSON written by us.

    Args:
        path: JSON file holding one ``CalibratedCamera``.

    Returns:
        The validated camera.

    Raises:
        ValueError: If the file is not valid JSON or does not match the schema.
    """
    try:
        return from_dict(CalibratedCamera, json.loads(path.read_text()))
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: not a CalibratedCamera: {error}") from error
