"""EgoHand's lens model for a calibrated camera, and the detector's raw hand classes."""

from functools import cache

import numpy as np
from egohand_inference.camera.camera_models import CameraModel, load_camera_calibration
from jaxtyping import Float64
from numpy import ndarray

from posekit.cameras import CalibratedCamera, CameraKey, camera_key
from posekit.predictions import HandSide


def detector_hand_side(label: int) -> HandSide:
    """Decode the raw NVIDIA class name; identity assignment is a separate stage."""
    if label == 1:
        return "left"
    if label == 2:
        return "right"
    raise ValueError(f"Unknown hand label {label}; expected 1 or 2")


def lens_model(camera: CalibratedCamera) -> CameraModel:
    """The validated upstream camera for ``camera``, built once per distinct calibration.

    The model caches its GPU lens coefficients, so reusing one instance per
    calibration keeps them resident across calls.
    """
    return _lens_model(camera_key(camera))


@cache
def _lens_model(key: CameraKey) -> CameraModel:
    lens, intrinsics, distortion, width, height = key
    if lens == "DOUBLE_SPHERE":
        from posekit.models.egohand_double_sphere import DoubleSphereCamera

        return DoubleSphereCamera(
            CalibratedCamera(
                np.frombuffer(intrinsics, dtype=np.float64).reshape(3, 3), np.frombuffer(distortion, dtype=np.float64), width, height, lens
            )
        )
    return load_camera_calibration(
        {
            "model": lens,
            "image_width": width,
            "image_height": height,
            "camera_matrix": {"matrix_3x3": np.frombuffer(intrinsics, dtype=np.float64).reshape(3, 3)},
            "distortion_coefficients": np.frombuffer(distortion, dtype=np.float64),
        }
    )


def project(camera: CalibratedCamera, points: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
    """Project camera-space points through every distortion term of ``camera``."""
    return lens_model(camera).project(points)
