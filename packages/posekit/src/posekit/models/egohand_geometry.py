"""EgoHand's lens model for a calibrated camera, and explicit anatomical hand correspondences."""

from functools import cache

import numpy as np
from egohand_inference.camera.camera_models import CameraModel, load_camera_calibration
from jaxtyping import Float32, Float64
from numpy import ndarray

from posekit.cameras import CalibratedCamera, CameraKey, camera_key
from posekit.predictions import SOMA_HAND21, HandSide


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
