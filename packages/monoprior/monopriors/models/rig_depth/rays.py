"""Camera-frame unit-ray fields for simplecv pinhole and KB4 cameras."""

from typing import Literal

import cv2
import numpy as np
from jaxtyping import Float32, Float64
from numpy import ndarray
from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters


def camera_type(camera: PinholeParameters | Fisheye62Parameters) -> Literal[0, 1]:
    """Return X-Lens's camera type id (``0`` fisheye, ``1`` pinhole)."""
    if isinstance(camera, PinholeParameters):
        return 1
    return 0


def unit_rays(camera: PinholeParameters | Fisheye62Parameters) -> Float32[ndarray, "h w 3"]:
    """Build OpenCV-axis unit rays at pixel centres.

    Args:
        camera: A simplecv pinhole camera or Kannala-Brandt fisheye camera.

    Returns:
        Camera-frame unit rays, ``Float32[ndarray, "h w 3"]``. Fisheye pixels
        outside the invertible front-hemisphere field are zero.

    Raises:
        ValueError: If a fisheye camera lacks distortion or uses unsupported
            fifth/sixth radial or tangential terms.
    """
    intrinsics = camera.intrinsics
    width: int = intrinsics.width
    height: int = intrinsics.height
    if intrinsics.fl_x is None or intrinsics.fl_y is None or intrinsics.cx is None or intrinsics.cy is None:
        raise ValueError(f"{camera.name}: camera intrinsics are incomplete")
    fl_x: float = float(intrinsics.fl_x)
    fl_y: float = float(intrinsics.fl_y)
    cx: float = float(intrinsics.cx)
    cy: float = float(intrinsics.cy)
    u_hw: Float32[ndarray, "h w"]
    v_hw: Float32[ndarray, "h w"]
    u_hw, v_hw = np.meshgrid(np.arange(width, dtype=np.float32) + 0.5, np.arange(height, dtype=np.float32) + 0.5, indexing="xy")

    if isinstance(camera, PinholeParameters):
        x_hw: Float32[ndarray, "h w"] = (u_hw - cx) / fl_x
        y_hw: Float32[ndarray, "h w"] = (v_hw - cy) / fl_y
        directions: Float32[ndarray, "h w 3"] = np.stack((x_hw, y_hw, np.ones_like(x_hw)), axis=-1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        return directions.astype(np.float32, copy=False)

    distortion = camera.distortion
    if distortion is None:
        raise ValueError(f"{camera.name}: fisheye camera has no Kannala-Brandt distortion")
    if any(abs(value) > 0.0 for value in (distortion.k5, distortion.k6, distortion.p1, distortion.p2)):
        raise ValueError(f"{camera.name}: cv2.fisheye supports only KB4 k1..k4 coefficients")

    K_33: Float64[ndarray, "3 3"] = np.asarray(intrinsics.k_matrix, dtype=np.float64)
    coefficients_4: Float64[ndarray, "4"] = np.array([distortion.k1, distortion.k2, distortion.k3, distortion.k4], dtype=np.float64)
    pixels_n12: Float64[ndarray, "n 1 2"] = np.stack((u_hw, v_hw), axis=-1).reshape(-1, 1, 2).astype(np.float64)
    normalized_n12: Float64[ndarray, "n 1 2"] = cv2.fisheye.undistortPoints(pixels_n12, K_33, coefficients_4)
    normalized_hw2: Float64[ndarray, "h w 2"] = normalized_n12.reshape(height, width, 2)
    directions64: Float64[ndarray, "h w 3"] = np.concatenate((normalized_hw2, np.ones((height, width, 1), dtype=np.float64)), axis=-1)
    directions64 /= np.maximum(np.linalg.norm(directions64, axis=-1, keepdims=True), 1e-12)

    x_distorted: Float64[ndarray, "h w"] = (u_hw.astype(np.float64) - cx) / fl_x
    y_distorted: Float64[ndarray, "h w"] = (v_hw.astype(np.float64) - cy) / fl_y
    radius_distorted: Float64[ndarray, "h w"] = np.hypot(x_distorted, y_distorted)
    theta_limit: float = np.pi / 2.0
    theta2: float = theta_limit * theta_limit
    max_radius: float = theta_limit * (
        1.0 + distortion.k1 * theta2 + distortion.k2 * theta2**2 + distortion.k3 * theta2**3 + distortion.k4 * theta2**4
    )
    valid: np.ndarray = np.isfinite(normalized_hw2).all(axis=-1) & (np.abs(normalized_hw2) < 1e5).all(axis=-1) & (radius_distorted <= max_radius)

    # OpenCV reports no convergence flag. Reject inverse solutions that do not
    # project back to their source pixel. Work in chunks so the Jacobian that
    # OpenCV allocates internally stays small for multi-megapixel cameras.
    valid_flat: np.ndarray = valid.reshape(-1)
    directions_n13: Float64[ndarray, "n 1 3"] = directions64.reshape(-1, 1, 3)
    reprojection_ok: np.ndarray = np.zeros(pixels_n12.shape[0], dtype=bool)
    zeros_3: Float64[ndarray, "3"] = np.zeros(3, dtype=np.float64)
    chunk_size = 32_768
    for start in range(0, pixels_n12.shape[0], chunk_size):
        stop = min(start + chunk_size, pixels_n12.shape[0])
        projected_n12, _jacobian = cv2.fisheye.projectPoints(
            directions_n13[start:stop],
            zeros_3,
            zeros_3,
            K_33,
            coefficients_4,
        )
        error_n: Float64[ndarray, "n"] = np.linalg.norm(projected_n12 - pixels_n12[start:stop], axis=-1).reshape(-1)
        reprojection_ok[start:stop] = np.isfinite(error_n) & (error_n <= 1e-3)
    valid_flat &= reprojection_ok
    valid = valid_flat.reshape(height, width)
    directions64[~valid] = 0.0
    return directions64.astype(np.float32)
