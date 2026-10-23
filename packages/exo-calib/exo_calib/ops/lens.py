"""Undistortion: raw pixels and frames to the pinhole image of the same intrinsics, the space the refine geometry works in."""

import cv2
import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray

from exo_calib.types import Lens


def undistort(lens: Lens, xy: Float64[ndarray, "n 2"], K: Float64[ndarray, "3 3"]) -> Float64[ndarray, "n 2"]:
    """The pinhole pixels (same ``K``) of raw pixels; the focal enters, so undistort again whenever it changes."""
    if lens.model == "pinhole" or len(xy) == 0:
        return xy.copy()
    raw: Float64[ndarray, "n 1 2"] = np.ascontiguousarray(xy, dtype=np.float64).reshape(-1, 1, 2)
    coefficients: Float64[ndarray, " k"] = np.asarray(lens.coefficients, dtype=np.float64)
    if lens.model == "kannala_brandt":
        return cv2.fisheye.undistortPoints(raw, K, coefficients[:4], P=K).reshape(-1, 2)
    return cv2.undistortPoints(raw, K, coefficients, P=K).reshape(-1, 2)


def rectify(lens: Lens, frames: UInt8[ndarray, "k h w 3"], K: Float64[ndarray, "3 3"]) -> UInt8[ndarray, "k h w 3"]:
    """Frames of one camera resampled to the pinhole image of the same ``K``: a raw pixel shows where ``undistort`` puts it,
    so a pinhole-only model (MoGe) sees a true pinhole image."""
    if lens.model == "pinhole":
        return frames
    size: tuple[int, int] = (frames.shape[2], frames.shape[1])
    coefficients: Float64[ndarray, " k"] = np.asarray(lens.coefficients, dtype=np.float64)
    if lens.model == "kannala_brandt":
        maps = cv2.fisheye.initUndistortRectifyMap(K, coefficients[:4], np.eye(3), K, size, cv2.CV_32FC1)
    else:
        maps = cv2.initUndistortRectifyMap(K, coefficients, np.eye(3), K, size, cv2.CV_32FC1)
    return np.stack([cv2.remap(frame, *maps, cv2.INTER_LINEAR) for frame in frames])
