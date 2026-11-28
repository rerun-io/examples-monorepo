"""Synthetic bundle-adjustment problems: frames on a circle around a point cloud (as ba_synth.py), any camera model.

Built in memory from a seed, so no data file is written or read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from caspar_mojo.problem import MAX_INTRINSICS, CameraModel

WIDTH: int = 1280
HEIGHT: int = 960


def default_intrinsics(model: CameraModel) -> Float64[ndarray, "12"]:
    """Realistic `camera.params` for a 1280 x 960 image, with non-zero distortion where the model has it."""
    values: dict[CameraModel, list[float]] = {
        CameraModel.PINHOLE: [900.0, 905.0, 640.0, 480.0],
        CameraModel.SIMPLE_RADIAL: [900.0, 640.0, 480.0, -0.05],
        CameraModel.OPENCV_FISHEYE: [450.0, 452.0, 640.0, 480.0, 0.02, -0.01, 0.003, -0.0005],
        CameraModel.FULL_OPENCV: [900.0, 905.0, 640.0, 480.0, -0.12, 0.05, 1e-3, -5e-4, 0.01, 0.02, 0.01, 0.005],
    }
    row: Float64[ndarray, "12"] = np.zeros(MAX_INTRINSICS)
    row[: len(values[model])] = values[model]
    return row


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The noise-free state the observations were rendered from."""

    intrinsics: Float64[ndarray, "n_cameras 12"]
    """True intrinsics."""
    sensor_from_rig: Float64[ndarray, "n_cameras 7"]
    """True rig extrinsics."""
    rig_from_world: Float64[ndarray, "n_frames 7"]
    """True frame poses."""
    points: Float64[ndarray, "n_points 3"]
    """True world points."""
