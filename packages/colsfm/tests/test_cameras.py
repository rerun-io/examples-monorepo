"""cuSFM camera parameters must convert to the same COLMAP cameras the blob wrote."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap

from colsfm.cameras import colmap_camera, fit_distortion_coefficients
from colsfm.frames_meta import CameraParams
from colsfm.geometry import rigid3d_from_axis_angle_degrees


def read_blob_cameras(path: Path) -> dict[int, pycolmap.Camera]:
    """Load the blob's own `cameras.txt` with COLMAP's reader.

    Args:
        path: Directory holding `cameras.txt`, `images.txt` and `points3D.txt`.

    Returns:
        Cameras keyed by `camera_id`.
    """
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction(str(path))
    return dict(reconstruction.cameras)


def assert_cameras_match(ours: dict[int, pycolmap.Camera], theirs: dict[int, pycolmap.Camera]) -> None:
    """Assert two camera sets agree on ids, models, sizes and parameters.

    Args:
        ours: Cameras produced by `colsfm.cameras`.
        theirs: Cameras read from a blob-written model.
    """
    assert sorted(ours) == sorted(theirs)
    for camera_id, expected in theirs.items():
        actual: pycolmap.Camera = ours[camera_id]
        assert actual.model.name == expected.model.name
        assert (actual.width, actual.height) == (expected.width, expected.height)
        assert np.allclose(actual.params, expected.params, atol=1e-9)


def distorted_pinhole_camera(distortion_coefficients: np.ndarray) -> CameraParams:
    """Build a `DISTORTED_PINHOLE` camera; no shipped dataset uses one.

    Args:
        distortion_coefficients: Coefficients to attach, of any length.

    Returns:
        Camera parameters with a known 3x3 intrinsic matrix.
    """
    return CameraParams(
        camera_params_id=3,
        sensor_name="synthetic",
        model_name="",
        sensor_id=3,
        frequency_hz=30.0,
        vehicle_T_cam=rigid3d_from_axis_angle_degrees(
            axis_xyz=np.array([0.0, 0.0, 1.0]), angle_degrees=0.0, translation_xyz=np.zeros(3)
        ),
        image_width=1280,
        image_height=720,
        projection_matrix=None,
        camera_matrix=np.array([[500.0, 0.0, 640.0], [0.0, 501.0, 360.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=distortion_coefficients,
        projection_model="DISTORTED_PINHOLE",
        rolling_shutter_delay_microseconds=0,
    )


def test_distorted_pinhole_becomes_full_opencv_with_twelve_parameters() -> None:
    """`DISTORTED_PINHOLE` maps to `FULL_OPENCV`: `fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6`."""
    coefficients: np.ndarray = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    camera: pycolmap.Camera = colmap_camera(distorted_pinhole_camera(coefficients))
    assert camera.model.name == "FULL_OPENCV"
    assert np.allclose(camera.params, np.concatenate([[500.0, 501.0, 640.0, 360.0], coefficients]))


def test_too_few_distortion_coefficients_are_zero_padded() -> None:
    """export.md §3.3: cuSFM warns and pads rather than failing."""
    fitted: np.ndarray = fit_distortion_coefficients(distorted_pinhole_camera(np.array([0.1, 0.2])), 8)
    assert np.allclose(fitted, np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_too_many_distortion_coefficients_are_truncated() -> None:
    """Only the first `wanted` coefficients survive, matching the blob's warning path."""
    fitted: np.ndarray = fit_distortion_coefficients(distorted_pinhole_camera(np.arange(12.0)), 8)
    assert np.allclose(fitted, np.arange(8.0))


def test_absent_distortion_coefficients_become_zeros() -> None:
    """An empty `distortion_coefficients` yields an undistorted `FULL_OPENCV` camera."""
    camera: pycolmap.Camera = colmap_camera(distorted_pinhole_camera(np.array([])))
    assert np.allclose(camera.params[4:], np.zeros(8))


