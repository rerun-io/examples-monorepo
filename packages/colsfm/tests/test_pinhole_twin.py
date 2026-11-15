"""The pinhole twin: undistorted keypoints land where the ideal camera projects the same rays."""

from __future__ import annotations

import numpy as np
import pycolmap

from colsfm.cameras import colmap_camera
from colsfm.frames_meta import CameraParams, FramesMeta
from colsfm.pinhole_twin import make_pinhole_twin, twin_camera, undistort_pixels


def _distorted() -> CameraParams:
    """An MSD-like 480x640 camera with OpenCV rational distortion (the MGO07 cam0 coefficients)."""
    return CameraParams(
        camera_params_id=0,
        sensor_name="cam0",
        model_name="",
        sensor_id=0,
        frequency_hz=30.0,
        vehicle_T_cam=pycolmap.Rigid3d(),
        image_width=480,
        image_height=640,
        projection_matrix=None,
        camera_matrix=np.array([[270.0, 0.0, 240.0], [0.0, 270.0, 320.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=np.array([0.30, -0.02, 6e-5, 2.5e-4, 0.016, 0.575, -0.063, 0.034]),
        projection_model="DISTORTED_PINHOLE",
        rolling_shutter_delay_microseconds=0,
    )


def test_undistorted_pixels_are_the_pinhole_projection_of_the_same_rays() -> None:
    raw_params: CameraParams = _distorted()
    raw: pycolmap.Camera = colmap_camera(raw_params)
    twin: pycolmap.Camera = colmap_camera(twin_camera(raw_params))
    assert twin.model.name == "PINHOLE" and np.allclose(twin.params, raw.params[:4])
    rays = np.random.default_rng(0).uniform(-0.6, 0.6, size=(200, 2))
    raw_xy = np.asarray(raw.img_from_cam(np.hstack([rays, np.ones((200, 1))])))
    twin_xy = undistort_pixels(raw, twin, raw_xy)
    expected = np.asarray(twin.img_from_cam(np.hstack([rays, np.ones((200, 1))])))
    np.testing.assert_allclose(twin_xy, expected, atol=1e-3)


def test_a_rig_without_distorted_pinholes_gets_no_twin(galileo_input: FramesMeta, tmp_path) -> None:  # noqa: ANN001
    assert make_pinhole_twin(galileo_input, tmp_path / "missing.db", tmp_path / "twin.db") is None
