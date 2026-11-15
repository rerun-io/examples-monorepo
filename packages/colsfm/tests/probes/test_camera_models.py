"""Probe (f): camera model ids and parameter layouts, against docs/camera.md."""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float
from numpy import ndarray

# The cuSFM -> COLMAP mapping table in docs/camera.md, transcribed.  cuSFM's
# FTHETA_WINDSHIELD has no COLMAP equivalent and is out of scope.
CUSFM_TO_COLMAP: dict[str, str] = {
    "PINHOLE": "PINHOLE",
    "DISTORTED_PINHOLE": "FULL_OPENCV",
    "OPENCV_FISHEYE": "OPENCV_FISHEYE",
}
"""cuSFM `camera_projection_model_type` to COLMAP camera model name."""

EXPECTED_PARAM_LAYOUT: dict[str, list[str]] = {
    "PINHOLE": ["fx", "fy", "cx", "cy"],
    "FULL_OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
    "OPENCV_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
}
"""Parameter order docs/camera.md claims for each COLMAP model."""


@pytest.mark.parametrize("model_name", sorted(EXPECTED_PARAM_LAYOUT))
def test_param_layout_matches_docs_camera_md(model_name: str) -> None:
    """`Camera.params_info` reproduces the ordering docs/camera.md documents.

    The FULL_OPENCV row is the one that matters: cuSFM's `distortion_coefficients`
    array is OpenCV rational order `[k1, k2, p1, p2, k3, k4, k5, k6]`, which is
    exactly COLMAP's params[4:12] — a straight copy, no permutation.
    """
    camera: pycolmap.Camera = pycolmap.Camera.create_from_model_name(
        1, model_name, 500.0, 1920, 1080
    )
    layout: list[str] = [name.strip() for name in camera.params_info.split(",")]
    assert layout == EXPECTED_PARAM_LAYOUT[model_name]
    assert len(camera.params) == len(EXPECTED_PARAM_LAYOUT[model_name])
    assert camera.verify_params()


@pytest.mark.parametrize(
    ("model_name", "focal_idxs", "principal_idxs", "extra_idxs"),
    [
        ("PINHOLE", [0, 1], [2, 3], []),
        ("FULL_OPENCV", [0, 1], [2, 3], [4, 5, 6, 7, 8, 9, 10, 11]),
        ("OPENCV_FISHEYE", [0, 1], [2, 3], [4, 5, 6, 7]),
    ],
)
def test_parameter_groups(
    model_name: str, focal_idxs: list[int], principal_idxs: list[int], extra_idxs: list[int]
) -> None:
    """`refine_focal_length` / `refine_principal_point` / `refine_extra_params` map to these."""
    camera: pycolmap.Camera = pycolmap.Camera.create_from_model_name(
        1, model_name, 500.0, 1920, 1080
    )
    assert camera.focal_length_idxs() == focal_idxs
    assert camera.principal_point_idxs() == principal_idxs
    assert camera.extra_params_idxs() == extra_idxs
    assert camera.metadata_params_idxs() == []


def test_model_ids_are_stable() -> None:
    """The numeric model ids written into `cameras.bin` are the classic COLMAP ones."""
    assert int(pycolmap.CameraModelId.SIMPLE_PINHOLE) == 0
    assert int(pycolmap.CameraModelId.PINHOLE) == 1
    assert int(pycolmap.CameraModelId.SIMPLE_RADIAL) == 2
    assert int(pycolmap.CameraModelId.RADIAL) == 3
    assert int(pycolmap.CameraModelId.OPENCV) == 4
    assert int(pycolmap.CameraModelId.OPENCV_FISHEYE) == 5
    assert int(pycolmap.CameraModelId.FULL_OPENCV) == 6
    assert int(pycolmap.CameraModelId.FOV) == 7
    assert int(pycolmap.CameraModelId.SIMPLE_RADIAL_FISHEYE) == 8
    assert int(pycolmap.CameraModelId.RADIAL_FISHEYE) == 9
    assert int(pycolmap.CameraModelId.THIN_PRISM_FISHEYE) == 10


def test_ftheta_windshield_has_no_colmap_equivalent() -> None:
    """cuSFM's FTHETA_WINDSHIELD really is unrepresentable, as docs/camera.md says."""
    available: set[str] = {
        name for name in dir(pycolmap.CameraModelId) if name.isupper()
    }
    assert "FTHETA" not in available
    assert not any("THETA" in name for name in available)
    # COLMAP 4.2 does add models COLMAP 3.x lacked; note them so the mapping
    # table can be revisited if a better fisheye fit is wanted.
    assert {"FISHEYE", "SIMPLE_FISHEYE", "EUCM", "DIVISION", "EQUIRECTANGULAR"} <= available


@pytest.mark.parametrize("model_name", sorted(EXPECTED_PARAM_LAYOUT))
def test_projection_round_trips(model_name: str) -> None:
    """`img_from_cam` and `cam_ray_from_img` invert each other for each model.

    This is what guarantees the cuSFM intrinsics survive the copy: if the
    parameter ordering were wrong, the round trip would not close.
    """
    camera: pycolmap.Camera = pycolmap.Camera.create_from_model_name(
        1, model_name, 500.0, 1920, 1080
    )
    params: Float[ndarray, " n"] = camera.params.copy()
    if model_name == "FULL_OPENCV":
        params[4:] = [0.1, -0.02, 0.001, 0.002, 0.005, 0.01, -0.001, 0.0005]
    elif model_name == "OPENCV_FISHEYE":
        params[4:] = [-0.02, 0.004, -0.0008, 0.0001]
    camera.params = params
    assert camera.verify_params()

    rng: np.random.Generator = np.random.default_rng(2)
    points_in_cam: Float[ndarray, "n 3"] = np.stack(
        [rng.uniform(-1.0, 1.0, 50), rng.uniform(-0.8, 0.8, 50), rng.uniform(3.0, 8.0, 50)], axis=1
    )
    pixels: Float[ndarray, "n 2"] = camera.img_from_cam(points_in_cam)
    rays: Float[ndarray, "n 3"] = camera.cam_ray_from_img(pixels)
    normalised: Float[ndarray, "n 3"] = points_in_cam / np.linalg.norm(
        points_in_cam, axis=1, keepdims=True
    )
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    assert np.allclose(rays, normalised, atol=1e-6)


def test_full_opencv_accepts_the_cusfm_coefficient_block() -> None:
    """A cuSFM DISTORTED_PINHOLE calibration copies into FULL_OPENCV verbatim."""
    camera_matrix: Float[ndarray, "3 3"] = np.array(
        [[500.0, 0.0, 960.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]]
    )
    distortion: Float[ndarray, " 8"] = np.array(
        [0.1, -0.2, 0.001, 0.002, 0.05, 0.01, -0.01, 0.005]
    )
    camera: pycolmap.Camera = pycolmap.Camera.create_from_model_name(
        1, CUSFM_TO_COLMAP["DISTORTED_PINHOLE"], 500.0, 1920, 1080
    )
    camera.params = np.concatenate(
        [
            [camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]],
            distortion,
        ]
    )
    assert camera.verify_params()
    assert np.allclose(camera.calibration_matrix(), camera_matrix)
    assert np.allclose(camera.params[camera.extra_params_idxs()], distortion)


def test_opencv_fisheye_takes_only_k1_to_k4() -> None:
    """RoboCap's Fisheye62 populates k1..k4 only, so the mapping is lossless.

    OPENCV_FISHEYE has exactly four extra parameters; there is no slot for the
    tangential or thin-prism terms a Fisheye62 could in principle carry.
    """
    camera: pycolmap.Camera = pycolmap.Camera.create_from_model_name(
        1, "OPENCV_FISHEYE", 400.0, 1920, 1080
    )
    assert len(camera.extra_params_idxs()) == 4
    camera.params = [400.0, 400.0, 960.0, 540.0, -0.02, 0.004, -0.0008, 0.0001]
    assert camera.verify_params()
    assert camera.is_perspective_fisheye()
    assert not camera.is_perspective_pinhole()
    assert not camera.is_undistorted()
