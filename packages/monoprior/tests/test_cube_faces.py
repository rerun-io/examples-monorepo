"""Unit tests for the 6-face cubemap face checks and spirula's prior encodings."""

import numpy as np
import pytest
from jaxtyping import Float32, Float64
from numpy import ndarray

from monopriors.cube_faces import CubeFace, depth_png_mm, normal_png_rgb

N: int = 800


def _face(name: str = "cube6_left", width: int = N, height: int = N, cx: float = (N - 1) / 2) -> CubeFace:
    k_matrix: Float64[ndarray, "3 3"] = np.array([[N / 2, 0.0, cx], [0.0, N / 2, (N - 1) / 2], [0.0, 0.0, 1.0]])
    return CubeFace(entity="/world/rig_00/cam_04", name=name, k_matrix=k_matrix, width=width, height=height)


@pytest.mark.parametrize("name", ["cube6_front0", "cube6_front1", "cube6_left", "cube6_right", "cube6_up", "cube6_down"])
def test_full_centred_faces_are_accepted(name: str) -> None:
    assert _face(name).name == name


def test_seam_split_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="cube6_"):
        _face("cube_0_left")


@pytest.mark.parametrize(("width", "height", "cx"), [(N, 480, (N - 1) / 2), (N, N, 79.5), (N, N, N / 2)])
def test_half_faces_and_off_centre_principal_points_are_rejected(width: int, height: int, cx: float) -> None:
    with pytest.raises(ValueError, match="full centred"):
        _face(width=width, height=height, cx=cx)


def test_depth_png_mm_marks_invalid_as_zero() -> None:
    depth_m: Float32[ndarray, "2 3"] = np.array([[1.0, 2.5, -1.0], [70.0, np.nan, 0.0004]], dtype=np.float32)
    valid: np.ndarray = np.array([[True, False, True], [True, True, True]])
    np.testing.assert_array_equal(depth_png_mm(depth_m, valid), np.array([[1000, 0, 0], [0, 0, 0]], dtype=np.uint16))


def test_normal_png_rgb_encoding() -> None:
    normals: Float32[ndarray, "1 2 3"] = np.array([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    valid: np.ndarray = np.array([[True, False]])
    np.testing.assert_array_equal(normal_png_rgb(normals, valid), np.array([[[128, 128, 0], [0, 0, 0]]], dtype=np.uint8))
