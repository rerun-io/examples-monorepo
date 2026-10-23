import cv2
import numpy as np
import pytest
from jaxtyping import Float64, UInt8
from numpy import ndarray

from exo_calib.ops.lens import rectify, undistort
from exo_calib.types import FISHEYE, PINHOLE, Lens

K: Float64[ndarray, "3 3"] = np.array([[880.0, 0.0, 960.0], [0.0, 880.0, 540.0], [0.0, 0.0, 1.0]])
# Rays up to ~55 deg off axis: a GoPro's raw frame reaches past the pinhole's image there.
RAYS: Float64[ndarray, "n 3"] = np.column_stack((np.random.default_rng(0).uniform(-1.4, 1.4, (200, 2)), np.ones(200)))
PINHOLE_XY: Float64[ndarray, "n 2"] = (RAYS @ K.T)[:, :2]


def test_a_pinhole_lens_leaves_pixels_alone() -> None:
    np.testing.assert_array_equal(undistort(PINHOLE, PINHOLE_XY, K), PINHOLE_XY)


def test_a_kannala_brandt_lens_undistorts_what_it_distorts() -> None:
    lens = Lens("kannala_brandt", (0.0369, 0.05713, -0.05715, 0.01846, 0.0, 0.0, 0.0, 0.0))  # the catalog's 8-term order
    raw: Float64[ndarray, "n 2"] = cv2.fisheye.distortPoints(RAYS[:, None, :2], K, np.asarray(lens.coefficients[:4])).reshape(-1, 2)
    np.testing.assert_allclose(undistort(lens, raw, K), PINHOLE_XY, atol=1e-3)


def test_a_brown_conrady_lens_undistorts_what_it_distorts() -> None:
    lens = Lens("brown_conrady", (-0.13, 0.12, 0.0003, -0.0002, -0.019, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    camera: Float64[ndarray, "3 3"] = np.array([[830.0, 0.0, 640.0], [0.0, 830.0, 360.0], [0.0, 0.0, 1.0]])  # an Assembly101 1280x720 cam
    rng: np.random.Generator = np.random.default_rng(1)
    rays: Float64[ndarray, "n 3"] = np.column_stack((rng.uniform(-0.78, 0.78, 200), rng.uniform(-0.44, 0.44, 200), np.ones(200)))  # in frame
    raw: Float64[ndarray, "n 2"] = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), camera, np.asarray(lens.coefficients))[0].reshape(-1, 2)
    np.testing.assert_allclose(undistort(lens, raw, camera), (rays @ camera.T)[:, :2], atol=1e-3)


def test_the_fisheye_fallback_is_equidistant() -> None:
    raw: Float64[ndarray, "n 2"] = cv2.fisheye.distortPoints(RAYS[:, None, :2], K, np.zeros(4)).reshape(-1, 2)
    np.testing.assert_allclose(undistort(FISHEYE, raw, K), PINHOLE_XY, atol=1e-3)


def test_a_kannala_brandt_lens_with_terms_beyond_k4_is_refused() -> None:
    with pytest.raises(ValueError, match="k1..k4"):
        Lens("kannala_brandt", (0.27, 0.10, -0.33, 0.26, -0.098, 0.015, 0.001, -0.0007))


@pytest.mark.parametrize(
    "lens", [Lens("kannala_brandt", (0.0369, 0.05713, -0.05715, 0.01846)), Lens("brown_conrady", (-0.13, 0.12, 0.0003, -0.0002, -0.019))], ids=lambda lens: lens.model
)
def test_a_rectified_frame_shows_a_raw_pixel_where_undistort_puts_it(lens: Lens) -> None:
    raw_dot: Float64[ndarray, "1 2"] = np.array([[1500.0, 820.0]])  # off axis, where the lens bends most
    frame: UInt8[ndarray, "1 1080 1920 3"] = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
    frame[0, 818:823, 1498:1503] = 255

    rectified: UInt8[ndarray, "1 1080 1920 3"] = rectify(lens, frame, K)

    rows, cols = np.nonzero(rectified[0, :, :, 0] > 64)
    np.testing.assert_allclose([cols.mean(), rows.mean()], undistort(lens, raw_dot, K)[0], atol=1.0)


def test_a_pinhole_frame_is_not_resampled() -> None:
    frame: UInt8[ndarray, "2 4 6 3"] = np.random.default_rng(0).integers(0, 255, (2, 4, 6, 3), dtype=np.uint8)
    np.testing.assert_array_equal(rectify(PINHOLE, frame, K), frame)
