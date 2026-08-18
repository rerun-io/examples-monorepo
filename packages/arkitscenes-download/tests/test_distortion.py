"""Behavior checks for Apple distortion evaluation and standard-model fitting."""

import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

from arkitscenes_download.ingest.distortion import AppleRadialPolynomial, OpenCVRationalFit, opencv_rational_from_apple

REAL_ULTRAWIDE_COEFFICIENTS_8: Float32[ndarray, "8"] = np.array(
    [-0.08353952, -6.350463, 5.248554, -1.9897834, 0.5831521, -0.10259675, 0.009266047, -0.0003309832],
    dtype=np.float32,
)


def test_apple_even_percent_polynomial_round_trips_real_frame_radii() -> None:
    """The empirically selected even-power forward model inverts within 0.1 px."""
    model: AppleRadialPolynomial = AppleRadialPolynomial(REAL_ULTRAWIDE_COEFFICIENTS_8)
    distorted_radii_n: Float32[ndarray, "n=5"] = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    rectified_radii_n: Float32[ndarray, "n=5"] = model.distorted_to_rectified_radius(distorted_radii_n)
    recovered_radii_n: Float32[ndarray, "n=5"] = model.rectified_to_distorted_radius(rectified_radii_n)

    np.testing.assert_allclose(
        rectified_radii_n,
        [0.0, 0.24992806, 0.49828497, 0.7403127, 0.97314256],
        atol=2e-7,
        rtol=0.0,
    )
    max_round_trip_error_px: float = float(np.max(np.abs(recovered_radii_n - distorted_radii_n)) * 2315.9016)
    assert max_round_trip_error_px < 0.1


def test_opencv_rational_fit_conforms_to_the_pinhole_principal_point() -> None:
    """The standard fit keeps K unchanged and stays below 0.01 px on segment 47115416."""
    center_reference_xy: Float32[ndarray, "2"] = np.array([1853.150634765625, 1371.03173828125], dtype=np.float32)
    K_native_33: Float32[ndarray, "3 3"] = np.array(
        [[245.39100646972656, 0.0, 321.8739929199219], [0.0, 245.39100646972656, 238.02699279785156], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    fit: OpenCVRationalFit = opencv_rational_from_apple(
        K_native_33,
        REAL_ULTRAWIDE_COEFFICIENTS_8,
        center_reference_xy,
        (3680, 2760),
        (640, 480),
    )

    assert fit.max_residual_px == pytest.approx(0.005637342, abs=5e-6)
    np.testing.assert_array_equal(fit.K_33, K_native_33)
    assert fit.distortion.k1 == pytest.approx(0.1422690471, rel=1e-3)
    assert fit.distortion.p1 == pytest.approx(-1.7744861e-5, rel=1e-3)
    assert fit.distortion.p2 == pytest.approx(-1.9211448e-5, rel=1e-3)


def test_opencv_fit_falls_back_to_polynomial_when_the_rational_denominator_misbehaves() -> None:
    """Segment 42898472's wide lens breaks the free rational denominator; the fit returns the k4..k6=0 polynomial form instead."""
    coefficients_8: Float32[ndarray, "8"] = np.array(
        [-0.00605488, -1.2373285, -0.27187824, 0.56518984, -0.19173706, 0.022887468, -0.00016848743, -9.48729e-5],
        dtype=np.float32,
    )
    K_native_33: Float32[ndarray, "3 3"] = np.array(
        [[1588.395, 0.0, 956.03998], [0.0, 1588.395, 739.82251], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    fit: OpenCVRationalFit = opencv_rational_from_apple(
        K_native_33,
        coefficients_8,
        np.array([2008.1995, 1554.1533], dtype=np.float32),
        (4032, 3024),
        (1920, 1440),
    )

    assert (fit.distortion.k4, fit.distortion.k5, fit.distortion.k6) == (0.0, 0.0, 0.0)
    assert fit.max_residual_px < 0.1
    np.testing.assert_array_equal(fit.K_33, K_native_33)


def test_opencv_rational_fit_rejects_a_non_invertible_calibration() -> None:
    """A pathological polynomial fails validation instead of returning a bad fit."""
    collapsing_8: Float32[ndarray, "8"] = np.array([-200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    K_native_33: Float32[ndarray, "3 3"] = np.array(
        [[245.0, 0.0, 320.0], [0.0, 245.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    with pytest.raises(ValueError):
        opencv_rational_from_apple(
            K_native_33,
            collapsing_8,
            np.array([1840.0, 1380.0], dtype=np.float32),
            (3680, 2760),
            (640, 480),
        )
