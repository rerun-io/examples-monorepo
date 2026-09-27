"""FOV-bounded FishEye62 projection against an independent scalar reference."""

import math

import numpy as np
import pytest

from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion, apply_radial_tangential_distortion
from simplecv.sensors.camera.base_camera import pixels_in_image
from simplecv.sensors.camera.fisheye62 import project_fisheye62
from simplecv.umetrack_temp.cameras import Camera


def test_scalar_reference_agrees_with_simplecv_distortion() -> None:
    # Scalar FishEye62 definition, including tangential distortion AFTER radial.
    x, y, z = 0.3, -0.2, 0.8
    radial = (0.1, -0.02, 0.003, -0.0004, 0.00005, -0.000006)
    p1, p2 = 0.002, -0.003
    radius = math.hypot(x, y)
    theta = math.atan2(radius, z)
    theta_d = theta * (1 + sum(k * theta ** (2 * i) for i, k in enumerate(radial, 1)))
    xr, yr = theta_d * x / radius, theta_d * y / radius
    rho2 = xr * xr + yr * yr
    expected = [xr + 2 * p2 * xr * yr + p1 * (rho2 + 2 * xr * xr), yr + 2 * p1 * xr * yr + p2 * (rho2 + 2 * yr * yr)]
    lens = KannalaBrandtDistortion(k1=radial[0], k2=radial[1], k3=radial[2], k4=radial[3], k5=radial[4], k6=radial[5], p1=p1, p2=p2)
    actual = apply_radial_tangential_distortion(lens, np.array([[theta * x / radius, theta * y / radius]], dtype=np.float64))
    np.testing.assert_allclose(actual[0], expected, atol=1e-14, rtol=0)


def test_projection_matches_scalar_and_camera_reference() -> None:
    lens = KannalaBrandtDistortion(k1=0.1, k2=-0.02, k3=0.003, k4=-0.0004, k5=0.00005, k6=-0.000006, p1=0.002, p2=-0.003)
    camera = Fisheye62Parameters(
        name="real",
        distortion=lens,
        extrinsics=Extrinsics(world_R_cam=np.eye(3), world_t_cam=np.zeros(3)),
        intrinsics=Intrinsics.from_focal_principal_point(camera_conventions="RDF", fl_x=220.0, fl_y=225.0, cx=318.0, cy=240.0, width=636, height=480),
    )
    point = np.array([[0.3, -0.2, 0.8]], dtype=np.float64)
    x, y, z = point[0]
    r = math.hypot(x, y)
    theta = math.atan2(r, z)
    theta_d = theta * (1 + sum(k * theta ** (2 * i) for i, k in enumerate((lens.k1, lens.k2, lens.k3, lens.k4, lens.k5, lens.k6), 1)))
    xr, yr = theta_d * x / r, theta_d * y / r
    rho2 = xr * xr + yr * yr
    expected = [
        220.0 * (xr + 2 * lens.p2 * xr * yr + lens.p1 * (rho2 + 2 * xr * xr)) + 318.0,
        225.0 * (yr + 2 * lens.p1 * xr * yr + lens.p2 * (rho2 + 2 * yr * yr)) + 240.0,
    ]
    pixels = project_fisheye62(point, camera)
    assert np.isfinite(pixels).all(axis=1).tolist() == [True]
    np.testing.assert_allclose(pixels[0], expected, atol=1e-6, rtol=0)
    # The legacy wrapper returns float32 pixels; compare at its storage precision.
    np.testing.assert_allclose(
        pixels.astype(np.float32), Camera(camera).camera_to_image(point.astype(np.float32)), atol=float(np.spacing(np.float32(expected[0]))), rtol=0
    )


def test_invalid_projections_become_nan() -> None:
    camera = Fisheye62Parameters(
        name="folding",
        distortion=KannalaBrandtDistortion(k1=-1.0, k2=0.0, k3=0.0, k4=0.0, k5=0.0, k6=0.0, p1=0.0, p2=0.0),
        extrinsics=Extrinsics(world_R_cam=np.eye(3), world_t_cam=np.zeros(3)),
        intrinsics=Intrinsics.from_focal_principal_point(camera_conventions="RDF", fl_x=220.0, fl_y=225.0, cx=50.0, cy=50.0, width=100, height=100),
    )
    # theta_max = sqrt(1/3); the folded ray maps BACK into the image and must still be rejected.
    points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [math.sin(1.0), 0, math.cos(1.0)], [0.4, 0, 1], [np.nan, 0, 1]], dtype=np.float64)
    pixels = project_fisheye62(points, camera)
    assert np.isfinite(pixels).all(axis=1).tolist() == [True, False, False, False, False, False]
    np.testing.assert_array_equal(pixels[0], [50, 50])
    assert np.isnan(pixels[1:]).all()


def test_pixels_in_image_is_half_open() -> None:
    pixels = np.array([[0.0, 0.0], [99.9, 49.9], [100.0, 10.0], [10.0, 50.0], [-0.1, 10.0], [10.0, np.nan], [np.inf, 1.0]], dtype=np.float32)
    assert pixels_in_image(pixels, 100, 50).tolist() == [True, True, False, False, False, False, False]


def test_a_camera_without_a_lens_is_refused() -> None:
    camera = Fisheye62Parameters(
        name="bare",
        distortion=None,
        extrinsics=Extrinsics(world_R_cam=np.eye(3), world_t_cam=np.zeros(3)),
        intrinsics=Intrinsics.from_focal_principal_point(camera_conventions="RDF", fl_x=220.0, fl_y=225.0, cx=50.0, cy=50.0, width=100, height=100),
    )
    with pytest.raises(ValueError, match="bare: project_fisheye62 needs a KannalaBrandt distortion"):
        project_fisheye62(np.array([[0.0, 0.0, 1.0]]), camera)
