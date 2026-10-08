"""Independent fixtures shared with the staged Rust camera kernels."""

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray
from serde.json import from_json

from simplecv.camera_fixture import ProjectionFixture, fixture_camera
from simplecv.camera_parameters import BrownConradyDistortion, PinholeParameters
from simplecv.sensors.camera.brown_conrady import project_brown_conrady_grid

FIXTURES: Path = Path(__file__).parents[2] / "kornia-staging" / "fixtures" / "cameras"


@pytest.mark.parametrize("count", [4, 5, 8, 12, 14])
def test_brown_shared_opencv_fixture(count: int) -> None:
    """Both Rust and NumPy consume the same externally generated projections."""
    fixture: ProjectionFixture = from_json(ProjectionFixture, (FIXTURES / f"brown{count}_opencv.json").read_text())
    points: Float64[ndarray, "n 3"] = fixture.points
    pixels: Float64[ndarray, "n 2"] = fixture.pixels
    names: list[str] = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tau_x", "tau_y"]
    coefficients: list[float] = fixture.distortion + [0.0] * (14 - count)
    lens: BrownConradyDistortion = BrownConradyDistortion(**dict(zip(names, coefficients, strict=True)))
    camera = fixture_camera(fixture.intrinsics, lens)
    assert isinstance(camera, PinholeParameters)
    actual: Float64[ndarray, "1 1 n 2"] = project_brown_conrady_grid(points[None], [camera], filter_invalid=False)
    np.testing.assert_allclose(actual[0, 0], pixels, atol=2e-10, rtol=0.0)


def test_fisheye62_shared_fixture() -> None:
    """Keep the checked NumPy projection tied to Rust's shared reference."""
    from simplecv.camera_parameters import Fisheye62Parameters, KannalaBrandtDistortion
    from simplecv.sensors.camera.fisheye62 import project_fisheye62

    fixture: ProjectionFixture = from_json(ProjectionFixture, (FIXTURES / "fisheye62_simplecv.json").read_text())
    points: Float64[ndarray, "n 3"] = fixture.points
    pixels: Float64[ndarray, "n 2"] = fixture.pixels
    lens: KannalaBrandtDistortion = KannalaBrandtDistortion(**dict(zip(["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"], fixture.distortion, strict=True)))
    camera = fixture_camera(fixture.intrinsics, lens)
    assert isinstance(camera, Fisheye62Parameters)
    np.testing.assert_array_equal(project_fisheye62(points, camera), pixels)
