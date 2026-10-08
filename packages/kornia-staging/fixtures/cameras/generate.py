"""Generate independent camera fixtures with OpenCV and SimpleCV.

Run from the repo root with `pixi run -e simplecv --frozen python
packages/kornia-staging/fixtures/cameras/generate.py`.
"""

from pathlib import Path

import cv2
import numpy as np
import orjson
from jaxtyping import Float64
from numpy import ndarray
from serde.json import to_json
from simplecv.camera_fixture import ProjectionFixture, fixture_camera
from simplecv.camera_parameters import Fisheye62Parameters, KannalaBrandtDistortion
from simplecv.sensors.camera.fisheye62 import project_fisheye62


def main() -> None:
    """Write Brown4/5/8/12/14 fixtures beside this generator."""
    folder: Path = Path(__file__).parent
    intrinsics: list[float] = [430.0, 415.0, 320.0, 240.0]
    matrix: Float64[ndarray, "3 3"] = np.array([[430.0, 0.0, 320.0], [0.0, 415.0, 240.0], [0.0, 0.0, 1.0]])
    coefficients: list[float] = [0.03, -0.007, 0.003, -0.004, 0.001, 0.002, -0.001, 0.0002, 0.0007, -0.0003, -0.0008, 0.0004, 0.04, -0.03]
    points: Float64[ndarray, "n 3"] = np.array([[x, y, z] for x in [-0.8, -0.1, 0.0, 0.4, 1.0] for y in [-0.7, 0.0, 0.6] for z in [0.8, 2.0]])
    for count in [4, 5, 8, 12, 14]:
        distortion: Float64[ndarray, "d"] = np.array(coefficients[:count])
        pixels: Float64[ndarray, "n 2"] = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
        fixture: ProjectionFixture = ProjectionFixture(f"OpenCV {cv2.__version__} projectPoints", intrinsics, coefficients[:count], points, pixels)
        (folder / f"brown{count}_opencv.json").write_text(to_json(fixture, option=orjson.OPT_INDENT_2) + "\n")

    fish_coefficients: list[float] = [0.1, -0.02, 0.003, -0.0004, 0.00005, -0.000006, 0.002, -0.003]
    fish_points: Float64[ndarray, "n 3"] = points.copy()
    fish_points[:, 2] += 2.0
    camera = fixture_camera(intrinsics, KannalaBrandtDistortion(**dict(zip(["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"], fish_coefficients, strict=True))))
    assert isinstance(camera, Fisheye62Parameters)
    fish_pixels: Float64[ndarray, "n 2"] = project_fisheye62(fish_points, camera)
    assert np.isfinite(fish_pixels).all()
    fish_fixture: ProjectionFixture = ProjectionFixture("simplecv checked project_fisheye62", intrinsics, fish_coefficients, fish_points, fish_pixels)
    (folder / "fisheye62_simplecv.json").write_text(to_json(fish_fixture, option=orjson.OPT_INDENT_2) + "\n")


if __name__ == "__main__":
    main()
