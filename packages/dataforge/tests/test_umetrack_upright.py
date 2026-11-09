"""UmeTrack ingests upright: a camera turned by whole quarter turns projects exactly where the turned image shows the point."""

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray
from simplecv.sensors.camera.fisheye62 import project_fisheye62

from dataforge.datasets.umetrack_layers import fisheye_camera
from dataforge.datasets.umetrack_source import Camera, quarter_turn_rotation, quarter_turns_from_angle, turn_camera
from dataforge.video_encoding import turn_filter

REAL: Camera = Camera(ImageSizeX=636, ImageSizeY=480, fx=237.3, fy=237.5, cx=316.8, cy=239.6, DistortionModel="FishEye62",
                      k1=-0.0207, k2=0.1058, k3=-0.0927, k4=0.0332, p1=-0.00073, p2=0.00056, k5=-0.0066, k6=0.0011)


def turn_pixels(uv: Float64[ndarray, "n 2"], width: int, height: int, turns: int) -> Float64[ndarray, "n 2"]:
    """Where pixel centres land after ``turns`` clockwise turns of a W x H image: (u, v) -> (H - 1 - v, u) per turn."""
    for _ in range(turns):
        uv = np.stack((height - 1 - uv[:, 1], uv[:, 0]), axis=1)
        width, height = height, width
    return uv


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_a_turned_camera_projects_like_the_turned_image(turns: int) -> None:
    rng = np.random.default_rng(turns)
    xyz: Float64[ndarray, "n 3"] = np.column_stack((rng.uniform(-0.6, 0.6, (500, 2)), rng.uniform(0.3, 1.0, 500)))
    original: Float64[ndarray, "n 2"] = project_fisheye62(xyz, fisheye_camera(REAL, np.eye(4)))
    expected: Float64[ndarray, "n 2"] = turn_pixels(original, REAL.ImageSizeX, REAL.ImageSizeY, turns)
    turned_camera: Camera = turn_camera(REAL, turns)
    rotated: Float64[ndarray, "n 3"] = xyz @ quarter_turn_rotation(turns).T  # the same points in the turned camera's axes
    observed: Float64[ndarray, "n 2"] = project_fisheye62(rotated, fisheye_camera(turned_camera, np.eye(4)))
    seen = np.isfinite(original).all(axis=1) & np.isfinite(observed).all(axis=1)
    assert seen.sum() > 300
    np.testing.assert_allclose(observed[seen], expected[seen], atol=1e-9)
    # The image bound is half-open, [0, W): a turned pixel centre sits in (-1, W - 1], so only points within a pixel of the
    # border may be in one image and not the other.
    disagree = np.isfinite(original).all(axis=1) != np.isfinite(observed).all(axis=1)
    near_border = (np.nan_to_num(expected, nan=0.0) < 1.0).any(axis=1) | (np.nan_to_num(observed, nan=0.0) < 1.0).any(axis=1) | ~np.isfinite(observed).all(axis=1)
    assert disagree.sum() <= 5 and bool(near_border[disagree].all())
    assert (turned_camera.ImageSizeX, turned_camera.ImageSizeY) == ((480, 636) if turns % 2 else (636, 480))


def test_the_release_angles_name_the_turns_and_leave_no_roll() -> None:
    # UmeTrack's camera_angles for its four cameras: 0, 90, 90, 180 degrees.
    assert [quarter_turns_from_angle(a) for a in (0.0, 90.0, 90.0, 180.0)] == [0, 3, 3, 2]


def test_ffmpeg_turns_are_transposes() -> None:
    assert [turn_filter(k) for k in range(4)] == ["", ",transpose=clock", ",transpose=clock,transpose=clock", ",transpose=cclock"]
