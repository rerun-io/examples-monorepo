"""Ground-plane selection must respect G3T's gravity prior, not just plane size."""

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.ops.ground_plane import select_ground_plane

UP = np.array([0.0, 0.0, 1.0])


def _plane_points(normal: np.ndarray, point: np.ndarray, extent: float, n: int, rng: np.random.Generator) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    coords = rng.uniform(-extent, extent, size=(n, 2))
    return point[None] + coords[:, :1] * u[None] + coords[:, 1:] * v[None] + rng.normal(0.0, 0.005, size=(n, 1)) * normal[None]


def test_dominant_wall_is_rejected_in_favour_of_the_floor() -> None:
    rng = np.random.default_rng(0)
    floor = _plane_points(UP, np.zeros(3), 2.0, 2000, rng)
    wall = _plane_points(np.array([0.0, 1.0, 0.0]), np.array([0.0, 3.0, 1.5]), 3.0, 8000, rng)
    cloud = np.concatenate([floor, wall])
    camera_centers = np.array([[0.0, -2.0, 1.5], [1.0, -2.0, 1.4], [-1.0, -2.0, 1.6]])

    up, ground_point = select_ground_plane(cloud, camera_centers, UP, distance_threshold=0.03, max_tilt_deg=60.0)

    assert np.dot(up, UP) > 0.99
    assert abs(ground_point[2]) < 0.05


def test_floor_below_cameras_is_accepted_first_try() -> None:
    rng = np.random.default_rng(1)
    cloud = _plane_points(UP, np.zeros(3), 2.0, 3000, rng)
    camera_centers = np.array([[0.0, -2.0, 1.5], [1.0, -2.0, 1.4]])

    up, _ = select_ground_plane(cloud, camera_centers, UP, distance_threshold=0.03, max_tilt_deg=60.0)

    assert np.dot(up, UP) > 0.99


def test_the_same_cloud_always_gives_the_same_floor() -> None:
    """The floor is the gauge of every later stage (Stage B projects its boxes through that rig), so it must be a function
    of the cloud: a thick floor (noise ~ the inlier distance), a wall and clutter make every random draw pick other inliers."""
    rng: np.random.Generator = np.random.default_rng(3)
    floor: Float64[ndarray, "6000 3"] = np.column_stack((rng.uniform(-3.0, 3.0, (2, 6000)).T, rng.normal(0.0, 0.03, 6000)))
    wall: Float64[ndarray, "4000 3"] = _plane_points(np.array([0.0, 1.0, 0.0]), np.array([0.0, 2.0, 1.2]), 3.0, 4000, rng)
    cloud: Float64[ndarray, "13000 3"] = np.concatenate([floor, wall, rng.uniform(-3.0, 3.0, size=(3000, 3))])
    camera_centers: Float64[ndarray, "2 3"] = np.array([[0.0, -2.0, 1.5], [1.0, -2.0, 1.4]])

    fits: list[tuple[Float64[ndarray, "3"], Float64[ndarray, "3"]]] = [
        select_ground_plane(cloud, camera_centers, UP, distance_threshold=0.06, max_tilt_deg=60.0) for _ in range(4)
    ]

    assert np.dot(fits[0][0], UP) > 0.99 and abs(fits[0][1][2]) < 0.02
    for up, ground_point in fits[1:]:
        np.testing.assert_array_equal(up, fits[0][0])
        np.testing.assert_array_equal(ground_point, fits[0][1])


@pytest.mark.parametrize("wall_first", [False, True])
def test_a_cloud_with_no_plane_left_to_fit_falls_back_instead_of_failing(wall_first: bool) -> None:
    """Collinear points span no plane: on its own, or as what is left once a wall's inliers are removed."""
    rng: np.random.Generator = np.random.default_rng(4)
    line: Float64[ndarray, "50 3"] = np.column_stack((np.linspace(-1.0, 1.0, 50), np.zeros(50), np.zeros(50)))
    wall: Float64[ndarray, "3000 3"] = _plane_points(np.array([0.0, 1.0, 0.0]), np.array([0.0, 3.0, 1.5]), 3.0, 3000, rng)
    cloud: Float64[ndarray, "n 3"] = np.concatenate([wall, line]) if wall_first else line
    camera_centers: Float64[ndarray, "2 3"] = np.array([[0.0, -2.0, 1.5], [1.0, -2.0, 1.4]])

    up, ground_point = select_ground_plane(cloud, camera_centers, UP, distance_threshold=0.03, max_tilt_deg=60.0)

    np.testing.assert_array_equal(up, UP)
    np.testing.assert_array_equal(ground_point, camera_centers[1])


def test_no_acceptable_plane_falls_back_to_prior_and_lowest_camera() -> None:
    rng = np.random.default_rng(2)
    wall = _plane_points(np.array([0.0, 1.0, 0.0]), np.array([0.0, 3.0, 1.5]), 3.0, 3000, rng)
    camera_centers = np.array([[0.0, -2.0, 1.5], [1.0, -2.0, 1.4]])

    up, ground_point = select_ground_plane(wall, camera_centers, UP, distance_threshold=0.03, max_tilt_deg=60.0)

    assert np.dot(up, UP) > 0.99
    assert ground_point[2] == pytest.approx(camera_centers[:, 2].min(), abs=1e-9)
