"""Ground-plane selection must respect G3T's gravity prior, not just plane size."""

import numpy as np
import pytest

from exo_calib.apis.calibrate_init import select_ground_plane

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


def test_no_acceptable_plane_falls_back_to_prior_and_lowest_camera() -> None:
    rng = np.random.default_rng(2)
    wall = _plane_points(np.array([0.0, 1.0, 0.0]), np.array([0.0, 3.0, 1.5]), 3.0, 3000, rng)
    camera_centers = np.array([[0.0, -2.0, 1.5], [1.0, -2.0, 1.4]])

    up, ground_point = select_ground_plane(wall, camera_centers, UP, distance_threshold=0.03, max_tilt_deg=60.0)

    assert np.dot(up, UP) > 0.99
    assert ground_point[2] == pytest.approx(camera_centers[:, 2].min(), abs=1e-9)
