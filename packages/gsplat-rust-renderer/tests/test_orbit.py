"""Scene orbit contracts: fitted from the capture cameras, written as gsplat-cli CameraSpecs."""

import json
from pathlib import Path

import numpy as np
from jaxtyping import Float64

from gsplat_rust_renderer.orbit import Orbit, look_at, orbit_from_cameras, orbit_specs, write_orbit_json

CENTER: Float64[np.ndarray, "3"] = np.array([0.4, 1.5, 1.1])
UP: Float64[np.ndarray, "3"] = np.array([0.0, -0.8, -0.6])


def ring(count: int, radius: float, height: float) -> Float64[np.ndarray, "n 4 4"]:
    """Capture cameras on a tilted ring, all looking at CENTER."""
    first: Float64[np.ndarray, "3"] = np.array([1.0, 0.0, 0.0])
    second: Float64[np.ndarray, "3"] = np.cross(UP, first)
    angles: Float64[np.ndarray, "n"] = np.linspace(0.3, 0.3 + 2.0 * np.pi, count, endpoint=False)
    return np.stack([look_at(CENTER + radius * (np.cos(a) * first + np.sin(a) * second) + height * UP, CENTER, UP) for a in angles])


def test_orbit_recovers_centre_up_radius_and_elevation() -> None:
    orbit: Orbit = orbit_from_cameras(ring(24, 3.0, 1.0), frames=300)
    np.testing.assert_allclose(orbit.center, CENTER, atol=1e-9)
    np.testing.assert_allclose(orbit.up, UP, atol=1e-9)
    assert abs(orbit.radius - 3.0) < 1e-9 and abs(orbit.elevation - 1.0) < 1e-9
    assert orbit.world_from_cameras.shape == (300, 4, 4)


def test_orbit_poses_look_at_the_centre_and_turn_a_quarter_in_75_frames() -> None:
    orbit: Orbit = orbit_from_cameras(ring(16, 2.0, 0.5), frames=300)
    poses: Float64[np.ndarray, "n 4 4"] = orbit.world_from_cameras
    for pose in poses[::37]:
        rotation: Float64[np.ndarray, "3 3"] = pose[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) > 0.0
        toward: Float64[np.ndarray, "3"] = (CENTER - pose[:3, 3]) / np.linalg.norm(CENTER - pose[:3, 3])
        np.testing.assert_allclose(pose[:3, 2], toward, atol=1e-12)
        assert pose[:3, 1] @ UP < 0.0, "image y points down, away from world up"
    first: Float64[np.ndarray, "3"] = poses[0, :3, 3] - CENTER
    quarter: Float64[np.ndarray, "3"] = poses[75, :3, 3] - CENTER
    first_flat: Float64[np.ndarray, "3"] = first - (first @ UP) * UP
    quarter_flat: Float64[np.ndarray, "3"] = quarter - (quarter @ UP) * UP
    assert abs(first_flat @ quarter_flat) < 1e-9
    np.testing.assert_allclose(np.linalg.norm(first_flat), 2.0, atol=1e-9)


def test_orbit_json_is_a_camera_spec_array(tmp_path: Path) -> None:
    orbit: Orbit = orbit_from_cameras(ring(8, 1.0, 0.2), frames=4)
    path: Path = tmp_path / "orbit.json"
    write_orbit_json(path, orbit_specs(orbit, 1920, 1080, fov_y=np.pi / 3.0))
    records: list[dict] = json.loads(path.read_text())
    assert len(records) == 4
    assert set(records[0]) == {"world_from_camera", "width", "height", "fx", "fy", "cx", "cy", "model"}
    assert records[0]["model"] == "Pinhole" and (records[0]["cx"], records[0]["cy"]) == (960.0, 540.0)
    np.testing.assert_allclose(records[0]["fy"], 540.0 / np.tan(np.pi / 6.0))
    np.testing.assert_allclose(records[2]["world_from_camera"], orbit.world_from_cameras[2])
