"""One orbit per scene, derived from its capture cameras and shared by the viewer, `gsplat probe` and `gsplat speed`."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from serde import serde
from serde.json import to_json


@serde
@dataclass(frozen=True, slots=True)
class CameraSpec:
    """gsplat-cli `CameraSpec`: OpenCV world-from-camera (row-major), pixel intrinsics, lens model."""

    world_from_camera: Float64[ndarray, "4 4"]
    """Row-major camera-to-world pose, OpenCV axes."""
    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""
    fx: float
    """Horizontal focal length in pixels."""
    fy: float
    """Vertical focal length in pixels."""
    cx: float
    """Principal point x in pixels."""
    cy: float
    """Principal point y in pixels."""
    model: str = "Pinhole"
    """Lens model; orbits are pinhole."""


@dataclass(frozen=True, slots=True)
class Orbit:
    """A closed circular camera path around the point the capture cameras look at."""

    center: Float64[ndarray, "3"]
    """Least-squares closest point to every camera's optical axis."""
    up: Float64[ndarray, "3"]
    """Unit world up: minus the mean camera y (down) axis."""
    radius: float
    """Median horizontal distance of the cameras from the centre."""
    elevation: float
    """Median camera height above the centre along `up`."""
    world_from_cameras: Float64[ndarray, "n 4 4"]
    """OpenCV camera-to-world poses, evenly spaced in azimuth, looking at the centre; pose 0 faces camera 0."""


def look_at(position: Float64[ndarray, "3"], target: Float64[ndarray, "3"], up: Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """OpenCV camera-to-world pose (x right, y down, z forward) at `position` looking at `target`."""
    forward: Float64[ndarray, "3"] = (target - position) / np.linalg.norm(target - position)
    right: Float64[ndarray, "3"] = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down: Float64[ndarray, "3"] = np.cross(forward, right)
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = position
    return pose


def orbit_from_cameras(world_from_cameras: Float64[ndarray, "n 4 4"], frames: int) -> Orbit:
    """Fit the scene orbit to capture cameras (OpenCV axes).

    Args:
        world_from_cameras: Float64[ndarray, "n 4 4"] camera-to-world poses of the capture.
        frames: number of poses on the closed path.
    """
    origins: Float64[ndarray, "n 3"] = world_from_cameras[:, :3, 3]
    axes: Float64[ndarray, "n 3"] = world_from_cameras[:, :3, 2]
    projectors: Float64[ndarray, "n 3 3"] = np.eye(3) - axes[:, :, None] * axes[:, None, :]
    center: Float64[ndarray, "3"] = np.linalg.solve(projectors.sum(axis=0), np.einsum("nij,nj->i", projectors, origins))
    up: Float64[ndarray, "3"] = -world_from_cameras[:, :3, 1].mean(axis=0)
    up /= np.linalg.norm(up)
    offsets: Float64[ndarray, "n 3"] = origins - center
    heights: Float64[ndarray, "n"] = offsets @ up
    horizontal: Float64[ndarray, "n 3"] = offsets - heights[:, None] * up
    first: Float64[ndarray, "3"] = horizontal[0] / np.linalg.norm(horizontal[0])
    second: Float64[ndarray, "3"] = np.cross(up, first)
    radius: float = float(np.median(np.linalg.norm(horizontal, axis=1)))
    elevation: float = float(np.median(heights))
    angles: Float64[ndarray, "m"] = np.arange(frames) * (2.0 * np.pi / frames)
    poses: Float64[ndarray, "m 4 4"] = np.stack(
        [look_at(center + radius * (np.cos(a) * first + np.sin(a) * second) + elevation * up, center, up) for a in angles]
    )
    return Orbit(center=center, up=up, radius=radius, elevation=elevation, world_from_cameras=poses)


def orbit_specs(orbit: Orbit, width: int, height: int, fov_y: float) -> list[CameraSpec]:
    """Square-pixel pinhole specs of every orbit pose at one resolution and vertical field of view."""
    focal: float = 0.5 * height / float(np.tan(0.5 * fov_y))
    return [CameraSpec(pose, width, height, focal, focal, 0.5 * width, 0.5 * height) for pose in orbit.world_from_cameras]


def write_orbit_json(path: Path, specs: list[CameraSpec]) -> None:
    """Write the `CameraSpec` array that `gsplat speed --path` and `gsplat render --camera` read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(specs))
