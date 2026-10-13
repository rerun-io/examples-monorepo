"""Shared synthetic geometry for the unit lane: a ring of pinhole cameras looking at the origin (world z up)."""

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

K_ENGINE: Float64[ndarray, "3 3"] = np.array([[800.0, 0.0, 512.0], [0.0, 800.0, 376.0], [0.0, 0.0, 1.0]])
"""Intrinsics of a 1024 x 752 engine image."""


def look_at(camera_position: Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """cam_T_world (RDF camera) of a camera at ``camera_position`` looking at the origin."""
    forward: Float64[ndarray, "3"] = -camera_position / np.linalg.norm(camera_position)
    right: Float64[ndarray, "3"] = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    rotation: Float64[ndarray, "3 3"] = np.stack([right, np.cross(forward, right), forward], axis=0)
    cam_T_world: Float64[ndarray, "4 4"] = np.eye(4)
    cam_T_world[:3, :3] = rotation
    cam_T_world[:3, 3] = -rotation @ camera_position
    return cam_T_world


def camera_ring(views: int, radius_m: float = 4.0, height_m: float = 1.4) -> Float64[ndarray, "v 4 4"]:
    """cam_T_world of ``views`` cameras evenly spaced on a horizontal ring."""
    angles: Float64[ndarray, " v"] = np.linspace(0.0, 2.0 * np.pi, views, endpoint=False)
    return np.stack([look_at(np.array([radius_m * np.cos(a), radius_m * np.sin(a), height_m])) for a in angles])


def project(points: Float64[ndarray, "n 3"], cam_T_world: Float64[ndarray, "v 4 4"]) -> Float64[ndarray, "v n 2"]:
    """Pixels of ``points`` in every camera of the ring (``K_ENGINE``)."""
    camera: Float64[ndarray, "v n 3"] = np.einsum("vij,nj->vni", cam_T_world[:, :3, :3], points) + cam_T_world[:, None, :3, 3]
    pixels: Float64[ndarray, "v n 3"] = camera @ K_ENGINE.T
    return pixels[..., :2] / pixels[..., 2:]
