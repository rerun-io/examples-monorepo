"""The spiral camera path through a single-image scene (RDF: x right, y down, z forward; the input camera is the origin)."""

import numpy as np
from jaxtyping import Float32, Float64
from numpy import ndarray


def backproject(depth: Float32[ndarray, "h w"], K: Float64[ndarray, "3 3"]) -> Float64[ndarray, "h w 3"]:
    """Camera-frame points of every pixel: ``depth * K^-1 [u v 1]``."""
    v, u = np.indices(depth.shape)
    uv1: Float64[ndarray, "h w 3"] = np.stack([u, v, np.ones_like(u)], axis=-1) * depth[..., None].astype(np.float64)
    return np.einsum("ab,hwb->hwa", np.linalg.inv(K), uv1)


def scene_radius(depth: Float32[ndarray, "h w"], K: Float64[ndarray, "3 3"], low_pct: float = 5.0, high_pct: float = 50.0) -> float:
    """Mean over x, y, z of the spread between two percentiles of the back-projected points: the path's scale."""
    points: Float64[ndarray, "n 3"] = backproject(depth, K).reshape(-1, 3)
    return float(np.mean(np.percentile(points, high_pct, axis=0) - np.percentile(points, low_pct, axis=0)))


def look_at(position: Float64[ndarray, "3"], target: Float64[ndarray, "3"], up: Float64[ndarray, "3"]) -> Float64[ndarray, "3 3"]:
    """``world_R_cam`` of a camera at ``position`` looking at ``target``; ``up`` points up in the world (image y is down)."""
    forward: Float64[ndarray, "3"] = (target - position) / np.linalg.norm(target - position)
    down: Float64[ndarray, "3"] = -up / np.linalg.norm(up)
    right: Float64[ndarray, "3"] = np.cross(down, forward)
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(forward, right), forward])


def spiral(
    depth: Float32[ndarray, "h w"],
    K: Float64[ndarray, "3 3"],
    n: int,
    forward_ratio: float = 0.3,
    backward_ratio: float = 0.4,
    rot_ratio: float = 0.3,
    look_at_ratio: float = 0.5,
) -> Float64[ndarray, "n 4 4"]:
    """``cam_T_world`` of ``n`` cameras on a spiral around the view of ``depth`` (the first and last at the origin).

    The radius comes from the scene's extent (``scene_radius``); landscape images move fully sideways, portrait ones 0.6x,
    both 0.3x vertically; moving forward scales depth by ``forward_ratio``, backward by ``backward_ratio``.
    """
    height, width = depth.shape
    radius: float = scene_radius(depth, K)
    t: Float64[ndarray, "n"] = np.linspace(0, 1, n)
    r: Float64[ndarray, "n"] = np.sin(2 * np.pi * t) * radius * rot_ratio
    theta: Float64[ndarray, "n"] = 2 * np.pi * t * n
    z: Float64[ndarray, "n"] = -r
    z[z < 0] *= forward_ratio
    z[z > 0] *= backward_ratio
    positions: Float64[ndarray, "n 3"] = np.stack([r * np.cos(theta) * (1.0 if width > height else 0.6), r * np.sin(theta) * 0.3, z], axis=1)
    target: Float64[ndarray, "3"] = np.array([0.0, 0.0, radius * look_at_ratio])
    world_T_cam: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (n, 1, 1))
    for i, position in enumerate(positions):
        world_T_cam[i, :3, :3] = look_at(position, target, np.array([0.0, -1.0, 0.0]))
        world_T_cam[i, :3, 3] = position
    return np.linalg.inv(world_T_cam)[::-1].copy()
