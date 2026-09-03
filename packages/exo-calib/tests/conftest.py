"""Synthetic multi-view scenes shared by the geometry tests."""

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.correspondences import ObservationSet


def ring_rig(n_views: int, radius_m: float) -> Float64[ndarray, "v 4 4"]:
    """Cameras on a horizontal ring of ``radius_m`` around the origin, each looking at the origin (RDF).

    Returns:
        Float64 world-to-camera transforms with shape ``(v, 4, 4)``.
    """
    angles: Float64[ndarray, "v"] = np.arange(n_views, dtype=np.float64) * (2.0 * np.pi / float(n_views))
    centers: Float64[ndarray, "v 3"] = np.column_stack((radius_m * np.cos(angles), radius_m * np.sin(angles), np.zeros(n_views, dtype=np.float64)))
    cam_T_world: Float64[ndarray, "v 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], n_views, axis=0)
    world_up: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for view_idx in range(n_views):
        forward: Float64[ndarray, "3"] = -centers[view_idx] / np.linalg.norm(centers[view_idx])
        right: Float64[ndarray, "3"] = np.cross(forward, world_up)
        down: Float64[ndarray, "3"] = np.cross(forward, right)
        rotation: Float64[ndarray, "3 3"] = np.stack((right, down, forward))
        cam_T_world[view_idx, :3, :3] = rotation
        cam_T_world[view_idx, :3, 3] = -rotation @ centers[view_idx]
    return cam_T_world


def observe_points(
    points_xyz: Float64[ndarray, "n 3"], intrinsics: Float64[ndarray, "v 3 3"], cam_T_world: Float64[ndarray, "v 4 4"]
) -> ObservationSet:
    """Project every point into every view with unit confidence, one point per frame and joint 0 (point-major, view-minor order)."""
    n_views: int = cam_T_world.shape[0]
    n_points: int = points_xyz.shape[0]
    points_homo: Float64[ndarray, "n 4"] = np.column_stack((points_xyz, np.ones(n_points, dtype=np.float64)))
    points_cam: Float64[ndarray, "v n 3"] = np.einsum("vij,nj->vni", cam_T_world[:, :3], points_homo)
    projected: Float64[ndarray, "v n 3"] = np.einsum("vij,vnj->vni", intrinsics, points_cam)
    projected_xy: Float64[ndarray, "v n 2"] = projected[:, :, :2] / projected[:, :, 2:3]
    return ObservationSet(
        point_frame_idx=np.arange(n_points, dtype=np.int64),
        point_joint_idx=np.zeros(n_points, dtype=np.int64),
        obs_point_idx=np.repeat(np.arange(n_points, dtype=np.int64), n_views),
        obs_view_idx=np.tile(np.arange(n_views, dtype=np.int64), n_points),
        obs_xy=projected_xy.transpose(1, 0, 2).reshape(-1, 2),
        obs_conf=np.ones(n_points * n_views, dtype=np.float64),
    )
