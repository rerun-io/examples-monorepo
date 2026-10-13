"""The floor of a metric scene cloud: the largest RANSAC plane that agrees with a gravity prior."""

import numpy as np
import open3d as o3d
from jaxtyping import Bool, Float64
from numpy import ndarray

GROUND_PLANE_ATTEMPTS: int = 5
"""Largest RANSAC planes tried before falling back to the gravity prior and the lowest camera."""


def select_ground_plane(
    cloud: Float64[ndarray, "n 3"],
    camera_centers: Float64[ndarray, "v 3"],
    up_prior: Float64[ndarray, "3"],
    *,
    distance_threshold: float,
    max_tilt_deg: float,
    max_attempts: int = GROUND_PLANE_ATTEMPTS,
) -> tuple[Float64[ndarray, "3"], Float64[ndarray, "3"]]:
    """Pick the floor: the largest RANSAC plane whose normal agrees with the gravity prior.

    Planes tilted more than ``max_tilt_deg`` from ``up_prior`` (walls, shelves)
    have their inliers removed and the fit repeats on the remainder. When no
    plane qualifies, the prior itself is the up direction and the ground passes
    through the lowest camera center.

    Args:
        cloud: Float64 metric world points with shape ``(n, 3)``.
        camera_centers: Float64 camera centers with shape ``(v, 3)``.
        up_prior: Float64 gravity-up direction with shape ``(3,)``.
        distance_threshold: RANSAC inlier distance in meters.
        max_tilt_deg: Maximum accepted angle between the plane normal and the prior.
        max_attempts: Number of successive plane fits before falling back.

    Returns:
        Float64 up direction with shape ``(3,)`` and a point on the ground with shape ``(3,)``.
    """
    up_prior = up_prior / np.linalg.norm(up_prior)
    remaining_points: Float64[ndarray, "m 3"] = cloud
    min_cosine: float = float(np.cos(np.radians(max_tilt_deg)))
    for _attempt in range(max_attempts):
        if remaining_points.shape[0] < 3:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining_points)
        plane, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=2000)
        normal: Float64[ndarray, "3"] = np.array(plane[:3], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        if float(np.dot(normal, up_prior)) < 0.0:
            normal = -normal
        ground_point: Float64[ndarray, "3"] = remaining_points[inliers].mean(axis=0)
        if float(np.dot(normal, up_prior)) >= min_cosine:
            return normal, ground_point
        keep_mask: Bool[ndarray, " n"] = np.ones(remaining_points.shape[0], dtype=bool)
        keep_mask[np.asarray(inliers, dtype=np.int64)] = False
        remaining_points = remaining_points[keep_mask]
    lowest_camera: Float64[ndarray, "3"] = camera_centers[np.argmin(camera_centers @ up_prior)]
    return up_prior, lowest_camera
