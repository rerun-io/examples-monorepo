"""The floor of a metric scene cloud: the largest RANSAC plane that agrees with a gravity prior."""

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

GROUND_PLANE_ATTEMPTS: int = 5
"""Largest RANSAC planes tried before falling back to the gravity prior and the lowest camera."""
RANSAC_ITERATIONS: int = 2000
"""Plane hypotheses (three random points each) per attempt."""
RANSAC_SEED: int = 0
"""The floor is the gauge of the whole rig, and Stage B projects its boxes through that rig: one fixed seed and numpy
without BLAS (einsum) make the floor a function of the cloud. Open3D's ``segment_plane`` is not one, seeded or not: its
OpenMP loop keeps the best candidate and stops early in thread-schedule order."""


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
    rng: np.random.Generator = np.random.default_rng(RANSAC_SEED)
    for _attempt in range(max_attempts):
        if remaining_points.shape[0] < 3:
            break
        # RANSAC: the plane through three random points with the most points closer than distance_threshold (the first
        # hypothesis on a tie), then the orthogonal least-squares plane of those inliers.
        triplets: Float64[ndarray, "k 3 3"] = remaining_points[rng.integers(0, remaining_points.shape[0], size=(RANSAC_ITERATIONS, 3))]
        normals: Float64[ndarray, "k 3"] = np.cross(triplets[:, 1] - triplets[:, 0], triplets[:, 2] - triplets[:, 0])
        lengths: Float64[ndarray, " k"] = np.linalg.norm(normals, axis=1)
        normals, triplets = normals[lengths > 0.0] / lengths[lengths > 0.0, None], triplets[lengths > 0.0]
        if normals.shape[0] == 0:  # every triplet repeats a point or is collinear: no plane left to fit
            break
        offsets: Float64[ndarray, " k"] = -(normals * triplets[:, 0]).sum(axis=1)
        counts: Int64[ndarray, " k"] = np.array(
            [
                np.count_nonzero(np.abs(np.einsum("mj,j->m", remaining_points, n) + d) < distance_threshold)
                for n, d in zip(normals, offsets, strict=True)
            ]
        )
        best: int = int(np.argmax(counts))
        inliers: Bool[ndarray, " m"] = np.abs(np.einsum("mj,j->m", remaining_points, normals[best]) + offsets[best]) < distance_threshold
        centered: Float64[ndarray, "i 3"] = remaining_points[inliers] - remaining_points[inliers].mean(axis=0)
        normal: Float64[ndarray, "3"] = np.linalg.eigh(np.einsum("ni,nj->ij", centered, centered))[1][:, 0]
        if float(np.dot(normal, up_prior)) < 0.0:
            normal = -normal
        ground_point: Float64[ndarray, "3"] = remaining_points[inliers].mean(axis=0)
        if float(np.dot(normal, up_prior)) >= min_cosine:
            return normal, ground_point
        remaining_points = remaining_points[~inliers]
    lowest_camera: Float64[ndarray, "3"] = camera_centers[np.argmin(camera_centers @ up_prior)]
    return up_prior, lowest_camera
