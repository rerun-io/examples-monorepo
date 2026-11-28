"""How far two bundle-adjustment solutions of the same start lie apart: the numbers of the CASPAR vs Mojo report.

Both reconstructions come from one model solved by two solvers with the same gauge, so they share a frame and are
compared directly, without an alignment. A bundle adjustment keeps its factor topology, so the problem read once from
the start (`colsfm.ba_mojo.problem_from_reconstruction`) names the observations and points of both solutions; each is
evaluated at the state its reconstruction holds after write-back. The cost is CASPAR's score, half the sum of squared
residuals over those observations, projected by COLMAP's own pose and camera model (CASPAR's 1e-6 projection epsilons
are below double rounding at these depths).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pycolmap
from caspar_mojo.problem import BaProblem
from jaxtyping import Float64, Int64
from numpy import ndarray

from colsfm.ba_mojo import BridgedProblem


@dataclass(frozen=True, slots=True)
class SolveComparison:
    """Solution B measured against solution A."""

    reprojection_a_px: float
    """COLMAP's mean reprojection error (`update_point_3d_errors` first)."""
    reprojection_b_px: float
    reprojection_relative_difference: float | None
    """(B - A) / A; 0.0 when both are zero, None when only A is (`relative_difference`)."""
    cost_a: float
    """Half the sum of squared residuals of the problem."""
    cost_b: float
    cost_relative_difference: float | None
    """(B - A) / A, as `reprojection_relative_difference`."""
    rotation_difference_max_deg: float
    """Angle of `cam_from_world` B relative to A, over the registered images."""
    rotation_difference_median_deg: float
    center_difference_max: float
    """Camera-centre distance over the scene extent (diagonal of the camera centres' bounding box in A)."""
    center_difference_median: float
    point_difference_max: float
    """Point distance over the scene extent, over the problem's points."""
    point_difference_median: float
    registered_images_a: int
    registered_images_b: int


def relative_difference(a: float, b: float) -> float | None:
    """(B - A) / A, defined for an exact reference too.

    Args:
        a: The reference value.
        b: The other value.

    Returns:
        The relative difference; 0.0 when both are zero, None when only A is (no relative scale exists).
    """
    if a == 0.0:
        return 0.0 if b == 0.0 else None
    return (b - a) / a


def _points(reconstruction: pycolmap.Reconstruction, point3D_ids: Int64[ndarray, " p"]) -> Float64[ndarray, "p 3"]:
    return np.array([reconstruction.point3D(int(point3D_id)).xyz for point3D_id in point3D_ids], np.float64).reshape(-1, 3)


def _cost(reconstruction: pycolmap.Reconstruction, bridged: BridgedProblem, points: Float64[ndarray, "p 3"]) -> float:
    problem: BaProblem = bridged.problem
    image_ids: Int64[ndarray, " n"] = bridged.ids.observation_image_ids
    # The bridge orders observations camera-first, then by image, so each image's observations are one slice.
    edges: Int64[ndarray, " k"] = np.concatenate([[0], np.flatnonzero(np.diff(image_ids)) + 1, [len(image_ids)]])
    total: float = 0.0
    for start, end in zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True):
        image: pycolmap.Image = reconstruction.image(int(image_ids[start]))
        pose: pycolmap.Rigid3d = image.cam_from_world()
        in_cam: Float64[ndarray, "m 3"] = (
            points[problem.observation_point[start:end]] @ np.asarray(pose.rotation.matrix()).T + np.asarray(pose.translation)
        )
        residuals: Float64[ndarray, "m 2"] = np.asarray(reconstruction.camera(image.camera_id).img_from_cam(in_cam)) - problem.pixels[start:end]
        total += 0.5 * float(np.sum(residuals**2))
    return total


def problem_cost(reconstruction: pycolmap.Reconstruction, bridged: BridgedProblem) -> float:
    """Half the sum of squared residuals of a problem's observations, at the state the reconstruction holds.

    Args:
        reconstruction: A solution of the problem: it holds every point, image and camera the problem names.
        bridged: The problem and its ids, from `colsfm.ba_mojo.problem_from_reconstruction`.

    Returns:
        The cost.
    """
    return _cost(reconstruction, bridged, _points(reconstruction, bridged.ids.point3D_ids))


def _mean_reprojection(reconstruction: pycolmap.Reconstruction) -> float:
    reconstruction.update_point_3d_errors()
    return reconstruction.compute_mean_reprojection_error()


def compare_solutions(a: pycolmap.Reconstruction, b: pycolmap.Reconstruction, bridged: BridgedProblem) -> SolveComparison:
    """Measure solution B against solution A of the same bundle adjustment.

    Args:
        a: The reference solution (CASPAR's).
        b: The other solution.
        bridged: The problem both solved, read from their common start.

    Returns:
        The comparison.

    Raises:
        ValueError: When the two share no registered image.
    """
    points_a: Float64[ndarray, "p 3"] = _points(a, bridged.ids.point3D_ids)
    points_b: Float64[ndarray, "p 3"] = _points(b, bridged.ids.point3D_ids)
    cost_a: float = _cost(a, bridged, points_a)
    cost_b: float = _cost(b, bridged, points_b)
    reprojection_a: float = _mean_reprojection(a)
    reprojection_b: float = _mean_reprojection(b)

    image_ids: list[int] = sorted(set(a.reg_image_ids()) & set(b.reg_image_ids()))
    if not image_ids:
        raise ValueError("the two solutions share no registered image")
    poses_a: list[pycolmap.Rigid3d] = [a.image(i).cam_from_world() for i in image_ids]
    poses_b: list[pycolmap.Rigid3d] = [b.image(i).cam_from_world() for i in image_ids]
    angles: Float64[ndarray, " m"] = np.array([np.rad2deg(pa.rotation.angle_to(pb.rotation)) for pa, pb in zip(poses_a, poses_b, strict=True)])
    centers_a: Float64[ndarray, "m 3"] = np.array([a.image(i).projection_center() for i in image_ids])
    centers_b: Float64[ndarray, "m 3"] = np.array([b.image(i).projection_center() for i in image_ids])
    extent: float = float(np.linalg.norm(np.ptp(centers_a, axis=0))) or 1.0
    center_differences: Float64[ndarray, " m"] = np.linalg.norm(centers_b - centers_a, axis=1) / extent
    point_differences: Float64[ndarray, " p"] = np.linalg.norm(points_b - points_a, axis=1) / extent if len(points_a) else np.zeros(1)
    return SolveComparison(
        reprojection_a_px=reprojection_a,
        reprojection_b_px=reprojection_b,
        reprojection_relative_difference=relative_difference(reprojection_a, reprojection_b),
        cost_a=cost_a,
        cost_b=cost_b,
        cost_relative_difference=relative_difference(cost_a, cost_b),
        rotation_difference_max_deg=float(angles.max()),
        rotation_difference_median_deg=float(np.median(angles)),
        center_difference_max=float(center_differences.max()),
        center_difference_median=float(np.median(center_differences)),
        point_difference_max=float(point_differences.max()),
        point_difference_median=float(np.median(point_differences)),
        registered_images_a=a.num_reg_images(),
        registered_images_b=b.num_reg_images(),
    )
