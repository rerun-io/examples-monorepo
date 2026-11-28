"""How far two bundle-adjustment solutions of the same start lie apart: the numbers of the CASPAR vs Mojo report.

Both reconstructions come from one model solved by two solvers with the same gauge, so they share a frame and are
compared directly, without an alignment. The cost of each is CASPAR's score, half the sum of squared residuals over
the observations of the problem `colsfm.ba_mojo` reads, projected by COLMAP's own camera model (CASPAR's 1e-6
projection epsilons are below double rounding at these depths).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pycolmap
from caspar_mojo.problem import BaProblem
from jaxtyping import Float64, Int64
from numpy import ndarray

from colsfm.ba_mojo import BridgedProblem, problem_from_reconstruction


@dataclass(frozen=True, slots=True)
class SolveComparison:
    """Solution B measured against solution A."""

    reprojection_a_px: float
    """COLMAP's mean reprojection error (`update_point_3d_errors` first)."""
    reprojection_b_px: float
    reprojection_relative_difference: float
    """(B - A) / A."""
    cost_a: float
    """Half the sum of squared residuals of the problem."""
    cost_b: float
    cost_relative_difference: float
    """(B - A) / A."""
    rotation_difference_max_deg: float
    """Angle of `cam_from_world` B relative to A, over the registered images."""
    rotation_difference_median_deg: float
    center_difference_max: float
    """Camera-centre distance over the scene extent (diagonal of the camera centres' bounding box in A)."""
    center_difference_median: float
    point_difference_max: float
    """Point distance over the scene extent, over the points both share."""
    point_difference_median: float
    registered_images_a: int
    registered_images_b: int


def problem_cost(reconstruction: pycolmap.Reconstruction, bridged: BridgedProblem) -> float:
    """Half the sum of squared residuals of a problem's observations, at the state the problem holds.

    Args:
        reconstruction: Supplies each camera's model and size; the parameters come from the problem.
        bridged: The problem and its ids, from `colsfm.ba_mojo.problem_from_reconstruction`.

    Returns:
        The cost.
    """
    problem: BaProblem = bridged.problem
    frame, camera_row, point = problem.observation_frame, problem.observation_camera, problem.observation_point
    in_rig: Float64[ndarray, "n 3"] = _rotate(problem.rig_from_world[frame, :4], problem.points[point]) + problem.rig_from_world[frame, 4:]
    in_cam: Float64[ndarray, "n 3"] = _rotate(problem.sensor_from_rig[camera_row, :4], in_rig) + problem.sensor_from_rig[camera_row, 4:]
    total: float = 0.0
    for row, camera_id in enumerate(bridged.ids.camera_ids.tolist()):
        selected = camera_row == row
        stored: pycolmap.Camera = reconstruction.camera(camera_id)
        camera = pycolmap.Camera(model=stored.model, width=stored.width, height=stored.height, params=problem.intrinsics[row, : len(stored.params)])
        residuals: Float64[ndarray, "m 2"] = np.asarray(camera.img_from_cam(in_cam[selected])) - problem.pixels[selected]
        total += 0.5 * float(np.sum(residuals**2))
    return total


def _rotate(quat: Float64[ndarray, "n 4"], vectors: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
    unit: Float64[ndarray, "n 4"] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    axis: Float64[ndarray, "n 3"] = unit[:, :3]
    twice_cross: Float64[ndarray, "n 3"] = 2.0 * np.cross(axis, vectors)
    return vectors + unit[:, 3:] * twice_cross + np.cross(axis, twice_cross)


def _mean_reprojection(reconstruction: pycolmap.Reconstruction) -> float:
    reconstruction.update_point_3d_errors()
    return reconstruction.compute_mean_reprojection_error()


def compare_solutions(
    a: pycolmap.Reconstruction, b: pycolmap.Reconstruction, config: pycolmap.BundleAdjustmentConfig, ba_options: pycolmap.BundleAdjustmentOptions
) -> SolveComparison:
    """Measure solution B against solution A of the same bundle adjustment.

    Args:
        a: The reference solution (CASPAR's).
        b: The other solution.
        config: The solve's config (selects the problem whose cost is compared).
        ba_options: The solve's options.

    Returns:
        The comparison.
    """
    bridged_a: BridgedProblem = problem_from_reconstruction(a, config, ba_options)
    bridged_b: BridgedProblem = problem_from_reconstruction(b, config, ba_options)
    cost_a: float = problem_cost(a, bridged_a)
    cost_b: float = problem_cost(b, bridged_b)
    reprojection_a: float = _mean_reprojection(a)
    reprojection_b: float = _mean_reprojection(b)

    image_ids: list[int] = sorted(set(a.reg_image_ids()) & set(b.reg_image_ids()))
    poses_a: list[pycolmap.Rigid3d] = [a.image(i).cam_from_world() for i in image_ids]
    poses_b: list[pycolmap.Rigid3d] = [b.image(i).cam_from_world() for i in image_ids]
    angles: Float64[ndarray, " m"] = np.array([np.rad2deg(pa.rotation.angle_to(pb.rotation)) for pa, pb in zip(poses_a, poses_b, strict=True)])
    centers_a: Float64[ndarray, "m 3"] = np.array([a.image(i).projection_center() for i in image_ids])
    centers_b: Float64[ndarray, "m 3"] = np.array([b.image(i).projection_center() for i in image_ids])
    extent: float = float(np.linalg.norm(np.ptp(centers_a, axis=0))) or 1.0
    center_differences: Float64[ndarray, " m"] = np.linalg.norm(centers_b - centers_a, axis=1) / extent

    shared: Int64[ndarray, " p"] = np.intersect1d(bridged_a.ids.point3D_ids, bridged_b.ids.point3D_ids)
    rows_a = np.searchsorted(bridged_a.ids.point3D_ids, shared)
    rows_b = np.searchsorted(bridged_b.ids.point3D_ids, shared)
    point_differences: Float64[ndarray, " p"] = (
        np.linalg.norm(bridged_b.problem.points[rows_b] - bridged_a.problem.points[rows_a], axis=1) / extent if len(shared) else np.zeros(1)
    )
    return SolveComparison(
        reprojection_a_px=reprojection_a,
        reprojection_b_px=reprojection_b,
        reprojection_relative_difference=(reprojection_b - reprojection_a) / reprojection_a,
        cost_a=cost_a,
        cost_b=cost_b,
        cost_relative_difference=(cost_b - cost_a) / cost_a,
        rotation_difference_max_deg=float(angles.max()),
        rotation_difference_median_deg=float(np.median(angles)),
        center_difference_max=float(center_differences.max()),
        center_difference_median=float(np.median(center_differences)),
        point_difference_max=float(point_differences.max()),
        point_difference_median=float(np.median(point_differences)),
        registered_images_a=a.num_reg_images(),
        registered_images_b=b.num_reg_images(),
    )
