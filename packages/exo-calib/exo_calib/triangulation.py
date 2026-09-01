"""Confidence-weighted multi-view triangulation."""

from typing import TypeAlias

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

from exo_calib.correspondences import ObservationSet

RobustTriangulationResult: TypeAlias = tuple[Float64[ndarray, "n_points 3"], ObservationSet]


def _reprojection_errors_px(
    obs: ObservationSet,
    points_xyz_n3: Float64[ndarray, "n_points 3"],
    k_v33: Float64[ndarray, "v 3 3"],
    cam_T_world_v44: Float64[ndarray, "v 4 4"],
) -> Float64[ndarray, "n_obs"]:
    """Compute pixel residuals, with infinity for invalid observations.

    Args:
        obs: Sparse pixel observations for the points.
        points_xyz_n3: Float64 world points with shape ``(n_points, 3)``.
        k_v33: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world_v44: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.

    Returns:
        Float64 reprojection errors with shape ``(n_obs,)``.
    """
    n_points: int = points_xyz_n3.shape[0]
    n_views: int = k_v33.shape[0]
    n_obs: int = obs.obs_point_idx.size
    reproj_error_m: Float64[ndarray, "n_obs"] = np.full(n_obs, np.inf, dtype=np.float64)
    valid_reference_m: Bool[ndarray, "n_obs"] = (
        (obs.obs_point_idx >= 0)
        & (obs.obs_point_idx < n_points)
        & (obs.obs_view_idx >= 0)
        & (obs.obs_view_idx < n_views)
        & np.isfinite(obs.obs_xy).all(axis=1)
        & np.isfinite(obs.obs_conf)
        & (obs.obs_conf > 0.0)
    )
    valid_reference_idx_q: Int64[ndarray, "q"] = np.flatnonzero(valid_reference_m).astype(np.int64)
    if valid_reference_idx_q.size == 0:
        return reproj_error_m

    point_idx_q: Int64[ndarray, "q"] = obs.obs_point_idx[valid_reference_idx_q]
    view_idx_q: Int64[ndarray, "q"] = obs.obs_view_idx[valid_reference_idx_q]
    point_homo_q4: Float64[ndarray, "q 4"] = np.column_stack(
        (points_xyz_n3[point_idx_q], np.ones(valid_reference_idx_q.size, dtype=np.float64))
    )
    points_cam_q3: Float64[ndarray, "q 3"] = np.einsum("qij,qj->qi", cam_T_world_v44[view_idx_q, :3, :], point_homo_q4)
    projected_homo_q3: Float64[ndarray, "q 3"] = np.einsum("qij,qj->qi", k_v33[view_idx_q], points_cam_q3)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_xy_q2: Float64[ndarray, "q 2"] = projected_homo_q3[:, :2] / projected_homo_q3[:, 2:3]
    projectable_q: Bool[ndarray, "q"] = (points_cam_q3[:, 2] > 0.0) & np.isfinite(projected_xy_q2).all(axis=1)
    valid_error_idx_r: Int64[ndarray, "r"] = valid_reference_idx_q[projectable_q]
    reproj_error_m[valid_error_idx_r] = np.linalg.norm(projected_xy_q2[projectable_q] - obs.obs_xy[valid_error_idx_r], axis=1)
    return reproj_error_m


def triangulate_points(
    obs: ObservationSet, k_v33: Float64[ndarray, "v 3 3"], cam_T_world_v44: Float64[ndarray, "v 4 4"]
) -> Float64[ndarray, "n_points 3"]:
    """Triangulate points with confidence-weighted DLT.

    Args:
        obs: Sparse pixel observations for the points.
        k_v33: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world_v44: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.

    Returns:
        Float64 world points with shape ``(n_points, 3)``. Invalid points are NaN.
    """
    n_points: int = obs.point_frame_idx.size
    n_views: int = k_v33.shape[0]
    projection_v34: Float64[ndarray, "v 3 4"] = np.einsum("vij,vjk->vik", k_v33, cam_T_world_v44[:, :3, :])
    points_xyz_n3: Float64[ndarray, "n_points 3"] = np.full((n_points, 3), np.nan, dtype=np.float64)
    point_idx: int
    for point_idx in range(n_points):
        point_obs_m: Bool[ndarray, "n_obs"] = obs.obs_point_idx == point_idx
        valid_obs_m: Bool[ndarray, "n_obs"] = (
            point_obs_m
            & np.isfinite(obs.obs_xy).all(axis=1)
            & np.isfinite(obs.obs_conf)
            & (obs.obs_conf > 0.0)
            & (obs.obs_view_idx >= 0)
            & (obs.obs_view_idx < n_views)
        )
        valid_obs_idx_q: Int64[ndarray, "q"] = np.flatnonzero(valid_obs_m).astype(np.int64)
        if valid_obs_idx_q.size < 2:
            continue

        dlt_rows_4: list[Float64[ndarray, "4"]] = []
        obs_idx: np.int64
        for obs_idx in valid_obs_idx_q:
            view_idx: int = int(obs.obs_view_idx[obs_idx])
            u_px: float = float(obs.obs_xy[obs_idx, 0])
            v_px: float = float(obs.obs_xy[obs_idx, 1])
            confidence: float = float(obs.obs_conf[obs_idx])
            projection_34: Float64[ndarray, "3 4"] = projection_v34[view_idx]
            dlt_rows_4.append(confidence * (u_px * projection_34[2] - projection_34[0]))
            dlt_rows_4.append(confidence * (v_px * projection_34[2] - projection_34[1]))

        dlt_matrix_r4: Float64[ndarray, "rows 4"] = np.stack(dlt_rows_4)
        svd_result: tuple[Float64[ndarray, "rows 4"], Float64[ndarray, "4"], Float64[ndarray, "4 4"]] = np.linalg.svd(
            dlt_matrix_r4, full_matrices=False
        )
        homogeneous_4: Float64[ndarray, "4"] = svd_result[2][-1]
        if np.isfinite(homogeneous_4).all() and abs(float(homogeneous_4[3])) > np.finfo(np.float64).eps:
            points_xyz_n3[point_idx] = homogeneous_4[:3] / homogeneous_4[3]
    return points_xyz_n3


def triangulate_robust(
    obs: ObservationSet,
    k_v33: Float64[ndarray, "v 3 3"],
    cam_T_world_v44: Float64[ndarray, "v 4 4"],
    *,
    reproj_threshold_px: float = 10.0,
    max_rounds: int = 2,
) -> RobustTriangulationResult:
    """Triangulate while iteratively pruning large reprojection residuals.

    Args:
        obs: Sparse pixel observations for the points.
        k_v33: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world_v44: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.
        reproj_threshold_px: Maximum retained reprojection error in pixels.
        max_rounds: Maximum pruning rounds.

    Returns:
        Float64 refined world points with shape ``(n_points, 3)`` and the pruned observations.
    """
    n_points: int = obs.point_frame_idx.size
    current_obs: ObservationSet = obs
    _round_idx: int
    for _round_idx in range(max_rounds):
        points_xyz_n3: Float64[ndarray, "n_points 3"] = triangulate_points(current_obs, k_v33, cam_T_world_v44)
        reproj_error_m: Float64[ndarray, "n_obs"] = _reprojection_errors_px(current_obs, points_xyz_n3, k_v33, cam_T_world_v44)
        keep_obs_m: Bool[ndarray, "n_obs"] = np.isfinite(reproj_error_m)
        point_idx: int
        for point_idx in range(n_points):
            point_obs_idx_q: Int64[ndarray, "q"] = np.flatnonzero(keep_obs_m & (current_obs.obs_point_idx == point_idx)).astype(np.int64)
            if point_obs_idx_q.size < 2:
                keep_obs_m[point_obs_idx_q] = False
                continue
            if np.all(reproj_error_m[point_obs_idx_q] <= reproj_threshold_px):
                continue
            if point_obs_idx_q.size == 2:
                keep_obs_m[point_obs_idx_q] = False
                continue

            best_candidate_idx: int = -1
            best_max_error_px: float = np.inf
            candidate_position: int
            for candidate_position in range(point_obs_idx_q.size):
                trial_obs_idx_r: Int64[ndarray, "r"] = np.delete(point_obs_idx_q, candidate_position)
                trial_obs: ObservationSet = ObservationSet(
                    point_frame_idx=current_obs.point_frame_idx[point_idx : point_idx + 1],
                    point_joint_idx=current_obs.point_joint_idx[point_idx : point_idx + 1],
                    obs_point_idx=np.zeros(trial_obs_idx_r.size, dtype=np.int64),
                    obs_view_idx=current_obs.obs_view_idx[trial_obs_idx_r],
                    obs_xy=current_obs.obs_xy[trial_obs_idx_r],
                    obs_conf=current_obs.obs_conf[trial_obs_idx_r],
                )
                trial_point_xyz_13: Float64[ndarray, "1 3"] = triangulate_points(trial_obs, k_v33, cam_T_world_v44)
                trial_error_r: Float64[ndarray, "r"] = _reprojection_errors_px(trial_obs, trial_point_xyz_13, k_v33, cam_T_world_v44)
                trial_max_error_px: float = float(np.max(trial_error_r))
                if trial_max_error_px < best_max_error_px:
                    best_max_error_px = trial_max_error_px
                    best_candidate_idx = int(point_obs_idx_q[candidate_position])
            keep_obs_m[best_candidate_idx] = False

        kept_per_point_n: Int64[ndarray, "n_points"] = np.bincount(
            current_obs.obs_point_idx[keep_obs_m], minlength=n_points
        ).astype(np.int64)
        keep_obs_m &= kept_per_point_n[current_obs.obs_point_idx] >= 2
        pruned_obs: ObservationSet = ObservationSet(
            point_frame_idx=current_obs.point_frame_idx,
            point_joint_idx=current_obs.point_joint_idx,
            obs_point_idx=current_obs.obs_point_idx[keep_obs_m],
            obs_view_idx=current_obs.obs_view_idx[keep_obs_m],
            obs_xy=current_obs.obs_xy[keep_obs_m],
            obs_conf=current_obs.obs_conf[keep_obs_m],
        )
        if pruned_obs.obs_point_idx.size == current_obs.obs_point_idx.size:
            current_obs = pruned_obs
            break
        current_obs = pruned_obs

    refined_xyz_n3: Float64[ndarray, "n_points 3"] = triangulate_points(current_obs, k_v33, cam_T_world_v44)
    return refined_xyz_n3, current_obs
