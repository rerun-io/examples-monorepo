"""Confidence-weighted multi-view triangulation (fully vectorized).

``triangulate_points`` solves every point's weighted DLT at once: each valid
observation contributes two confidence-scaled rows, the rows scatter into
per-point 4x4 Gram matrices (``A^T A``) via bincount, and one batched ``eigh``
extracts the null-space vector per point. ``triangulate_robust`` prunes
observations whose reprojection residual exceeds a threshold, resolving
ambiguous cases with leave-one-out consensus — all trials of a round are
triangulated in a single batched call, grouped by view count.
"""

from typing import TypeAlias

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

from exo_calib.correspondences import ObservationSet

RobustTriangulationResult: TypeAlias = tuple[Float64[ndarray, "n_points 3"], ObservationSet]


def reprojection_errors_px(
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
    n_views: int = k_v33.shape[0]
    n_obs: int = obs.obs_point_idx.size
    reproj_error_m: Float64[ndarray, "n_obs"] = np.full(n_obs, np.inf, dtype=np.float64)
    valid_reference_m: Bool[ndarray, "n_obs"] = valid_observation_mask(obs, n_views, n_points=points_xyz_n3.shape[0])
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


def valid_observation_mask(obs: ObservationSet, n_views: int, n_points: int | None = None) -> Bool[ndarray, "n_obs"]:
    """Return the mask of usable observations; ``n_points`` adds point-index bounds checks."""
    mask: Bool[ndarray, "n_obs"] = (
        np.isfinite(obs.obs_xy).all(axis=1)
        & np.isfinite(obs.obs_conf)
        & (obs.obs_conf > 0.0)
        & (obs.obs_view_idx >= 0)
        & (obs.obs_view_idx < n_views)
    )
    if n_points is not None:
        mask &= (obs.obs_point_idx >= 0) & (obs.obs_point_idx < n_points)
    return mask


def triangulate_points(
    obs: ObservationSet, k_v33: Float64[ndarray, "v 3 3"], cam_T_world_v44: Float64[ndarray, "v 4 4"]
) -> Float64[ndarray, "n_points 3"]:
    """Triangulate all points at once with confidence-weighted DLT.

    Each observation contributes rows ``w * (u * P[2] - P[0])`` and
    ``w * (v * P[2] - P[1])``; the per-point normal matrix ``A^T A`` is
    accumulated with bincount scatters (zero rows contribute nothing), and the
    null-space vector is the eigenvector of the smallest eigenvalue from one
    batched ``eigh`` over all solvable points.

    Args:
        obs: Sparse pixel observations for the points.
        k_v33: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world_v44: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.

    Returns:
        Float64 world points with shape ``(n_points, 3)``. Invalid points are NaN.
    """
    n_points: int = obs.point_frame_idx.size
    n_views: int = k_v33.shape[0]
    points_xyz_n3: Float64[ndarray, "n_points 3"] = np.full((n_points, 3), np.nan, dtype=np.float64)
    valid_m: Bool[ndarray, "n_obs"] = valid_observation_mask(obs, n_views)
    if not valid_m.any():
        return points_xyz_n3

    point_idx_q: Int64[ndarray, "q"] = obs.obs_point_idx[valid_m]
    view_idx_q: Int64[ndarray, "q"] = obs.obs_view_idx[valid_m]
    xy_q2: Float64[ndarray, "q 2"] = obs.obs_xy[valid_m]
    conf_q: Float64[ndarray, "q"] = obs.obs_conf[valid_m]

    projection_v34: Float64[ndarray, "v 3 4"] = np.einsum("vij,vjk->vik", k_v33, cam_T_world_v44[:, :3, :])
    projection_q34: Float64[ndarray, "q 3 4"] = projection_v34[view_idx_q]
    row_u_q4: Float64[ndarray, "q 4"] = conf_q[:, None] * (xy_q2[:, 0:1] * projection_q34[:, 2] - projection_q34[:, 0])
    row_v_q4: Float64[ndarray, "q 4"] = conf_q[:, None] * (xy_q2[:, 1:2] * projection_q34[:, 2] - projection_q34[:, 1])
    rows_r4: Float64[ndarray, "r 4"] = np.concatenate((row_u_q4, row_v_q4))
    row_point_idx_r: Int64[ndarray, "r"] = np.concatenate((point_idx_q, point_idx_q))

    gram_n44: Float64[ndarray, "n_points 4 4"] = np.zeros((n_points, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(i, 4):
            entry_n: Float64[ndarray, "n_points"] = np.bincount(
                row_point_idx_r, weights=rows_r4[:, i] * rows_r4[:, j], minlength=n_points
            )
            gram_n44[:, i, j] = entry_n
            gram_n44[:, j, i] = entry_n

    obs_count_n: Int64[ndarray, "n_points"] = np.bincount(point_idx_q, minlength=n_points).astype(np.int64)
    solvable_n: Bool[ndarray, "n_points"] = (obs_count_n >= 2) & np.isfinite(gram_n44).all(axis=(1, 2))
    if not solvable_n.any():
        return points_xyz_n3

    eig_result: tuple[Float64[ndarray, "s 4"], Float64[ndarray, "s 4 4"]] = np.linalg.eigh(gram_n44[solvable_n])
    homogeneous_s4: Float64[ndarray, "s 4"] = eig_result[1][:, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        dehomogenized_s3: Float64[ndarray, "s 3"] = homogeneous_s4[:, :3] / homogeneous_s4[:, 3:4]
    usable_s: Bool[ndarray, "s"] = (np.abs(homogeneous_s4[:, 3]) > np.finfo(np.float64).eps) & np.isfinite(dehomogenized_s3).all(axis=1)
    solvable_idx_s: Int64[ndarray, "s"] = np.flatnonzero(solvable_n).astype(np.int64)
    points_xyz_n3[solvable_idx_s[usable_s]] = dehomogenized_s3[usable_s]
    return points_xyz_n3


def _leave_one_out_drops(
    obs: ObservationSet,
    keep_obs_m: Bool[ndarray, "n_obs"],
    offending_point_m: Bool[ndarray, "n_points"],
    k_v33: Float64[ndarray, "v 3 3"],
    cam_T_world_v44: Float64[ndarray, "v 4 4"],
) -> Int64[ndarray, " n_drop"]:
    """For each offending point (>= 3 kept obs), pick the single observation whose
    removal minimizes the point's max reprojection error.

    All leave-one-out trials of all points with the same view count are batched
    into one synthetic ``ObservationSet`` and triangulated in a single call.

    Returns:
        Observation indices to drop, one per offending point.
    """
    kept_idx_k: Int64[ndarray, "k"] = np.flatnonzero(keep_obs_m & offending_point_m[obs.obs_point_idx]).astype(np.int64)
    order_k: Int64[ndarray, "k"] = np.argsort(obs.obs_point_idx[kept_idx_k], kind="stable").astype(np.int64)
    kept_idx_k = kept_idx_k[order_k]
    kept_point_k: Int64[ndarray, "k"] = obs.obs_point_idx[kept_idx_k]
    group_starts_g: Int64[ndarray, "g"] = np.flatnonzero(np.r_[True, kept_point_k[1:] != kept_point_k[:-1]]).astype(np.int64)
    group_sizes_g: Int64[ndarray, "g"] = np.diff(np.r_[group_starts_g, kept_point_k.size]).astype(np.int64)

    drop_indices: list[Int64[ndarray, " d"]] = []
    for group_size in np.unique(group_sizes_g):
        size: int = int(group_size)
        starts_h: Int64[ndarray, "h"] = group_starts_g[group_sizes_g == group_size]
        obs_idx_hq: Int64[ndarray, "h q"] = kept_idx_k[starts_h[:, None] + np.arange(size)[None, :]]
        # (h, q, q-1): trial s of group keeps every position except s.
        off_diag_qq1: Int64[ndarray, "q q1"] = np.stack([np.delete(np.arange(size), s) for s in range(size)]).astype(np.int64)
        trial_obs_idx_f: Int64[ndarray, "f"] = obs_idx_hq[:, off_diag_qq1].reshape(-1)
        n_trials: int = starts_h.size * size
        trial_id_f: Int64[ndarray, "f"] = np.repeat(np.arange(n_trials, dtype=np.int64), size - 1)
        trial_set: ObservationSet = ObservationSet(
            point_frame_idx=np.zeros(n_trials, dtype=np.int64),
            point_joint_idx=np.zeros(n_trials, dtype=np.int64),
            obs_point_idx=trial_id_f,
            obs_view_idx=obs.obs_view_idx[trial_obs_idx_f],
            obs_xy=obs.obs_xy[trial_obs_idx_f],
            obs_conf=obs.obs_conf[trial_obs_idx_f],
        )
        trial_points_t3: Float64[ndarray, "t 3"] = triangulate_points(trial_set, k_v33, cam_T_world_v44)
        trial_errors_f: Float64[ndarray, "f"] = reprojection_errors_px(trial_set, trial_points_t3, k_v33, cam_T_world_v44)
        max_error_hq: Float64[ndarray, "h q"] = trial_errors_f.reshape(starts_h.size, size, size - 1).max(axis=2)
        best_position_h: Int64[ndarray, "h"] = np.argmin(max_error_hq, axis=1).astype(np.int64)
        drop_indices.append(obs_idx_hq[np.arange(starts_h.size), best_position_h])
    return np.concatenate(drop_indices) if drop_indices else np.empty(0, dtype=np.int64)


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
    points_xyz_n3: Float64[ndarray, "n_points 3"] = triangulate_points(current_obs, k_v33, cam_T_world_v44)
    for _round_idx in range(max_rounds):
        reproj_error_m: Float64[ndarray, "n_obs"] = reprojection_errors_px(current_obs, points_xyz_n3, k_v33, cam_T_world_v44)
        keep_obs_m: Bool[ndarray, "n_obs"] = np.isfinite(reproj_error_m)

        kept_count_n: Int64[ndarray, "n_points"] = np.bincount(current_obs.obs_point_idx[keep_obs_m], minlength=n_points).astype(np.int64)
        over_threshold_m: Bool[ndarray, "n_obs"] = keep_obs_m & (reproj_error_m > reproj_threshold_px)
        offending_n: Bool[ndarray, "n_points"] = (
            np.bincount(current_obs.obs_point_idx[over_threshold_m], minlength=n_points).astype(np.int64) > 0
        )

        # < 2 kept observations, or an offending 2-view point: drop everything.
        drop_all_n: Bool[ndarray, "n_points"] = (kept_count_n < 2) | (offending_n & (kept_count_n == 2))
        keep_obs_m &= ~drop_all_n[current_obs.obs_point_idx]

        # Offending points with >= 3 views: leave-one-out consensus, drop one obs each.
        loo_n: Bool[ndarray, "n_points"] = offending_n & (kept_count_n >= 3)
        if loo_n.any():
            keep_obs_m[_leave_one_out_drops(current_obs, keep_obs_m, loo_n, k_v33, cam_T_world_v44)] = False

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
        points_xyz_n3 = triangulate_points(current_obs, k_v33, cam_T_world_v44)

    return points_xyz_n3, current_obs
