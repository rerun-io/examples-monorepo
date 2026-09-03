"""Confidence-weighted multi-view triangulation (fully vectorized).

``triangulate_points`` solves every point's weighted DLT at once: each valid
observation contributes two confidence-scaled rows, the rows scatter into
per-point 4x4 Gram matrices (``A^T A``) via bincount, and one batched ``eigh``
extracts the null-space vector per point. ``triangulate_robust`` prunes
observations whose reprojection residual exceeds a threshold, resolving
ambiguous cases with leave-one-out consensus — all trials of a round are
triangulated in a single batched call, grouped by view count.
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

from exo_calib.correspondences import ObservationSet

ROBUST_TRIANGULATION_ROUNDS: int = 2
"""Triangulate-then-prune passes; the second pass re-fits without the observations the first one rejected."""


class RobustTriangulationResult(NamedTuple):
    """Triangulated points and the observations that survived the reprojection gate."""

    points_xyz: Float64[ndarray, "n_points 3"]
    obs: ObservationSet


def reprojection_errors_px(
    obs: ObservationSet,
    points_xyz: Float64[ndarray, "n_points 3"],
    intrinsics: Float64[ndarray, "v 3 3"],
    cam_T_world: Float64[ndarray, "v 4 4"],
) -> Float64[ndarray, "n_obs"]:
    """Compute pixel residuals, with infinity for invalid observations.

    Args:
        obs: Sparse pixel observations for the points.
        points_xyz: Float64 world points with shape ``(n_points, 3)``.
        intrinsics: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.

    Returns:
        Float64 reprojection errors with shape ``(n_obs,)``.
    """
    n_views: int = intrinsics.shape[0]
    n_obs: int = obs.obs_point_idx.size
    reproj_error: Float64[ndarray, "n_obs"] = np.full(n_obs, np.inf, dtype=np.float64)
    valid_reference_mask: Bool[ndarray, "n_obs"] = valid_observation_mask(obs, n_views, n_points=points_xyz.shape[0])
    valid_reference_idx: Int64[ndarray, "q"] = np.flatnonzero(valid_reference_mask).astype(np.int64)
    if valid_reference_idx.size == 0:
        return reproj_error

    observation_point_idx: Int64[ndarray, "q"] = obs.obs_point_idx[valid_reference_idx]
    observation_view_idx: Int64[ndarray, "q"] = obs.obs_view_idx[valid_reference_idx]
    point_homo: Float64[ndarray, "q 4"] = np.column_stack(
        (points_xyz[observation_point_idx], np.ones(valid_reference_idx.size, dtype=np.float64))
    )
    points_cam: Float64[ndarray, "q 3"] = np.einsum("qij,qj->qi", cam_T_world[observation_view_idx, :3, :], point_homo)
    projected_homo: Float64[ndarray, "q 3"] = np.einsum("qij,qj->qi", intrinsics[observation_view_idx], points_cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_xy: Float64[ndarray, "q 2"] = projected_homo[:, :2] / projected_homo[:, 2:3]
    projectable: Bool[ndarray, "q"] = (points_cam[:, 2] > 0.0) & np.isfinite(projected_xy).all(axis=1)
    valid_error_idx: Int64[ndarray, "r"] = valid_reference_idx[projectable]
    reproj_error[valid_error_idx] = np.linalg.norm(projected_xy[projectable] - obs.obs_xy[valid_error_idx], axis=1)
    return reproj_error


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
    obs: ObservationSet, intrinsics: Float64[ndarray, "v 3 3"], cam_T_world: Float64[ndarray, "v 4 4"]
) -> Float64[ndarray, "n_points 3"]:
    """Triangulate all points at once with confidence-weighted DLT.

    Each observation contributes rows ``w * (u * P[2] - P[0])`` and
    ``w * (v * P[2] - P[1])``; the per-point normal matrix ``A^T A`` is
    accumulated with bincount scatters (zero rows contribute nothing), and the
    null-space vector is the eigenvector of the smallest eigenvalue from one
    batched ``eigh`` over all solvable points.

    Args:
        obs: Sparse pixel observations for the points.
        intrinsics: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.

    Returns:
        Float64 world points with shape ``(n_points, 3)``. Invalid points are NaN.
    """
    n_points: int = obs.point_frame_idx.size
    n_views: int = intrinsics.shape[0]
    points_xyz: Float64[ndarray, "n_points 3"] = np.full((n_points, 3), np.nan, dtype=np.float64)
    valid_mask: Bool[ndarray, "n_obs"] = valid_observation_mask(obs, n_views)
    if not valid_mask.any():
        return points_xyz

    observation_point_idx: Int64[ndarray, "q"] = obs.obs_point_idx[valid_mask]
    observation_view_idx: Int64[ndarray, "q"] = obs.obs_view_idx[valid_mask]
    xy: Float64[ndarray, "q 2"] = obs.obs_xy[valid_mask]
    observation_conf: Float64[ndarray, "q"] = obs.obs_conf[valid_mask]

    projection: Float64[ndarray, "v 3 4"] = np.einsum("vij,vjk->vik", intrinsics, cam_T_world[:, :3, :])
    observation_projection: Float64[ndarray, "q 3 4"] = projection[observation_view_idx]
    row_u: Float64[ndarray, "q 4"] = observation_conf[:, None] * (xy[:, 0:1] * observation_projection[:, 2] - observation_projection[:, 0])
    row_v: Float64[ndarray, "q 4"] = observation_conf[:, None] * (xy[:, 1:2] * observation_projection[:, 2] - observation_projection[:, 1])
    rows: Float64[ndarray, "r 4"] = np.concatenate((row_u, row_v))
    row_point_idx: Int64[ndarray, "r"] = np.concatenate((observation_point_idx, observation_point_idx))

    gram: Float64[ndarray, "n_points 4 4"] = np.zeros((n_points, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(i, 4):
            entry: Float64[ndarray, "n_points"] = np.bincount(
                row_point_idx, weights=rows[:, i] * rows[:, j], minlength=n_points
            )
            gram[:, i, j] = entry
            gram[:, j, i] = entry

    obs_count: Int64[ndarray, "n_points"] = np.bincount(observation_point_idx, minlength=n_points).astype(np.int64)
    solvable: Bool[ndarray, "n_points"] = (obs_count >= 2) & np.isfinite(gram).all(axis=(1, 2))
    if not solvable.any():
        return points_xyz

    eig_result: tuple[Float64[ndarray, "s 4"], Float64[ndarray, "s 4 4"]] = np.linalg.eigh(gram[solvable])
    homogeneous: Float64[ndarray, "s 4"] = eig_result[1][:, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        dehomogenized: Float64[ndarray, "s 3"] = homogeneous[:, :3] / homogeneous[:, 3:4]
    usable: Bool[ndarray, "s"] = (np.abs(homogeneous[:, 3]) > np.finfo(np.float64).eps) & np.isfinite(dehomogenized).all(axis=1)
    solvable_idx: Int64[ndarray, "s"] = np.flatnonzero(solvable).astype(np.int64)
    points_xyz[solvable_idx[usable]] = dehomogenized[usable]
    return points_xyz


def _leave_one_out_drops(
    obs: ObservationSet,
    keep_obs_mask: Bool[ndarray, "n_obs"],
    offending_point: Bool[ndarray, "n_points"],
    intrinsics: Float64[ndarray, "v 3 3"],
    cam_T_world: Float64[ndarray, "v 4 4"],
) -> Int64[ndarray, " n_drop"]:
    """For each offending point (>= 3 kept obs), pick the single observation whose
    removal minimizes the point's max reprojection error.

    All leave-one-out trials of all points with the same view count are batched
    into one synthetic ``ObservationSet`` and triangulated in a single call.

    Returns:
        Observation indices to drop, one per offending point.
    """
    kept_idx: Int64[ndarray, "k"] = np.flatnonzero(keep_obs_mask & offending_point[obs.obs_point_idx]).astype(np.int64)
    order: Int64[ndarray, "k"] = np.argsort(obs.obs_point_idx[kept_idx], kind="stable").astype(np.int64)
    kept_idx = kept_idx[order]
    kept_point: Int64[ndarray, "k"] = obs.obs_point_idx[kept_idx]
    group_starts: Int64[ndarray, "g"] = np.flatnonzero(np.r_[True, kept_point[1:] != kept_point[:-1]]).astype(np.int64)
    group_sizes: Int64[ndarray, "g"] = np.diff(np.r_[group_starts, kept_point.size]).astype(np.int64)

    drop_indices: list[Int64[ndarray, " d"]] = []
    for group_size in np.unique(group_sizes):
        size: int = int(group_size)
        starts: Int64[ndarray, "h"] = group_starts[group_sizes == group_size]
        obs_idx: Int64[ndarray, "h q"] = kept_idx[starts[:, None] + np.arange(size)[None, :]]
        # (h, q, q-1): trial s of group keeps every position except s.
        off_diag: Int64[ndarray, "q q1"] = np.stack([np.delete(np.arange(size), s) for s in range(size)]).astype(np.int64)
        trial_obs_idx: Int64[ndarray, "f"] = obs_idx[:, off_diag].reshape(-1)
        n_trials: int = starts.size * size
        trial_id: Int64[ndarray, "f"] = np.repeat(np.arange(n_trials, dtype=np.int64), size - 1)
        trial_set: ObservationSet = ObservationSet(
            point_frame_idx=np.zeros(n_trials, dtype=np.int64),
            point_joint_idx=np.zeros(n_trials, dtype=np.int64),
            obs_point_idx=trial_id,
            obs_view_idx=obs.obs_view_idx[trial_obs_idx],
            obs_xy=obs.obs_xy[trial_obs_idx],
            obs_conf=obs.obs_conf[trial_obs_idx],
        )
        trial_points: Float64[ndarray, "t 3"] = triangulate_points(trial_set, intrinsics, cam_T_world)
        trial_errors: Float64[ndarray, "f"] = reprojection_errors_px(trial_set, trial_points, intrinsics, cam_T_world)
        max_error: Float64[ndarray, "h q"] = trial_errors.reshape(starts.size, size, size - 1).max(axis=2)
        best_position: Int64[ndarray, "h"] = np.argmin(max_error, axis=1).astype(np.int64)
        drop_indices.append(obs_idx[np.arange(starts.size), best_position])
    return np.concatenate(drop_indices) if drop_indices else np.empty(0, dtype=np.int64)


def triangulate_robust(
    obs: ObservationSet,
    intrinsics: Float64[ndarray, "v 3 3"],
    cam_T_world: Float64[ndarray, "v 4 4"],
    *,
    reproj_threshold_px: float,
) -> RobustTriangulationResult:
    """Triangulate while iteratively pruning large reprojection residuals.

    Args:
        obs: Sparse pixel observations for the points.
        intrinsics: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.
        reproj_threshold_px: Maximum retained reprojection error in pixels.
        ROBUST_TRIANGULATION_ROUNDS: Maximum pruning rounds.

    Returns:
        Float64 refined world points with shape ``(n_points, 3)`` and the pruned observations.
    """
    n_points: int = obs.point_frame_idx.size
    current_obs: ObservationSet = obs
    points_xyz: Float64[ndarray, "n_points 3"] = triangulate_points(current_obs, intrinsics, cam_T_world)
    for _round_idx in range(ROBUST_TRIANGULATION_ROUNDS):
        reproj_error: Float64[ndarray, "n_obs"] = reprojection_errors_px(current_obs, points_xyz, intrinsics, cam_T_world)
        keep_obs_mask: Bool[ndarray, "n_obs"] = np.isfinite(reproj_error)

        kept_count: Int64[ndarray, "n_points"] = np.bincount(current_obs.obs_point_idx[keep_obs_mask], minlength=n_points).astype(np.int64)
        over_threshold: Bool[ndarray, "n_obs"] = keep_obs_mask & (reproj_error > reproj_threshold_px)
        offending: Bool[ndarray, "n_points"] = (
            np.bincount(current_obs.obs_point_idx[over_threshold], minlength=n_points).astype(np.int64) > 0
        )

        # < 2 kept observations, or an offending 2-view point: drop everything.
        drop_all: Bool[ndarray, "n_points"] = (kept_count < 2) | (offending & (kept_count == 2))
        keep_obs_mask &= ~drop_all[current_obs.obs_point_idx]

        # Offending points with >= 3 views: leave-one-out consensus, drop one obs each.
        leave_one_out: Bool[ndarray, "n_points"] = offending & (kept_count >= 3)
        if leave_one_out.any():
            keep_obs_mask[_leave_one_out_drops(current_obs, keep_obs_mask, leave_one_out, intrinsics, cam_T_world)] = False

        kept_per_point: Int64[ndarray, "n_points"] = np.bincount(
            current_obs.obs_point_idx[keep_obs_mask], minlength=n_points
        ).astype(np.int64)
        keep_obs_mask &= kept_per_point[current_obs.obs_point_idx] >= 2
        pruned_obs: ObservationSet = ObservationSet(
            point_frame_idx=current_obs.point_frame_idx,
            point_joint_idx=current_obs.point_joint_idx,
            obs_point_idx=current_obs.obs_point_idx[keep_obs_mask],
            obs_view_idx=current_obs.obs_view_idx[keep_obs_mask],
            obs_xy=current_obs.obs_xy[keep_obs_mask],
            obs_conf=current_obs.obs_conf[keep_obs_mask],
        )
        if pruned_obs.obs_point_idx.size == current_obs.obs_point_idx.size:
            current_obs = pruned_obs
            break
        current_obs = pruned_obs
        points_xyz = triangulate_points(current_obs, intrinsics, cam_T_world)

    return RobustTriangulationResult(points_xyz=points_xyz, obs=current_obs)


def triangulate_frame_keypoints(
    kp_xy: Float64[ndarray, "v j 2"],
    conf: Float64[ndarray, "v j"],
    intrinsics: Float64[ndarray, "v 3 3"],
    cam_T_world: Float64[ndarray, "v 4 4"],
    *,
    min_confidence: float,
    reproj_threshold_px: float,
) -> tuple[Float64[ndarray, "j 3"], Float64[ndarray, "j"]]:
    """Triangulate one multi-view keypoint frame onto its full joint grid.

    Args:
        kp_xy: Float64 image keypoints with shape ``(v, j, 2)``.
        conf: Float64 keypoint confidences with shape ``(v, j)``.
        intrinsics: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.
        min_confidence: Inclusive per-view confidence threshold.
        reproj_threshold_px: Robust-triangulation reprojection threshold.

    Returns:
        Float64 world keypoints with shape ``(j, 3)`` and median retained
        confidences with shape ``(j,)``. Untriangulated joint rows are NaN/zero.
    """
    num_joints: int = kp_xy.shape[1]
    points_xyz: Float64[ndarray, "j 3"] = np.full((num_joints, 3), np.nan, dtype=np.float64)
    points_conf: Float64[ndarray, "j"] = np.zeros(num_joints, dtype=np.float64)
    usable: Bool[ndarray, "v j"] = (
        np.isfinite(kp_xy).all(axis=2) & np.isfinite(conf) & (conf >= min_confidence)
    )
    triangulatable: Bool[ndarray, "j"] = usable.sum(axis=0) >= 2
    point_joint_idx: Int64[ndarray, "n"] = np.flatnonzero(triangulatable).astype(np.int64)
    if point_joint_idx.size == 0:
        return points_xyz, points_conf

    joint_to_point: Int64[ndarray, "j"] = np.full(num_joints, -1, dtype=np.int64)
    joint_to_point[point_joint_idx] = np.arange(point_joint_idx.size, dtype=np.int64)
    obs_view_idx, obs_joint_idx = np.nonzero(usable & triangulatable[None])
    obs_view_idx = obs_view_idx.astype(np.int64)
    obs_joint_idx = obs_joint_idx.astype(np.int64)
    obs: ObservationSet = ObservationSet(
        point_frame_idx=np.zeros(point_joint_idx.size, dtype=np.int64),
        point_joint_idx=point_joint_idx,
        obs_point_idx=joint_to_point[obs_joint_idx],
        obs_view_idx=obs_view_idx,
        obs_xy=kp_xy[obs_view_idx, obs_joint_idx],
        obs_conf=conf[obs_view_idx, obs_joint_idx],
    )
    robust_result: RobustTriangulationResult = triangulate_robust(
        obs, intrinsics, cam_T_world, reproj_threshold_px=reproj_threshold_px
    )
    points_xyz[point_joint_idx] = robust_result.points_xyz
    retained_obs: ObservationSet = robust_result.obs
    retained_conf: Float64[ndarray, "v n"] = np.full((kp_xy.shape[0], point_joint_idx.size), np.nan, dtype=np.float64)
    retained_conf[retained_obs.obs_view_idx, retained_obs.obs_point_idx] = retained_obs.obs_conf
    has_obs: Bool[ndarray, "n"] = np.isfinite(retained_conf).any(axis=0)
    points_conf[point_joint_idx[has_obs]] = np.nanmedian(retained_conf[:, has_obs], axis=0)
    points_conf[~np.isfinite(points_xyz).all(axis=1)] = 0.0
    return points_xyz, points_conf
