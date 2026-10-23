"""exo-calib's triangulation policy over the shared DLT kernel (``simplecv.ops.triangulate.triangulate_weighted``).

``triangulate_robust`` prunes observations whose reprojection residual exceeds a threshold, resolving ambiguous cases
with leave-one-out consensus (all trials of a round in one batched call); ``triangulate_frame_keypoints`` is the
online tracker's per-frame entry.
"""

from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from simplecv.ops.triangulate import triangulate_weighted

from exo_calib.ops.correspondences import ObservationSet

ROBUST_TRIANGULATION_ROUNDS: int = 2
"""Triangulate-then-prune passes; the second pass re-fits without the observations the first one rejected."""
MIN_JOINTS_TO_JUDGE_VIEW: int = 4
"""The tracker's view gate judges only views that keep at least this many joints."""


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
    """Triangulate all points at once with the shared weighted DLT kernel (rows scaled by confidence: weight = conf²).

    Every point holds at most one observation per view, so the sparse set scatters into dense ``(v, n)`` arrays.

    Returns:
        Float64 world points with shape ``(n_points, 3)``. Points with fewer than two valid observations are NaN.
    """
    n_views: int = intrinsics.shape[0]
    valid: Bool[ndarray, "n_obs"] = valid_observation_mask(obs, n_views)
    xy: Float64[ndarray, "v n 2"] = np.full((n_views, obs.point_frame_idx.size, 2), np.nan)
    weight: Float64[ndarray, "v n"] = np.zeros((n_views, obs.point_frame_idx.size))
    xy[obs.obs_view_idx[valid], obs.obs_point_idx[valid]] = obs.obs_xy[valid]
    weight[obs.obs_view_idx[valid], obs.obs_point_idx[valid]] = obs.obs_conf[valid] ** 2
    projection: Float64[ndarray, "v 3 4"] = np.einsum("vij,vjk->vik", intrinsics, cam_T_world[:, :3, :])
    points_xyz, _ = triangulate_weighted(torch.from_numpy(xy), torch.from_numpy(weight), torch.from_numpy(projection))
    return points_xyz.numpy()


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
    view_gate_px: float,
) -> tuple[Float64[ndarray, "j 3"], Float64[ndarray, "j"]]:
    """Triangulate one multi-view keypoint frame onto its full joint grid.

    A view whose kept joints reproject more than ``view_gate_px`` off on average is dropped, worst first while more than two
    views remain, and the frame triangulated again (mamma's ``triangulate_gated``): one view holding another person spreads its
    error over every view, under the per-observation gate, and would otherwise move the whole skeleton.

    Args:
        kp_xy: Float64 image keypoints with shape ``(v, j, 2)``.
        conf: Float64 keypoint confidences with shape ``(v, j)``.
        intrinsics: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.
        min_confidence: Inclusive per-view confidence threshold.
        reproj_threshold_px: Robust-triangulation reprojection threshold.
        view_gate_px: Mean reprojection error of a view (``MIN_JOINTS_TO_JUDGE_VIEW`` kept joints) past which the worst view is dropped.

    Returns:
        Float64 world keypoints with shape ``(j, 3)`` and median retained
        confidences with shape ``(j,)``. Untriangulated joint rows are NaN/zero.
    """
    num_views, num_joints = kp_xy.shape[:2]
    usable: Bool[ndarray, "v j"] = np.isfinite(kp_xy).all(axis=2) & np.isfinite(conf) & (conf >= min_confidence)
    while True:  # one point per joint row; the kernel leaves a row with fewer than two observations NaN
        obs_view_idx, obs_joint_idx = np.nonzero(usable)
        obs: ObservationSet = ObservationSet(
            point_frame_idx=np.zeros(num_joints, dtype=np.int64), point_joint_idx=np.arange(num_joints, dtype=np.int64), obs_point_idx=obs_joint_idx,
            obs_view_idx=obs_view_idx, obs_xy=kp_xy[obs_view_idx, obs_joint_idx], obs_conf=conf[obs_view_idx, obs_joint_idx],
        )  # fmt: skip
        robust: RobustTriangulationResult = triangulate_robust(obs, intrinsics, cam_T_world, reproj_threshold_px=reproj_threshold_px)
        # a joint behind its camera (inf) counts as far off: nan_to_num makes it the largest float, so that view goes first
        error_px: Float64[ndarray, "r"] = np.nan_to_num(reprojection_errors_px(robust.obs, robust.points_xyz, intrinsics, cam_T_world))
        counts: Int64[ndarray, "v"] = np.bincount(robust.obs.obs_view_idx, minlength=num_views)
        mean_px: Float64[ndarray, "v"] = np.bincount(robust.obs.obs_view_idx, weights=error_px, minlength=num_views) / np.maximum(counts, 1)
        mean_px[counts < MIN_JOINTS_TO_JUDGE_VIEW] = 0.0
        worst: int = int(np.argmax(mean_px))
        if mean_px[worst] <= view_gate_px or np.count_nonzero(counts) <= 2:
            break
        usable[worst] = False
    retained_conf: Float64[ndarray, "v j"] = np.full((num_views, num_joints), np.nan, dtype=np.float64)
    retained_conf[robust.obs.obs_view_idx, robust.obs.obs_point_idx] = robust.obs.obs_conf
    has_obs: Bool[ndarray, "j"] = np.isfinite(retained_conf).any(axis=0)
    points_conf: Float64[ndarray, "j"] = np.zeros(num_joints, dtype=np.float64)
    points_conf[has_obs] = np.nanmedian(retained_conf[:, has_obs], axis=0)
    points_conf[~np.isfinite(robust.points_xyz).all(axis=1)] = 0.0
    return robust.points_xyz, points_conf
