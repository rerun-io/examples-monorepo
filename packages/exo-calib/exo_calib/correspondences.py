"""Selection and storage of multi-view keypoint observations."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray


@dataclass(slots=True, frozen=True)
class ObservationSet:
    """Sparse observations associated with frame-joint 3D points."""

    point_frame_idx: Int64[ndarray, "n_points"]
    """Frame index of each 3D point."""
    point_joint_idx: Int64[ndarray, "n_points"]
    """Joint index of each 3D point."""
    obs_point_idx: Int64[ndarray, "n_obs"]
    """3D point row referenced by each observation."""
    obs_view_idx: Int64[ndarray, "n_obs"]
    """Camera index of each observation."""
    obs_xy: Float64[ndarray, "n_obs 2"]
    """Observed pixel coordinates."""
    obs_conf: Float64[ndarray, "n_obs"]
    """Modulated detector confidence of each observation."""


def build_observations(
    kp_xy_vtj2: Float64[ndarray, "v t j 2"],
    conf_vtj: Float64[ndarray, "v t j"],
    *,
    frame_stride: int = 1,
    min_pair_conf: float = 0.75,
    min_views: int = 2,
    max_points: int | None = None,
    seed: int = 0,
) -> ObservationSet:
    """Build sparse observations using Kineo's cross-view confidence rule.

    Args:
        kp_xy_vtj2: Float64 keypoint pixels with shape ``(v, t, j, 2)``.
        conf_vtj: Float64 detector confidences with shape ``(v, t, j)``.
        frame_stride: Keep frames whose index is divisible by this stride.
        min_pair_conf: Strict lower bound for a pair's geometric-mean confidence.
        min_views: Minimum number of surviving views per point.
        max_points: Optional uniform random point budget.
        seed: Seed for deterministic point subsampling.

    Returns:
        Sparse point and observation arrays.
    """
    n_frames: int = kp_xy_vtj2.shape[1]
    n_joints: int = kp_xy_vtj2.shape[2]
    point_frame_indices: list[int] = []
    point_joint_indices: list[int] = []
    obs_point_indices: list[int] = []
    obs_view_indices: list[int] = []
    obs_xy_rows_2: list[Float64[ndarray, "2"]] = []
    obs_conf_values: list[float] = []

    frame_idx: int
    joint_idx: int
    for frame_idx in range(0, n_frames, frame_stride):
        for joint_idx in range(n_joints):
            conf_v: Float64[ndarray, "v"] = conf_vtj[:, frame_idx, joint_idx]
            xy_v2: Float64[ndarray, "v 2"] = kp_xy_vtj2[:, frame_idx, joint_idx]
            candidate_v: Bool[ndarray, "v"] = (conf_v > 0.0) & np.isfinite(xy_v2).all(axis=1)
            candidate_conf_v: Float64[ndarray, "v"] = np.where(candidate_v, conf_v, 0.0)
            pair_strength_vv: Float64[ndarray, "v v"] = np.sqrt(candidate_conf_v[:, None] * candidate_conf_v[None, :])
            pair_survives_vv: Bool[ndarray, "v v"] = (pair_strength_vv > min_pair_conf) & candidate_v[:, None] & candidate_v[None, :]
            np.fill_diagonal(pair_survives_vv, False)
            surviving_v: Bool[ndarray, "v"] = np.any(pair_survives_vv, axis=1)
            surviving_view_idx_s: Int64[ndarray, "s"] = np.flatnonzero(surviving_v).astype(np.int64)
            if surviving_view_idx_s.size < min_views:
                continue

            point_idx: int = len(point_frame_indices)
            point_frame_indices.append(frame_idx)
            point_joint_indices.append(joint_idx)
            view_idx: np.int64
            for view_idx in surviving_view_idx_s:
                obs_point_indices.append(point_idx)
                obs_view_indices.append(int(view_idx))
                obs_xy_rows_2.append(xy_v2[view_idx])
                obs_conf_values.append(float(conf_v[view_idx]))

    point_frame_idx_all_n: Int64[ndarray, "n_all"] = np.asarray(point_frame_indices, dtype=np.int64)
    point_joint_idx_all_n: Int64[ndarray, "n_all"] = np.asarray(point_joint_indices, dtype=np.int64)
    obs_point_idx_all_m: Int64[ndarray, "m_all"] = np.asarray(obs_point_indices, dtype=np.int64)
    obs_view_idx_all_m: Int64[ndarray, "m_all"] = np.asarray(obs_view_indices, dtype=np.int64)
    obs_xy_all_m2: Float64[ndarray, "m_all 2"] = np.asarray(obs_xy_rows_2, dtype=np.float64).reshape(-1, 2)
    obs_conf_all_m: Float64[ndarray, "m_all"] = np.asarray(obs_conf_values, dtype=np.float64)

    n_all_points: int = point_frame_idx_all_n.size
    if max_points is not None and n_all_points > max_points:
        rng: np.random.Generator = np.random.default_rng(seed)
        selected_old_idx_n: Int64[ndarray, "n_points"] = np.sort(rng.choice(n_all_points, size=max_points, replace=False)).astype(np.int64)
    else:
        selected_old_idx_n = np.arange(n_all_points, dtype=np.int64)

    old_to_new_n: Int64[ndarray, "n_all"] = np.full(n_all_points, -1, dtype=np.int64)
    old_to_new_n[selected_old_idx_n] = np.arange(selected_old_idx_n.size, dtype=np.int64)
    keep_obs_m: Bool[ndarray, "m_all"] = old_to_new_n[obs_point_idx_all_m] >= 0
    return ObservationSet(
        point_frame_idx=point_frame_idx_all_n[selected_old_idx_n],
        point_joint_idx=point_joint_idx_all_n[selected_old_idx_n],
        obs_point_idx=old_to_new_n[obs_point_idx_all_m[keep_obs_m]],
        obs_view_idx=obs_view_idx_all_m[keep_obs_m],
        obs_xy=obs_xy_all_m2[keep_obs_m],
        obs_conf=obs_conf_all_m[keep_obs_m],
    )
