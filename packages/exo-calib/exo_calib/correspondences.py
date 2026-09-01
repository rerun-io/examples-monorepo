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
    n_views: int = kp_xy_vtj2.shape[0]
    n_frames: int = kp_xy_vtj2.shape[1]
    selected_frames_f: Int64[ndarray, "f"] = np.arange(0, n_frames, frame_stride, dtype=np.int64)
    conf_vfj: Float64[ndarray, "v f j"] = conf_vtj[:, selected_frames_f]
    xy_vfj2: Float64[ndarray, "v f j 2"] = kp_xy_vtj2[:, selected_frames_f]

    candidate_vfj: Bool[ndarray, "v f j"] = (conf_vfj > 0.0) & np.isfinite(xy_vfj2).all(axis=-1)
    candidate_conf_vfj: Float64[ndarray, "v f j"] = np.where(candidate_vfj, conf_vfj, 0.0)
    # Kineo pair rule over all view pairs at once: a view survives a cell when it
    # forms at least one pair whose geometric-mean confidence clears the bound.
    pair_survives_wvfj: Bool[ndarray, "w v f j"] = (
        (np.sqrt(candidate_conf_vfj[None, :] * candidate_conf_vfj[:, None]) > min_pair_conf)
        & candidate_vfj[None, :]
        & candidate_vfj[:, None]
    )
    view_diag_idx_v: Int64[ndarray, "v"] = np.arange(n_views, dtype=np.int64)
    pair_survives_wvfj[view_diag_idx_v, view_diag_idx_v] = False
    surviving_vfj: Bool[ndarray, "v f j"] = pair_survives_wvfj.any(axis=0)
    is_point_fj: Bool[ndarray, "f j"] = surviving_vfj.sum(axis=0) >= min_views

    point_sel_frame_n, point_joint_idx_all_n = np.nonzero(is_point_fj)  # frame-major order
    point_frame_idx_all_n: Int64[ndarray, "n_all"] = selected_frames_f[point_sel_frame_n].astype(np.int64)
    point_joint_idx_all_n = point_joint_idx_all_n.astype(np.int64)
    point_row_fj: Int64[ndarray, "f j"] = np.full(is_point_fj.shape, -1, dtype=np.int64)
    point_row_fj[point_sel_frame_n, point_joint_idx_all_n] = np.arange(point_frame_idx_all_n.size, dtype=np.int64)

    obs_view_raw_m, obs_frame_raw_m, obs_joint_raw_m = np.nonzero(surviving_vfj & is_point_fj[None])
    obs_point_raw_m: Int64[ndarray, "m_all"] = point_row_fj[obs_frame_raw_m, obs_joint_raw_m]
    obs_order_m: Int64[ndarray, "m_all"] = np.lexsort((obs_view_raw_m, obs_point_raw_m)).astype(np.int64)
    obs_point_idx_all_m: Int64[ndarray, "m_all"] = obs_point_raw_m[obs_order_m]
    obs_view_idx_all_m: Int64[ndarray, "m_all"] = obs_view_raw_m[obs_order_m].astype(np.int64)
    obs_xy_all_m2: Float64[ndarray, "m_all 2"] = xy_vfj2[obs_view_raw_m, obs_frame_raw_m, obs_joint_raw_m][obs_order_m]
    obs_conf_all_m: Float64[ndarray, "m_all"] = conf_vfj[obs_view_raw_m, obs_frame_raw_m, obs_joint_raw_m][obs_order_m]

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
