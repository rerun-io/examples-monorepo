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
    kp_xy: Float64[ndarray, "v t j 2"],
    conf: Float64[ndarray, "v t j"],
    *,
    min_pair_conf: float,
    min_views: int,
) -> ObservationSet:
    """Build sparse observations over every frame using Kineo's cross-view confidence rule.

    Args:
        kp_xy: Float64 keypoint pixels with shape ``(v, t, j, 2)``.
        conf: Float64 detector confidences with shape ``(v, t, j)``.
        min_pair_conf: Strict lower bound for a pair's geometric-mean confidence.
        min_views: Minimum number of surviving views per point.

    Returns:
        Sparse point and observation arrays.
    """
    n_views: int = kp_xy.shape[0]
    candidate_mask: Bool[ndarray, "v t j"] = (conf > 0.0) & np.isfinite(kp_xy).all(axis=-1)
    candidate_conf: Float64[ndarray, "v t j"] = np.where(candidate_mask, conf, 0.0)
    # Kineo pair rule over all view pairs at once: a view survives a cell when it
    # forms at least one pair whose geometric-mean confidence clears the bound.
    pair_survives: Bool[ndarray, "w v t j"] = (
        (np.sqrt(candidate_conf[None, :] * candidate_conf[:, None]) > min_pair_conf)
        & candidate_mask[None, :]
        & candidate_mask[:, None]
    )
    view_diag_idx: Int64[ndarray, "v"] = np.arange(n_views, dtype=np.int64)
    pair_survives[view_diag_idx, view_diag_idx] = False
    surviving: Bool[ndarray, "v t j"] = pair_survives.any(axis=0)
    point_mask: Bool[ndarray, "t j"] = surviving.sum(axis=0) >= min_views

    point_frame_raw, point_joint_raw = np.nonzero(point_mask)  # frame-major order
    point_frame_idx: Int64[ndarray, "n_points"] = point_frame_raw.astype(np.int64)
    point_row: Int64[ndarray, "t j"] = np.full(point_mask.shape, -1, dtype=np.int64)
    point_row[point_frame_raw, point_joint_raw] = np.arange(point_frame_idx.size, dtype=np.int64)

    obs_view_raw, obs_frame_raw, obs_joint_raw = np.nonzero(surviving & point_mask[None])
    obs_point_raw: Int64[ndarray, "n_obs"] = point_row[obs_frame_raw, obs_joint_raw]
    obs_order: Int64[ndarray, "n_obs"] = np.lexsort((obs_view_raw, obs_point_raw)).astype(np.int64)
    return ObservationSet(
        point_frame_idx=point_frame_idx,
        point_joint_idx=point_joint_raw.astype(np.int64),
        obs_point_idx=obs_point_raw[obs_order],
        obs_view_idx=obs_view_raw[obs_order].astype(np.int64),
        obs_xy=kp_xy[obs_view_raw, obs_frame_raw, obs_joint_raw][obs_order],
        obs_conf=conf[obs_view_raw, obs_frame_raw, obs_joint_raw][obs_order],
    )


def under_supported_view_indices(
    obs_view_idx: Int64[ndarray, "m"],
    n_views: int,
    minimum_ratio: float,
) -> tuple[int, ...]:
    """Find camera views with anomalously little robust correspondence support.

    Args:
        obs_view_idx: Camera index of every surviving observation.
        n_views: Number of cameras.
        minimum_ratio: Fraction of the median per-view support below which a view
            counts as under-supported.

    Returns:
        Indices of the under-supported views.
    """
    if minimum_ratio < 0.0:
        raise ValueError("minimum_ratio must be nonnegative")
    support: Int64[ndarray, "v"] = np.bincount(obs_view_idx, minlength=n_views).astype(np.int64)
    positive_support: Int64[ndarray, " p"] = support[support > 0]
    reference_support: float = float(np.median(positive_support)) if positive_support.size else 0.0
    minimum_support: int = max(1, int(np.ceil(reference_support * minimum_ratio)))
    return tuple(int(view_idx) for view_idx in np.flatnonzero(support < minimum_support))
