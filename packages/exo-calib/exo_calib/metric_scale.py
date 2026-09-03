"""Global metric rescale of a refined rig from MoGe-2 metric depth at the surviving 2D observations."""

import numpy as np
import torch
from jaxtyping import Bool, Float64, Int64, UInt8
from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
from numpy import ndarray
from rerun.catalog import DatasetEntry
from torch import Tensor

from exo_calib.correspondences import ObservationSet
from exo_calib.video_io import ExoVideoStreams, open_exo_streams

METRIC_RESCALE_FRAMES: int = 6
"""Frames per view sampled for the MoGe-2 metric rescale after BA."""
MIN_RESCALE_GROUPS: int = 8
"""Fewest (view, frame) depth-ratio groups before the rescale is trusted; below it the rig keeps its scale."""
MIN_RESCALE_SAMPLES_PER_GROUP: int = 3
"""Fewest valid depth ratios a (view, frame) group needs to contribute its median."""
MIN_DEPTH_M: float = 0.1
"""Depths at or below this (metric or triangulated) are not trusted as ratio samples."""


def grouped_medians(group: Int64[ndarray, " n"], values: Float64[ndarray, " n"], min_count: int) -> tuple[Float64[ndarray, " g"], int]:
    """Median of ``values`` within each group, for the groups that hold at least ``min_count`` samples.

    One sort by (group, value) replaces a mask-and-median pass per group; the
    medians equal :func:`numpy.median` over each group's values exactly.

    Args:
        group: Group id per sample.
        values: Sample per row of ``group``.
        min_count: Smallest group that contributes a median.

    Returns:
        The medians in ascending group-id order and the number of samples they pooled.
    """
    order: Int64[ndarray, " n"] = np.lexsort((values, group))
    sorted_group: Int64[ndarray, " n"] = group[order]
    sorted_values: Float64[ndarray, " n"] = values[order]
    _, start, count = np.unique(sorted_group, return_index=True, return_counts=True)
    keep: Bool[ndarray, " u"] = count >= min_count
    start, count = start[keep], count[keep]
    upper: Int64[ndarray, " g"] = start + count // 2
    lower: Int64[ndarray, " g"] = np.where(count % 2 == 0, upper - 1, upper)
    return 0.5 * (sorted_values[lower] + sorted_values[upper]), int(count.sum())


def estimate_metric_rescale(
    dataset: DatasetEntry,
    segment_id: str,
    cam_T_world: Float64[ndarray, "v 4 4"],
    intrinsics: Float64[ndarray, "v 3 3"],
    obs: ObservationSet,
    points_xyz: Float64[ndarray, "n 3"],
    sample_indices: Int64[ndarray, "t"],
    names: tuple[str, ...],
) -> float | None:
    """Estimate a global scale correction from MoGe-2 metric depth at the keypoints.

    For ``METRIC_RESCALE_FRAMES`` frames spread over the capture, each view's
    MoGe-2 metric depth is sampled at the surviving 2D observations and divided
    by the depth of the triangulated point in that camera; the median ratio over
    all (view, frame) groups is the correction (1.0 = the rig is already metric).
    Pooling thousands of depth ratios beats Stage A's few single-frame medians.

    Args:
        dataset: Catalog dataset entry (frames are decoded from it).
        segment_id: Segment being refined.
        cam_T_world: Refined world-to-camera transforms.
        intrinsics: Camera intrinsics at video resolution.
        obs: Surviving observations (pixel coordinates index the video frames).
        points_xyz: Triangulated points in the rig's (pre-rescale) frame.
        sample_indices: Stage B decoder sample index per correspondence frame.
        names: Exo camera entity names in view order.

    Returns:
        Median metric/current depth ratio over all valid groups, or ``None`` when fewer than
        ``MIN_RESCALE_GROUPS`` groups had enough valid samples.
    """
    # Evenly-spaced frames among those dense enough to form valid (view, frame)
    # groups. On dense captures every frame qualifies; on long captures the points
    # spread thin and a blind linspace would leave every group under the floor.
    n_views: int = cam_T_world.shape[0]
    observation_frames: Int64[ndarray, " n_obs"] = obs.point_frame_idx[obs.obs_point_idx]
    frames_used: Int64[ndarray, " f"] = np.unique(obs.point_frame_idx)
    obs_per_frame: Int64[ndarray, " f"] = np.bincount(np.searchsorted(frames_used, observation_frames), minlength=frames_used.size).astype(np.int64)
    eligible_frames: Int64[ndarray, " e"] = frames_used[obs_per_frame >= MIN_RESCALE_SAMPLES_PER_GROUP * n_views]
    if eligible_frames.size == 0:
        eligible_frames = frames_used
    chosen_frames: Int64[ndarray, " c"] = eligible_frames[
        np.linspace(0, eligible_frames.size - 1, min(METRIC_RESCALE_FRAMES, eligible_frames.size)).astype(np.int64)
    ]

    # Every observation on a chosen frame joins the (view, frame) group ``view * c + position``;
    # its triangulated depth in that camera comes from one batched transform.
    position: Int64[ndarray, " n_obs"] = np.minimum(np.searchsorted(chosen_frames, observation_frames), chosen_frames.size - 1)
    rows: Int64[ndarray, " r"] = np.flatnonzero(chosen_frames[position] == observation_frames)
    view: Int64[ndarray, " r"] = obs.obs_view_idx[rows]
    group: Int64[ndarray, " r"] = view * chosen_frames.size + position[rows]
    points: Float64[ndarray, "r 3"] = points_xyz[obs.obs_point_idx[rows]]
    finite: Bool[ndarray, " r"] = np.isfinite(points).all(axis=1)
    camera_rows: Float64[ndarray, "r 4 4"] = cam_T_world[view]
    current_z: Float64[ndarray, " r"] = np.einsum("rj,rj->r", camera_rows[:, 2, :3], points) + camera_rows[:, 2, 3]

    # MoGe runs once per (view, frame) group that has a finite point to compare against, one
    # batched NVDEC read per view; all surviving joints feed the estimator, since body-only
    # sampling has too few correspondences and MoGe's body depth runs a few percent short.
    streams: ExoVideoStreams = open_exo_streams(dataset, segment_id, names)
    moge: MoGeV2MetricPredictor = MoGeV2MetricPredictor(device="cuda")
    metric_z: Float64[ndarray, " r"] = np.full(rows.size, np.nan)
    for view_idx in np.unique(view[finite]):
        view_groups: Int64[ndarray, " k"] = np.unique(group[finite & (view == view_idx)])
        decode_idx: list[int] = [int(sample_indices[chosen_frames[int(g) - int(view_idx) * chosen_frames.size]]) for g in view_groups]
        frames: UInt8[Tensor, "k 3 h w"] = streams.decoders[int(view_idx)].get_frames_at(decode_idx).data
        for group_id, frame in zip(view_groups, frames, strict=True):
            frame_rgb: UInt8[ndarray, "h w 3"] = frame.permute(1, 2, 0).contiguous().cpu().numpy()
            depth: Float64[ndarray, "h w"] = moge(frame_rgb, K_33=intrinsics[int(view_idx)].astype(np.float32)).depth_meters.astype(np.float64)
            in_group: Bool[ndarray, " r"] = group == group_id
            px: Int64[ndarray, " m"] = np.clip(np.round(obs.obs_xy[rows[in_group], 0]).astype(np.int64), 0, depth.shape[1] - 1)
            py: Int64[ndarray, " m"] = np.clip(np.round(obs.obs_xy[rows[in_group], 1]).astype(np.int64), 0, depth.shape[0] - 1)
            metric_z[in_group] = depth[py, px]
    del moge
    torch.cuda.empty_cache()

    valid: Bool[ndarray, " r"] = finite & np.isfinite(metric_z) & (metric_z > MIN_DEPTH_M) & (current_z > MIN_DEPTH_M)
    group_medians, n_samples = grouped_medians(group[valid], metric_z[valid] / current_z[valid], MIN_RESCALE_SAMPLES_PER_GROUP)
    if group_medians.size < MIN_RESCALE_GROUPS:
        print(f"metric rescale skipped: only {group_medians.size} (view, frame) groups")
        return None
    scale: float = float(np.median(group_medians))
    print(f"metric rescale: {n_samples} samples in {group_medians.size} groups, median-of-medians = {scale:.4f}")
    return scale
