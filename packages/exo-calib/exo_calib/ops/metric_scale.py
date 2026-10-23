"""Global metric rescale of a refined rig: from metric depth at the surviving 2D observations (depth model injected) and from the
person's arm bones, fused in log space."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

from exo_calib.ops.correspondences import ObservationSet, build_observations
from exo_calib.ops.triangulation import RobustTriangulationResult, triangulate_robust

DepthAt: TypeAlias = Callable[[int, list[int]], list[Float64[ndarray, "h w"]]]
"""(view, decoder sample indices) -> metric depth (metres) of each of those frames at video resolution."""

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
    cam_T_world: Float64[ndarray, "v 4 4"],
    obs: ObservationSet,
    points_xyz: Float64[ndarray, "n 3"],
    video_frames: Int64[ndarray, "v t"],
    depth_at: DepthAt,
) -> float | None:
    """Estimate a global scale correction from metric depth (MoGe-2 in the pipeline) at the keypoints.

    For ``METRIC_RESCALE_FRAMES`` frames spread over the capture, each view's metric depth is sampled at the surviving
    2D observations and divided by the depth of the triangulated point in that camera; the median of the per-(view,
    frame) medians is the correction (1.0 = the rig is already metric). Pooling thousands of depth ratios beats
    Stage A's few single-frame medians.

    Args:
        cam_T_world: Refined world-to-camera transforms.
        obs: Surviving observations; their pixel coordinates index the frames ``depth_at`` measures (outside them: skipped).
        points_xyz: Triangulated points in the rig's (pre-rescale) frame.
        video_frames: Each camera's own video frame of every correspondence frame (cameras that start apart differ).
        depth_at: Metric depth of the requested frames of one view.

    Returns:
        The correction, or ``None`` when fewer than ``MIN_RESCALE_GROUPS`` groups had enough valid samples.
    """
    # Evenly-spaced frames among those dense enough to form valid (view, frame) groups. On dense captures every
    # frame qualifies; on long captures the points spread thin and a blind linspace would leave every group under the floor.
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

    # Depth runs once per (view, frame) group that has a finite point to compare against; all surviving joints feed
    # the estimator, since body-only sampling has too few correspondences and MoGe's body depth runs a few percent short.
    metric_z: Float64[ndarray, " r"] = np.full(rows.size, np.nan)
    for view_idx in np.unique(view[finite]).tolist():
        view_groups: Int64[ndarray, " k"] = np.unique(group[finite & (view == view_idx)])
        decode_idx: list[int] = [int(video_frames[view_idx, chosen_frames[int(g) - view_idx * chosen_frames.size]]) for g in view_groups]
        for group_id, depth in zip(view_groups, depth_at(view_idx, decode_idx), strict=True):
            in_group: Bool[ndarray, " r"] = group == group_id
            px: Int64[ndarray, " m"] = np.round(obs.obs_xy[rows[in_group], 0]).astype(np.int64)
            py: Int64[ndarray, " m"] = np.round(obs.obs_xy[rows[in_group], 1]).astype(np.int64)
            in_frame: Bool[ndarray, " m"] = (px >= 0) & (px < depth.shape[1]) & (py >= 0) & (py < depth.shape[0])  # undistorted pixels can leave it
            metric_z[np.flatnonzero(in_group)[in_frame]] = depth[py[in_frame], px[in_frame]]

    valid: Bool[ndarray, " r"] = finite & np.isfinite(metric_z) & (metric_z > MIN_DEPTH_M) & (current_z > MIN_DEPTH_M)
    group_medians, n_samples = grouped_medians(group[valid], metric_z[valid] / current_z[valid], MIN_RESCALE_SAMPLES_PER_GROUP)
    if group_medians.size < MIN_RESCALE_GROUPS:
        print(f"metric rescale skipped: only {group_medians.size} (view, frame) groups")
        return None
    scale: float = float(np.median(group_medians))
    print(f"metric rescale: {n_samples} samples in {group_medians.size} groups, median-of-medians = {scale:.4f}")
    return scale


ARM_BONES: dict[str, tuple[float, tuple[tuple[int, int], ...]]] = {"upper arm": (0.260, ((5, 7), (6, 8))), "forearm": (0.237, ((7, 9), (8, 10)))}
"""Each bone kind's length (metres) in an average adult as RTMW places its COCO-17 joints, and its (shoulder, elbow) or (elbow,
wrist) pairs: the SMPL-X neutral mean shape's upper arm 0.272 m and forearm 0.252 m, times 0.957 and 0.942, the ratios RTMW's
joints measured through the GT rigs of 10 Ego-Exo4D, HOCap and Assembly101 takes. Arms only: seated hips and the shoulder width
are not well defined in COCO."""
ARM_JOINTS: list[int] = sorted({joint for _, pairs in ARM_BONES.values() for pair in pairs for joint in pair})
"""COCO-17 shoulders, elbows and wrists."""
MIN_ARM_BONES: int = 10
"""Fewest measured lengths per bone kind (upper arm, forearm) before the body's scale is trusted."""
ARM_PAIR_CONFIDENCE: float = 0.6
"""Pair rule for the arm joints: the 0.9 of the refinement's correspondences keeps almost no shoulder or wrist."""
ARM_REPROJECTION_THRESHOLD_PX: float = 15.0
"""Per-observation gate when the arm joints are triangulated through the refined rig."""
BODY_SCALE_WEIGHT: float = 0.75
"""Weight of the person's arm bones against MoGe-2's scene depth in the metric rescale (log space): inverse variances of their
errors on the harness takes (arms about 7 %, MoGe on rectified frames about 12 %)."""


def triangulate_arms(
    xy: Float64[ndarray, "v t 133 2"], conf: Float64[ndarray, "v t 133"], K: Float64[ndarray, "v 3 3"], cam_T_world: Float64[ndarray, "v 4 4"]
) -> RobustTriangulationResult:
    """The shoulders, elbows and wrists of every frame through a rig (pinhole pixels), with their own pair rule and gate."""
    arm_conf: Float64[ndarray, "v t 133"] = np.zeros_like(conf)
    arm_conf[..., ARM_JOINTS] = conf[..., ARM_JOINTS]
    candidates: ObservationSet = build_observations(xy, arm_conf, min_pair_conf=ARM_PAIR_CONFIDENCE, min_views=2)
    return triangulate_robust(candidates, K, cam_T_world, reproj_threshold_px=ARM_REPROJECTION_THRESHOLD_PX)


def estimate_body_rescale(points_xyz: Float64[ndarray, "n 3"], joint_idx: Int64[ndarray, " n"], frame_idx: Int64[ndarray, " n"]) -> float | None:
    """The factor that makes a rig metric from the person's arms: an average adult's arm bones over their median lengths in
    the triangulated points (``joint_idx``: COCO-133 joint, ``frame_idx``: frame of each point; one point per pair).

    A person's size spreads about 6 % around the average (SMPL-X's first shape component, one sigma), so this is a prior on
    scale, independent of the scene depth MoGe reads.

    Returns:
        The correction (1.0 = metric), or ``None`` when an upper arm or a forearm was measured fewer than ``MIN_ARM_BONES`` times
        (a bone of zero length, its joints triangulated onto one point, measures nothing).
    """
    finite: Bool[ndarray, " n"] = np.isfinite(points_xyz).all(axis=1)
    by_joint: dict[tuple[int, int], Float64[ndarray, "3"]] = {(int(f), int(j)): p for f, j, p in zip(frame_idx[finite], joint_idx[finite], points_xyz[finite], strict=True)}
    frames: list[int] = np.unique(frame_idx).tolist()
    ratios: list[float] = []
    for length_m, pairs in ARM_BONES.values():
        lengths = [np.linalg.norm(by_joint[(f, b)] - by_joint[(f, a)]) / length_m for a, b in pairs for f in frames if (f, a) in by_joint and (f, b) in by_joint]
        lengths = [length for length in lengths if length > 0.0]
        if len(lengths) < MIN_ARM_BONES:
            return None
        ratios.append(float(np.median(lengths)))
    return 1.0 / float(np.mean(ratios))


def fuse_metric_scales(depth: float | None, body: float | None, *, body_weight: float) -> float:
    """One correction from the scene-depth and the body estimates: their weighted geometric mean (either alone when the other
    is missing, 1.0 when neither was measured: the rig keeps its scale). The two err for unrelated reasons (a depth model's
    bias in a room vs one person's size), so they average."""
    if depth is None:
        return 1.0 if body is None else body
    if body is None:
        return depth
    return float(np.exp((1.0 - body_weight) * np.log(depth) + body_weight * np.log(body)))


@dataclass(frozen=True, slots=True)
class MetricRescale:
    """The factor that makes a refined rig metric, and the two estimates it fuses."""

    depth: float | None
    """MoGe-2 depth at the observations (``estimate_metric_rescale``); None when too few depth samples."""
    body: float | None
    """The person's arm bones (``estimate_body_rescale``); None when too few bones."""
    factor: float
    """``fuse_metric_scales`` of the two (1.0 when neither was measured)."""


def metric_rescale(
    xy: Float64[ndarray, "v t 133 2"],
    conf: Float64[ndarray, "v t 133"],
    K: Float64[ndarray, "v 3 3"],
    cam_T_world: Float64[ndarray, "v 4 4"],
    obs: ObservationSet,
    points_xyz: Float64[ndarray, "n 3"],
    video_frames: Int64[ndarray, "v t"],
    depth_at: DepthAt,
) -> MetricRescale:
    """MoGe-2 depth at the refined rig's observations (``obs``, ``points_xyz``) fused with the person's arm bones, which this
    triangulates from every view's pinhole pixels ``xy`` with their own pair rule (``triangulate_arms``)."""
    depth: float | None = estimate_metric_rescale(cam_T_world, obs, points_xyz, video_frames, depth_at)
    arms: RobustTriangulationResult = triangulate_arms(xy, conf, K, cam_T_world)
    body: float | None = estimate_body_rescale(arms.points_xyz, arms.obs.point_joint_idx, arms.obs.point_frame_idx)
    factor: float = fuse_metric_scales(depth, body, body_weight=BODY_SCALE_WEIGHT)
    print(f"metric rescale: depth {depth} body {body} -> {factor}")
    return MetricRescale(depth, body, factor)
