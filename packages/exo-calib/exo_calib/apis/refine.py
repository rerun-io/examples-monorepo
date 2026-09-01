"""Stage C+D: Kineo-style correspondences, robust triangulation, and staged BA.

Consumes Stage A (``exocalib_init`` layer) and Stage B (``exocalib_kp2d``
layer) straight from the catalog: applies
the AssemblyHands-X confidence post-processing (temporal median filter,
crop-margin downweight, threshold-rescale), builds spatio-temporal
correspondences with Kineo's pair rule, triangulates robustly against the
initial cameras, and refines the extrinsics with kornia-rs bundle adjustment
(re-triangulating between rounds). Logs init/refined 3D keypoint tracks with
Kineo's pairwise reprojection consensus confidence, plus the refined frusta.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.rerun_custom_types import Points3DWithConfidence

from exo_calib.apis.calibrate_init import log_cameras_layer
from exo_calib.apis.keypoints2d import (
    AHX_CONF_TAU,
    AHX_MARGIN_PX,
    AHX_MEDIAN_WINDOW,
    CameraKeypoints,
    load_stage_b,
    stack_postprocessed,
)
from exo_calib.catalog_io import (
    DEFAULT_CATALOG_URL,
    DEFAULT_DATASET_NAME,
    EXO_CAMERA_NAMES,
    EXOCALIB_ROOT,
    TIMELINE,
    RigCameras,
    connect_dataset,
    exocalib_entity,
    kp3d_entity,
    log_coco133_class_context,
    new_layer_recording,
    only_segment_id,
    read_rig_cameras,
    register_layer,
)
from exo_calib.correspondences import ObservationSet


@dataclass
class RefineConfig:
    """Config for the Stage C+D refinement tool."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """Rerun catalog server URL."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset holding the registered segment."""
    segment_id: str | None = None
    """Segment to refine; ``None`` uses the dataset's single segment."""
    output_dir: Path = Path("data/outputs")
    """Directory for the generated layer RRD."""
    median_window: int = AHX_MEDIAN_WINDOW
    """Temporal median filter window (frames) on the 2D keypoints."""
    margin_px: float = AHX_MARGIN_PX
    """AssemblyHands-X crop-edge margin, in crop pixels of the model crop frame."""
    conf_tau: float = AHX_CONF_TAU
    """AssemblyHands-X confidence threshold before rescaling to [0, 1]."""
    min_pair_conf: float = 0.90
    """Kineo pair rule: geometric-mean confidence a view pair must exceed.
    (Kineo's default is 0.75; 0.90 won the hill-climb on Assembly101.)"""
    frame_stride: int = 3
    """Temporal stride over Stage B frames when building correspondences."""
    max_points: int | None = 6000
    """Budget on 3D points entering bundle adjustment (None = no cap)."""
    reproj_threshold_px: float = 10.0
    """Per-observation reprojection threshold for robust triangulation."""
    ba_rounds: int = 2
    """Bundle-adjust / re-triangulate rounds."""
    ba_max_iterations: int = 50
    """LM iterations per bundle-adjust call."""
    robust: Literal["identity", "huber", "cauchy", "tukey"] = "cauchy"
    """Robust kernel for the BA residuals."""
    robust_scale_px: float = 1.0
    """Robust-kernel scale in pixels for the BA residuals."""
    pose_prior_sigma_m: float = 0.10
    """Soft prior (meters) anchoring camera centers to the Stage A estimate;
    pins the similarity gauge (scale drift) and keeps the BA a refinement."""
    fix_first_camera: bool = False
    """Hard-fix camera 0 instead of relying on the center priors for gauge."""
    metric_rescale_frames: int = 5
    """After BA, re-estimate the global metric scale from MoGe-2 depth at the
    tracked keypoints over this many frames per view (0 disables). Pools
    thousands of depth ratios instead of Stage A's eight single-frame medians."""
    lambda_decay: float = 1.0
    """Kineo exponential decay for the 3D point confidence."""
    seed: int = 0
    """Subsampling seed (deterministic budgets)."""
    layer_name: str = "exocalib_refined"
    """Catalog layer name for refined cameras + 3D keypoint tracks."""
    application_id: str = "exocalib"
    """Application id of generated layer recordings."""
    register: bool = True
    """Register the layer RRD into the catalog after writing it."""


def metric_rescale_from_keypoints(
    dataset: DatasetEntry,
    segment_id: str,
    cam_T_world_v44: Float64[ndarray, "v 4 4"],
    k_v33: Float64[ndarray, "v 3 3"],
    obs: ObservationSet,
    points_xyz_n3: Float64[ndarray, "n 3"],
    sample_indices_t: ndarray,
    n_frames: int,
) -> float:
    """Estimate a global scale correction from MoGe-2 metric depth at the keypoints.

    For ``n_frames`` frames spread over the window, each view's MoGe-2 metric
    depth is sampled at the surviving 2D observations and divided by the depth
    of the triangulated point in that camera; the median ratio over all samples
    is the correction (1.0 = the rig is already metric).

    Args:
        dataset: Catalog dataset entry (frames are decoded from it).
        segment_id: Segment being refined.
        cam_T_world_v44: Refined world-to-camera transforms.
        k_v33: Camera intrinsics at video resolution.
        obs: Surviving observations (pixel coordinates index the video frames).
        points_xyz_n3: Triangulated points in the rig's (pre-rescale) frame.
        sample_indices_t: Stage B decoder sample index per correspondence frame.
        n_frames: Number of distinct frames to sample.

    Returns:
        Median metric/current depth ratio over all valid samples.
    """
    import torch
    from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor

    from exo_calib.video_io import open_exo_streams

    # Evenly-spaced frames among those dense enough to form valid (view, frame)
    # groups (>= 3 samples per view). On dense windows every frame qualifies,
    # preserving the hill-climbed behavior exactly; on long windows the point
    # budget spreads to a few points per frame and a blind linspace would leave
    # every group under the validity floor (rescale silently skipped).
    n_views: int = cam_T_world_v44.shape[0]
    frames_used: Int64[ndarray, " f"] = np.unique(obs.point_frame_idx)
    obs_per_frame_f: Int64[ndarray, " f"] = np.bincount(
        np.searchsorted(frames_used, obs.point_frame_idx[obs.obs_point_idx]), minlength=frames_used.size
    ).astype(np.int64)
    eligible_f: Int64[ndarray, " e"] = frames_used[obs_per_frame_f >= 3 * n_views]
    if eligible_f.size == 0:
        eligible_f = frames_used
    chosen_frames_f: Int64[ndarray, " f"] = eligible_f[np.linspace(0, eligible_f.size - 1, min(n_frames, eligible_f.size)).astype(np.int64)]
    streams = open_exo_streams(dataset, segment_id)
    moge: MoGeV2MetricPredictor = MoGeV2MetricPredictor(device="cuda")

    # Empirically the all-joint estimator tracks GT scale best on Assembly101:
    # body-only sampling has too few surviving correspondences, and MoGe's body
    # depth runs a few percent short on this scene.
    frame_of_obs_m: Int64[ndarray, "n_obs"] = obs.point_frame_idx[obs.obs_point_idx]
    group_medians: list[float] = []
    n_samples: int = 0
    for view in np.unique(obs.obs_view_idx):
        frame_rows: list[tuple[int, Int64[ndarray, " r"]]] = []
        for frame in chosen_frames_f:
            rows_r: Int64[ndarray, " r"] = np.flatnonzero((frame_of_obs_m == frame) & (obs.obs_view_idx == view)).astype(np.int64)
            if rows_r.size and np.isfinite(points_xyz_n3[obs.obs_point_idx[rows_r]]).all(axis=1).any():
                frame_rows.append((int(frame), rows_r))
        if not frame_rows:
            continue
        # One batched NVDEC read per view instead of interleaved single-frame seeks.
        decode_idx: list[int] = [int(sample_indices_t[frame]) for frame, _ in frame_rows]
        frames_bchw: torch.Tensor = streams.decoders[int(view)].get_frames_at(decode_idx).data
        for (_frame, rows_r), frame_chw in zip(frame_rows, frames_bchw, strict=True):
            point_rows_r: Int64[ndarray, " r"] = obs.obs_point_idx[rows_r]
            points_r3: Float64[ndarray, "r 3"] = points_xyz_n3[point_rows_r]
            finite_r: ndarray = np.isfinite(points_r3).all(axis=1)
            frame_rgb: ndarray = frame_chw.permute(1, 2, 0).contiguous().cpu().numpy()
            depth_hw: Float64[ndarray, "h w"] = moge(frame_rgb, K_33=k_v33[int(view)].astype(np.float32)).depth_meters.astype(np.float64)
            xy_r2: Float64[ndarray, "r 2"] = obs.obs_xy[rows_r]
            px_r: Int64[ndarray, " r"] = np.clip(np.round(xy_r2[:, 0]).astype(np.int64), 0, depth_hw.shape[1] - 1)
            py_r: Int64[ndarray, " r"] = np.clip(np.round(xy_r2[:, 1]).astype(np.int64), 0, depth_hw.shape[0] - 1)
            metric_z_r: Float64[ndarray, " r"] = depth_hw[py_r, px_r]
            points_homo_r4: Float64[ndarray, "r 4"] = np.column_stack((points_r3, np.ones(points_r3.shape[0])))
            current_z_r: Float64[ndarray, " r"] = (cam_T_world_v44[int(view)][:3, :] @ points_homo_r4.T).T[:, 2]
            valid_r: ndarray = finite_r & np.isfinite(metric_z_r) & (metric_z_r > 0.1) & (current_z_r > 0.1)
            if int(valid_r.sum()) >= 3:
                group_medians.append(float(np.median(metric_z_r[valid_r] / current_z_r[valid_r])))
                n_samples += int(valid_r.sum())
    del moge
    torch.cuda.empty_cache()
    if len(group_medians) < 8:
        print(f"metric rescale skipped: only {len(group_medians)} (view, frame) groups")
        return 1.0
    scale: float = float(np.median(np.asarray(group_medians)))
    print(f"metric rescale: {n_samples} samples in {len(group_medians)} groups, median-of-medians = {scale:.4f}")
    return scale


def main(config: RefineConfig) -> None:
    """Run correspondence building, robust triangulation, staged BA, and layer logging."""
    from exo_calib.ba import mean_reprojection_error_px, refine_extrinsics
    from exo_calib.correspondences import build_observations
    from exo_calib.triangulation import triangulate_robust

    dataset = connect_dataset(config.catalog_url, config.dataset_name)
    segment_id: str = config.segment_id or only_segment_id(dataset)
    segment_dir: Path = config.output_dir / segment_id
    init: RigCameras = read_rig_cameras(dataset, segment_id, root=exocalib_entity("init"))
    per_camera: dict[str, CameraKeypoints] = load_stage_b(dataset, segment_id, EXO_CAMERA_NAMES)

    kp_xy_vtj2, conf_vtj = stack_postprocessed(per_camera, EXO_CAMERA_NAMES, config.median_window, config.margin_px, config.conf_tau)
    print(f"{kp_xy_vtj2.shape[1]} frames x {len(EXO_CAMERA_NAMES)} views; mean post conf {conf_vtj[conf_vtj > 0].mean():.3f}")

    obs = build_observations(
        kp_xy_vtj2,
        conf_vtj,
        frame_stride=config.frame_stride,
        min_pair_conf=config.min_pair_conf,
        max_points=config.max_points,
        seed=config.seed,
    )
    print(f"observations: {len(obs.obs_view_idx)} over {len(obs.point_frame_idx)} points")

    points_init, obs = triangulate_robust(
        obs, init.k_v33, init.cam_T_world_v44, reproj_threshold_px=config.reproj_threshold_px
    )
    err_init: float = mean_reprojection_error_px(obs, points_init, init.k_v33, init.cam_T_world_v44)
    print(f"init mean reprojection error: {err_init:.2f} px")

    result = refine_extrinsics(
        init.cam_T_world_v44,
        init.k_v33,
        obs,
        points_init,
        rounds=config.ba_rounds,
        robust=config.robust,
        robust_scale_px=config.robust_scale_px,
        max_iterations=config.ba_max_iterations,
        pose_prior_sigma_m=config.pose_prior_sigma_m,
        fixed_pose_indices=(0,) if config.fix_first_camera else (),
    )
    print(f"BA reprojection error per round (px): {[round(e, 3) for e in result.mean_reproj_px_per_round]}, converged={result.converged}")

    scale_fix: float = 1.0
    if config.metric_rescale_frames > 0:
        scale_fix = metric_rescale_from_keypoints(
            dataset,
            segment_id,
            result.cam_T_world_v44,
            init.k_v33,
            obs,
            result.points_xyz_n3,
            per_camera[EXO_CAMERA_NAMES[0]].sample_indices,
            config.metric_rescale_frames,
        )
        result.cam_T_world_v44[:, :3, 3] *= scale_fix
        result.points_xyz_n3 *= scale_fix

    # Kineo pairwise reprojection consensus confidences for both camera sets.
    conf3d_by_variant: dict[str, tuple[Float64[ndarray, "n 3"], Float64[ndarray, " n"]]] = {}
    for variant, cams_T, points in (("init", init.cam_T_world_v44, points_init), ("refined", result.cam_T_world_v44, result.points_xyz_n3)):
        conf_points: Float64[ndarray, " n"] = kineo_point_confidence_for_obs(
            points, obs, init.k_v33, cams_T, kp_xy_vtj2, conf_vtj, config.lambda_decay
        )
        conf3d_by_variant[variant] = (points, conf_points)
        valid = np.isfinite(points).all(axis=1)
        print(f"{variant}: {int(valid.sum())} points, mean 3D confidence {conf_points[valid].mean():.3f}")

    recording, rrd_path = new_layer_recording(config.application_id, segment_id, segment_dir / f"{config.layer_name}.rrd")
    recording.log(exocalib_entity("refined"), rr.AnyValues(metric_rescale_fix=np.array([scale_fix])), static=True)
    log_coco133_class_context(
        recording,
        EXOCALIB_ROOT,
        ((1, "kp3d init", (255, 214, 0)), (2, "kp3d refined", (0, 200, 255))),
    )
    first_cam: CameraKeypoints = per_camera[EXO_CAMERA_NAMES[0]]
    video_wh: tuple[int, int] = (int(first_cam.video_wh[0]), int(first_cam.video_wh[1]))
    log_cameras_layer(init.k_v33, result.cam_T_world_v44, tuple(init.names), video_wh, exocalib_entity("refined"), recording)
    for class_id, (variant, (points, conf_points)) in enumerate(conf3d_by_variant.items(), start=1):
        _log_point_tracks(recording, kp3d_entity(variant), obs, points, conf_points, first_cam.times_ns, class_id)
    recording.flush(timeout_sec=30.0)
    print(f"wrote {rrd_path}")
    if config.register:
        register_layer(dataset, rrd_path, config.layer_name)
        print(f"registered layer {config.layer_name}")


def kineo_point_confidence_for_obs(
    points_xyz_n3: Float64[ndarray, "n 3"],
    obs: ObservationSet,
    k_v33: Float64[ndarray, "v 3 3"],
    cam_T_world_v44: Float64[ndarray, "v 4 4"],
    kp_xy_vtj2: Float64[ndarray, "v t 133 2"],
    conf_vtj: Float64[ndarray, "v t 133"],
    lambda_decay: float,
) -> Float64[ndarray, " n"]:
    """Score every triangulated point with the Kineo pairwise consensus confidence.

    Builds the dense per-view keypoint/confidence arrays for each point's
    (frame, joint) and delegates to :func:`exo_calib.confidence.kineo_point_confidence`.
    """
    from exo_calib.confidence import kineo_point_confidence

    num_views: int = k_v33.shape[0]
    kp_view_n2: Float64[ndarray, "v n 2"] = np.stack(
        [kp_xy_vtj2[v, obs.point_frame_idx, obs.point_joint_idx] for v in range(num_views)]
    )
    conf_view_n: Float64[ndarray, "v n"] = np.stack([conf_vtj[v, obs.point_frame_idx, obs.point_joint_idx] for v in range(num_views)])
    return kineo_point_confidence(points_xyz_n3, kp_view_n2, conf_view_n, k_v33, cam_T_world_v44, lambda_decay=lambda_decay)


def _log_point_tracks(
    recording: rr.RecordingStream,
    entity: str,
    obs: ObservationSet,
    points_xyz_n3: Float64[ndarray, "n 3"],
    conf_n: Float64[ndarray, " n"],
    times_ns: ndarray,
    class_id: int,
) -> None:
    """Log triangulated points frame-by-frame with per-point Kineo confidences."""
    for frame in np.unique(obs.point_frame_idx):
        rows: ndarray = np.nonzero(obs.point_frame_idx == frame)[0]
        points: Float64[ndarray, "m 3"] = points_xyz_n3[rows]
        keep: ndarray = np.isfinite(points).all(axis=1)
        if not keep.any():
            continue
        recording.set_time(TIMELINE, duration=1e-9 * float(times_ns[int(frame)]))
        recording.log(
            entity,
            Points3DWithConfidence(
                positions=points[keep],
                confidences=conf_n[rows][keep],
                class_ids=class_id,
                keypoint_ids=obs.point_joint_idx[rows][keep].astype(np.int64),
                radii=0.008,
            ),
        )
