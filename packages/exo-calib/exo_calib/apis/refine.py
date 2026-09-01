"""Stage C+D: Kineo-style correspondences, robust triangulation, and staged BA.

Consumes Stage A (``init_cameras.npz``) and Stage B (``kp2d/*.npz``): applies
the AssemblyHands-X confidence post-processing (temporal median filter,
crop-margin downweight, threshold-rescale), builds spatio-temporal
correspondences with Kineo's pair rule, triangulates robustly against the
initial cameras, and refines the extrinsics with kornia-rs bundle adjustment
(re-triangulating between rounds). Logs init/refined 3D keypoint tracks with
Kineo's pairwise reprojection consensus confidence, plus the refined frusta.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
from jaxtyping import Float32, Float64
from numpy import ndarray
from simplecv.rerun_custom_types import Points3DWithConfidence

from exo_calib.apis.calibrate_init import InitCameras, log_cameras_layer
from exo_calib.apis.keypoints2d import CameraKeypoints
from exo_calib.catalog_io import (
    DEFAULT_CATALOG_URL,
    DEFAULT_DATASET_NAME,
    EXO_CAMERA_NAMES,
    TIMELINE,
    connect_dataset,
    new_layer_recording,
    only_segment_id,
    register_layer,
)


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
    """Directory holding Stage A/B outputs; refined outputs land beside them."""
    median_window: int = 5
    """Temporal median filter window (frames) on the 2D keypoints."""
    margin_px: float = 32.0
    """AssemblyHands-X crop-edge margin, in crop pixels (crop is 768x1024)."""
    conf_tau: float = 0.15
    """AssemblyHands-X confidence threshold before rescaling to [0, 1]."""
    min_pair_conf: float = 0.75
    """Kineo pair rule: geometric-mean confidence a view pair must exceed."""
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
    robust_scale_px: float = 2.0
    """Huber scale in pixels for the BA residuals."""
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


def load_stage_b(output_dir: Path, segment_id: str, names: tuple[str, ...]) -> dict[str, CameraKeypoints]:
    """Load Stage B per-camera NPZs."""
    per_camera: dict[str, CameraKeypoints] = {}
    for name in names:
        npz_path: Path = output_dir / segment_id / "kp2d" / f"{name.replace('/', '_')}.npz"
        per_camera[name] = CameraKeypoints.load(npz_path)
    return per_camera


def postprocess_confidences(cam: CameraKeypoints, median_window: int, margin_px: float, conf_tau: float, crop_wh: tuple[int, int]) -> tuple[
    Float32[ndarray, "t 133 2"], Float64[ndarray, "t 133"]
]:
    """Apply the AssemblyHands-X 2D post-processing to one camera's keypoints.

    Returns the median-filtered keypoints and the modulated + rescaled confidences.
    """
    from exo_calib.confidence import median_filter_keypoints, modulate_crop_edge_confidence, threshold_rescale_confidence

    kp_xy_t2: Float64[ndarray, "t 133 2"] = median_filter_keypoints(cam.kp_xy.astype(np.float64), window=median_window)
    num_frames: int = kp_xy_t2.shape[0]
    conf_out: Float64[ndarray, "t 133"] = np.zeros((num_frames, 133), dtype=np.float64)
    for t in range(num_frames):
        if not np.isfinite(cam.crop_origin_xy[t]).all():
            continue
        size_wh: Float64[ndarray, "2"] = cam.crop_size_wh[t].astype(np.float64)
        kp_crop_xy: Float64[ndarray, "133 2"] = (kp_xy_t2[t] - cam.crop_origin_xy[t].astype(np.float64)) / size_wh * np.asarray(crop_wh, dtype=np.float64)
        modulated: Float64[ndarray, "133"] = modulate_crop_edge_confidence(kp_crop_xy, cam.conf[t].astype(np.float64), crop_wh, margin_px)
        conf_out[t] = threshold_rescale_confidence(modulated, tau=conf_tau)
    return kp_xy_t2.astype(np.float32), conf_out


def main(config: RefineConfig) -> None:
    """Run correspondence building, robust triangulation, staged BA, and layer logging."""
    from exo_calib.ba import mean_reprojection_error_px, refine_extrinsics
    from exo_calib.correspondences import build_observations
    from exo_calib.triangulation import triangulate_robust

    dataset = connect_dataset(config.catalog_url, config.dataset_name)
    segment_id: str = config.segment_id or only_segment_id(dataset)
    segment_dir: Path = config.output_dir / segment_id
    init: InitCameras = InitCameras.load(segment_dir / "init_cameras.npz")
    per_camera: dict[str, CameraKeypoints] = load_stage_b(config.output_dir, segment_id, EXO_CAMERA_NAMES)

    # Common frame grid across cameras (Stage B used the same stride everywhere).
    num_frames: int = min(len(cam.sample_indices) for cam in per_camera.values())
    kp_list: list[Float32[ndarray, "t 133 2"]] = []
    conf_list: list[Float64[ndarray, "t 133"]] = []
    for name in EXO_CAMERA_NAMES:
        kp_xy, conf = postprocess_confidences(per_camera[name], config.median_window, config.margin_px, config.conf_tau, crop_wh=(768, 1024))
        kp_list.append(kp_xy[:num_frames])
        conf_list.append(conf[:num_frames])
    kp_xy_vtj2: Float64[ndarray, "v t 133 2"] = np.stack(kp_list).astype(np.float64)
    conf_vtj: Float64[ndarray, "v t 133"] = np.stack(conf_list)
    print(f"{num_frames} frames x {len(EXO_CAMERA_NAMES)} views; mean post conf {conf_vtj[conf_vtj > 0].mean():.3f}")

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
        robust_scale_px=config.robust_scale_px,
        max_iterations=config.ba_max_iterations,
    )
    print(f"BA reprojection error per round (px): {[round(e, 3) for e in result.mean_reproj_px_per_round]}, converged={result.converged}")

    refined = InitCameras(names=init.names, k_v33=init.k_v33, cam_T_world_v44=result.cam_T_world_v44, metric_scale=init.metric_scale)
    refined.save(segment_dir / "refined_cameras.npz")
    print(f"wrote {segment_dir / 'refined_cameras.npz'}")

    # Kineo pairwise reprojection consensus confidences for both camera sets.
    conf3d_by_variant: dict[str, tuple[Float64[ndarray, "n 3"], Float64[ndarray, " n"]]] = {}
    for variant, cams_T, points in (("init", init.cam_T_world_v44, points_init), ("refined", result.cam_T_world_v44, result.points_xyz_n3)):
        conf_points: Float64[ndarray, " n"] = kineo_point_confidence_for_obs(
            points, obs, init.k_v33, cams_T, kp_xy_vtj2, conf_vtj, config.lambda_decay
        )
        conf3d_by_variant[variant] = (points, conf_points)
        valid = np.isfinite(points).all(axis=1)
        print(f"{variant}: {int(valid.sum())} points, mean 3D confidence {conf_points[valid].mean():.3f}")

    rrd_path: Path = segment_dir / f"{config.layer_name}.rrd"
    recording: rr.RecordingStream = new_layer_recording(config.application_id, segment_id, rrd_path)
    video_wh: tuple[int, int] = (1280, 720)
    log_cameras_layer(init.k_v33, result.cam_T_world_v44, tuple(init.names), video_wh, "/world/exocalib/refined", recording)
    times_ns = per_camera[EXO_CAMERA_NAMES[0]].times_ns
    for variant, (points, conf_points) in conf3d_by_variant.items():
        _log_point_tracks(recording, f"/world/exocalib/kp3d_{variant}", obs, points, conf_points, times_ns)
    recording.flush(timeout_sec=30.0)
    print(f"wrote {rrd_path}")
    if config.register:
        register_layer(dataset, rrd_path, config.layer_name)
        print(f"registered layer {config.layer_name}")


def kineo_point_confidence_for_obs(
    points_xyz_n3: Float64[ndarray, "n 3"],
    obs: Any,
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
    obs: Any,
    points_xyz_n3: Float64[ndarray, "n 3"],
    conf_n: Float64[ndarray, " n"],
    times_ns: ndarray,
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
                keypoint_ids=obs.point_joint_idx[rows][keep].astype(np.int64),
                radii=0.008,
            ),
        )
