"""Stage A: self-contained initial exo-camera calibration.

One synchronized frame from each exo video (read from the catalog, NVDEC) goes
through the gravity-aligned multi-view backend (G3T), MoGe-2 metric depth fixes
the global scale, and a RANSAC ground plane puts gravity on +Z with the floor at
z=0 — the ``monopriors.apis.calibrate_synced_videos`` metric route, catalog-native.
Ground truth is never read here; the result is evaluated against GT elsewhere.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import open3d as o3d
import rerun as rr
import tyro
from jaxtyping import Bool, Float64, UInt8
from numpy import ndarray
from simplecv.camera_orient_utils import rotation_matrix_between

from exo_calib.cameras import RigCameras, camera_centers
from exo_calib.catalog_io import StageConfig, StageContext, stage_context
from exo_calib.layer_io import INIT_LAYER, VARIANT_COLOR, exocalib_entity, log_cameras_layer, new_layer_recording, register_layer
from exo_calib.video_io import ExoVideoStreams, open_exo_streams

KEEP_TOP_PERCENT: float = 30.0
"""Percentage of high-confidence pixels the multi-view geometry pass retains."""
GROUND_DISTANCE_M: float = 0.06
"""RANSAC inlier distance (meters) for the ground-plane fit."""
MAX_GROUND_TILT_DEG: float = 60.0
"""Largest angle between a candidate floor normal and G3T's gravity before the
plane is rejected as a wall or table and the next-largest plane is tried."""
UNPROJECT_STRIDE: int = 24
"""Pixel stride when unprojecting depth to the ground-fit point cloud."""
UP: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0])
"""G3T normalizes its predicted gravity to +Z; the floor fit refines that direction."""
GROUND_PLANE_ATTEMPTS: int = 5
"""Largest RANSAC planes tried before falling back to the gravity prior and the lowest camera."""


@dataclass
class InitCalibrationConfig:
    """Config for the Stage A initial calibration tool."""

    stage: tyro.conf.OmitArgPrefixes[StageConfig] = field(default_factory=StageConfig)
    """Shared stage flags (catalog, dataset, segment, rigs, output, register); the CLI shows them unprefixed."""
    frame_index: int = 0
    """Sample index of the synchronized frame used for calibration (rig is static)."""


class InitEstimate(NamedTuple):
    """Stage A's output: the metric, +Z-up rig (ground at z=0) and the MoGe-2 scale that made it metric."""

    cameras: RigCameras
    metric_scale: float


def _unproject_world(
    depth: Float64[ndarray, "h w"],
    intrinsics: Float64[ndarray, "3 3"],
    cam_T_world: Float64[ndarray, "4 4"],
    valid: Bool[ndarray, "h w"],
    stride: int,
) -> Float64[ndarray, "n 3"]:
    """Lift confident depth pixels of one view to world points."""
    from monopriors.models.multiview.multiview_pointcloud import unproject_selected_pixels

    ys, xs = np.where(valid)
    ys, xs = ys[::stride], xs[::stride]
    world_T_cam: Float64[ndarray, "4 4"] = np.linalg.inv(cam_T_world)
    return unproject_selected_pixels(depth, intrinsics, world_T_cam, ys, xs).astype(np.float64, copy=False)


def select_ground_plane(
    cloud: Float64[ndarray, "n 3"],
    camera_centers: Float64[ndarray, "v 3"],
    up_prior: Float64[ndarray, "3"],
    *,
    distance_threshold: float,
    max_tilt_deg: float,
    max_attempts: int = GROUND_PLANE_ATTEMPTS,
) -> tuple[Float64[ndarray, "3"], Float64[ndarray, "3"]]:
    """Pick the floor: the largest RANSAC plane whose normal agrees with the gravity prior.

    Planes tilted more than ``max_tilt_deg`` from ``up_prior`` (walls, shelves)
    have their inliers removed and the fit repeats on the remainder. When no
    plane qualifies, the prior itself is the up direction and the ground passes
    through the lowest camera center.

    Args:
        cloud: Float64 metric world points with shape ``(n, 3)``.
        camera_centers: Float64 camera centers with shape ``(v, 3)``.
        up_prior: Float64 gravity-up direction with shape ``(3,)``.
        distance_threshold: RANSAC inlier distance in meters.
        max_tilt_deg: Maximum accepted angle between the plane normal and the prior.
        max_attempts: Number of successive plane fits before falling back.

    Returns:
        Float64 up direction with shape ``(3,)`` and a point on the ground with shape ``(3,)``.
    """
    up_prior = up_prior / np.linalg.norm(up_prior)
    remaining_points: Float64[ndarray, "m 3"] = cloud
    min_cosine: float = float(np.cos(np.radians(max_tilt_deg)))
    for _attempt in range(max_attempts):
        if remaining_points.shape[0] < 3:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining_points)
        plane, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=2000)
        normal: Float64[ndarray, "3"] = np.array(plane[:3], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        if float(np.dot(normal, up_prior)) < 0.0:
            normal = -normal
        ground_point: Float64[ndarray, "3"] = remaining_points[inliers].mean(axis=0)
        if float(np.dot(normal, up_prior)) >= min_cosine:
            return normal, ground_point
        keep_mask: Bool[ndarray, " n"] = np.ones(remaining_points.shape[0], dtype=bool)
        keep_mask[np.asarray(inliers, dtype=np.int64)] = False
        remaining_points = remaining_points[keep_mask]
    lowest_camera: Float64[ndarray, "3"] = camera_centers[np.argmin(camera_centers @ up_prior)]
    return up_prior, lowest_camera


def estimate_init_cameras(frames_rgb: list[UInt8[ndarray, "h w 3"]], names: tuple[str, ...]) -> InitEstimate:
    """Estimate metric, gravity-aligned exo cameras from one synchronized frame set.

    Args:
        frames_rgb: One RGB frame per camera, native video resolution.
        names: Camera entity names, index-aligned with ``frames_rgb``.

    Returns:
        Estimated cameras with intrinsics rescaled to the native video resolution.
    """
    import cv2
    from monopriors.apis.multiview_geometry import MultiviewGeometryConfig, MultiviewGeometryResult, run_multiview_geometry
    from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
    from monopriors.models.multiview.multiview_predictor import MultiviewPredictor, MultiviewPredictorConfig

    predictor: MultiviewPredictor = MultiviewPredictor(MultiviewPredictorConfig(model_name="g3t"))
    geo: MultiviewGeometryResult = run_multiview_geometry(
        rgb_list=frames_rgb,
        multiview_predictor=predictor,
        config=MultiviewGeometryConfig(keep_top_percent=KEEP_TOP_PERCENT),
    )
    mv = geo.mv_pred_list
    intrinsics_per_view: list[Float64[ndarray, "3 3"]] = [np.asarray(p.pinhole_param.intrinsics.k_matrix, dtype=np.float64) for p in mv]
    cam_T_world_per_view: list[Float64[ndarray, "4 4"]] = [np.asarray(p.pinhole_param.extrinsics.cam_T_world, dtype=np.float64) for p in mv]

    # MoGe-2 metric depth -> one global scale (median over views of per-view median ratio).
    moge: MoGeV2MetricPredictor = MoGeV2MetricPredictor(device="cuda")
    metric_per: list[Float64[ndarray, "h w"]] = []
    valid_per: list[Bool[ndarray, "h w"]] = []
    per_view_scale: list[float] = []
    for i, pred in enumerate(mv):
        metric_depth: Float64[ndarray, "h w"] = moge(frames_rgb[i], K_33=intrinsics_per_view[i].astype(np.float32)).depth_meters.astype(np.float64)
        mv_depth: Float64[ndarray, "h w"] = np.asarray(pred.depth_map, dtype=np.float64)
        if metric_depth.shape != mv_depth.shape:
            metric_depth = cv2.resize(metric_depth, (mv_depth.shape[1], mv_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid: Bool[ndarray, "h w"] = (
            (np.asarray(geo.depth_confidences[i]) > 0) & np.isfinite(metric_depth) & np.isfinite(mv_depth) & (mv_depth > 1e-6) & (metric_depth > 1e-6)
        )
        metric_per.append(metric_depth)
        valid_per.append(valid)
        if int(valid.sum()) >= 100:
            per_view_scale.append(float(np.median(metric_depth[valid] / mv_depth[valid])))
    if not per_view_scale:
        raise RuntimeError("no view yielded enough confident pixels for the metric scale")
    scale: float = float(np.median(per_view_scale))
    print(f"metric scale s={scale:.4f} (per-view {min(per_view_scale):.3f}..{max(per_view_scale):.3f}, n={len(per_view_scale)})")

    metric_cam_T_world: list[Float64[ndarray, "4 4"]] = []
    for estimated_cam_T_world in cam_T_world_per_view:
        scaled_cam_T_world: Float64[ndarray, "4 4"] = estimated_cam_T_world.copy()
        scaled_cam_T_world[:3, 3] *= scale
        metric_cam_T_world.append(scaled_cam_T_world)
    estimated_camera_centers: Float64[ndarray, "v 3"] = camera_centers(np.stack(metric_cam_T_world))

    # Floor from the metric cloud. G3T already normalizes its predicted gravity to
    # +Z, so the plane fit only fixes the ground height (plus a small tilt
    # correction); a dominant wall or table is rejected by the gravity prior.
    cloud: Float64[ndarray, "n 3"] = np.concatenate(
        [_unproject_world(metric_per[i], intrinsics_per_view[i], metric_cam_T_world[i], valid_per[i], UNPROJECT_STRIDE) for i in range(len(mv))], axis=0
    )
    cloud = cloud[np.isfinite(cloud).all(1)]
    up, ground_point = select_ground_plane(
        cloud, estimated_camera_centers, UP, distance_threshold=GROUND_DISTANCE_M, max_tilt_deg=MAX_GROUND_TILT_DEG
    )
    gravity_rotation: Float64[ndarray, "3 3"] = rotation_matrix_between(up, UP)
    shift: Float64[ndarray, "3"] = np.array([0.0, 0.0, -float((gravity_rotation @ ground_point)[2])])
    print(f"ground plane: up={np.round(up, 3)} ({np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))):.1f} deg from G3T gravity), ground z={ground_point[2]:.3f} m")

    cam_T_world_list: list[Float64[ndarray, "4 4"]] = []
    video_intrinsics_list: list[Float64[ndarray, "3 3"]] = []
    for i in range(len(mv)):
        new_rotation: Float64[ndarray, "3 3"] = metric_cam_T_world[i][:3, :3] @ gravity_rotation.T
        new_translation: Float64[ndarray, "3"] = metric_cam_T_world[i][:3, 3] - new_rotation @ shift
        cam_T_world: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        cam_T_world[:3, :3] = new_rotation
        cam_T_world[:3, 3] = new_translation
        cam_T_world_list.append(cam_T_world)
        # Rescale the estimated K from the geometry pass resolution to the native video resolution.
        video_h, video_w = frames_rgb[i].shape[:2]
        pass_w: int = int(mv[i].pinhole_param.intrinsics.width)
        pass_h: int = int(mv[i].pinhole_param.intrinsics.height)
        scale_wh: Float64[ndarray, "3 3"] = np.diag([video_w / pass_w, video_h / pass_h, 1.0])
        video_intrinsics_list.append(scale_wh @ intrinsics_per_view[i])

    return InitEstimate(cameras=RigCameras(names=names, intrinsics=np.stack(video_intrinsics_list), cam_T_world=np.stack(cam_T_world_list)), metric_scale=scale)


def main(config: InitCalibrationConfig) -> None:
    """Run Stage A and (optionally) register the ``exocalib_init`` layer."""
    context: StageContext = stage_context(config.stage)
    streams: ExoVideoStreams = open_exo_streams(context.dataset, context.segment_id, context.layout.exo_camera_names)
    frames_rgb: list[UInt8[ndarray, "h w 3"]] = [streams.frame_rgb(i, config.frame_index) for i in range(len(streams.names))]
    print(f"segment {context.segment_id}: calibrating from frame {config.frame_index} of {len(frames_rgb)} exo views")

    estimate: InitEstimate = estimate_init_cameras(frames_rgb, streams.names)

    recording, rrd_path = new_layer_recording(context.segment_id, context.segment_dir / f"{INIT_LAYER}.rrd")
    recording.log(exocalib_entity("init"), rr.AnyValues(metric_scale=np.array([estimate.metric_scale])), static=True)
    video_h, video_w = frames_rgb[0].shape[:2]
    log_cameras_layer(estimate.cameras, (video_w, video_h), exocalib_entity("init"), recording, color=VARIANT_COLOR["init"])
    recording.flush(timeout_sec=30.0)
    print(f"wrote {rrd_path}")
    if config.stage.register:
        register_layer(context.dataset, rrd_path, INIT_LAYER)
        print(f"registered layer {INIT_LAYER}")
