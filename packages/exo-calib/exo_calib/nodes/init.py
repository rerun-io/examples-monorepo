"""Stage A: metric, gravity-aligned exo cameras from one synchronized frameset (G3T + MoGe-2 scale + RANSAC floor).

The ``monopriors.apis.calibrate_synced_videos`` metric route, catalog-native. Ground truth is never read here.
Models come from the graph resources ``g3t`` (``MultiviewPredictor``) and ``moge`` (``MoGeV2MetricPredictor``).
"""

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from dagkit.node import Emit, PortType, RunContext, Tick
from jaxtyping import Bool, Float64, UInt8
from monopriors.models.multiview.multiview_model import robust_filter_confidences
from monopriors.models.multiview.multiview_pointcloud import unproject_selected_pixels
from numpy import ndarray
from simplecv.camera_orient_utils import rotation_matrix_between
from simplecv.types.frames import MultiViewFrames
from simplecv.types.rig import RigCameras

from exo_calib.ops.eval import camera_centers
from exo_calib.ops.ground_plane import select_ground_plane

KEEP_TOP_PERCENT: float = 30.0
"""Percentage of high-confidence pixels the multi-view geometry pass retains."""
GROUND_DISTANCE_M: float = 0.06
"""RANSAC inlier distance (meters) for the ground-plane fit."""
MAX_GROUND_TILT_DEG: float = 60.0
"""Largest angle between a candidate floor normal and G3T's gravity before the plane is rejected as a wall or table."""
UNPROJECT_STRIDE: int = 24
"""Pixel stride when unprojecting depth to the ground-fit point cloud."""
UP: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0])
"""G3T normalizes its predicted gravity to +Z; the floor fit refines that direction."""


def estimate_init_cameras(frames_rgb: list[UInt8[ndarray, "h w 3"]], names: tuple[str, ...], g3t: Any, moge: Any) -> RigCameras:
    """Metric, +Z-up exo cameras (ground at z = 0) with intrinsics at the native video resolution."""
    mv = g3t(frames_rgb, preprocessing_mode="pad", center_method="none")
    depth_confidences: list[UInt8[ndarray, "H W"]] = [robust_filter_confidences(p.confidence_mask, keep_top_percent=KEEP_TOP_PERCENT) for p in mv]
    intrinsics_per_view: list[Float64[ndarray, "3 3"]] = [np.asarray(p.pinhole_param.intrinsics.k_matrix, dtype=np.float64) for p in mv]

    # MoGe-2 metric depth -> one global scale (median over views of the per-view median ratio).
    metric_per: list[Float64[ndarray, "h w"]] = []
    valid_per: list[Bool[ndarray, "h w"]] = []
    per_view_scale: list[float] = []
    for i, pred in enumerate(mv):
        metric_depth: Float64[ndarray, "h w"] = moge(frames_rgb[i], K_33=intrinsics_per_view[i].astype(np.float32)).depth_meters.astype(np.float64)
        mv_depth: Float64[ndarray, "h w"] = np.asarray(pred.depth_map, dtype=np.float64)
        if metric_depth.shape != mv_depth.shape:
            metric_depth = cv2.resize(metric_depth, (mv_depth.shape[1], mv_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid: Bool[ndarray, "h w"] = (
            (np.asarray(depth_confidences[i]) > 0) & np.isfinite(metric_depth) & np.isfinite(mv_depth) & (mv_depth > 1e-6) & (metric_depth > 1e-6)
        )
        metric_per.append(metric_depth)
        valid_per.append(valid)
        if int(valid.sum()) >= 100:
            per_view_scale.append(float(np.median(metric_depth[valid] / mv_depth[valid])))
    if not per_view_scale:
        raise RuntimeError("no view yielded enough confident pixels for the metric scale")
    scale: float = float(np.median(per_view_scale))
    print(f"metric scale s={scale:.4f} (per-view {min(per_view_scale):.3f}..{max(per_view_scale):.3f}, n={len(per_view_scale)})")
    metric_cam_T_world: Float64[ndarray, "v 4 4"] = np.stack([np.asarray(p.pinhole_param.extrinsics.cam_T_world, dtype=np.float64) for p in mv])
    metric_cam_T_world[:, :3, 3] *= scale

    # Floor from the metric cloud. G3T already normalizes its predicted gravity to +Z, so the plane fit only fixes the
    # ground height (plus a small tilt correction); a dominant wall or table is rejected by the gravity prior.
    clouds: list[Float64[ndarray, "n 3"]] = []
    for i in range(len(mv)):
        ys, xs = np.where(valid_per[i])
        ys, xs = ys[::UNPROJECT_STRIDE], xs[::UNPROJECT_STRIDE]
        world_T_cam: Float64[ndarray, "4 4"] = np.linalg.inv(metric_cam_T_world[i])
        clouds.append(unproject_selected_pixels(metric_per[i], intrinsics_per_view[i], world_T_cam, ys, xs).astype(np.float64, copy=False))
    cloud: Float64[ndarray, "n 3"] = np.concatenate(clouds, axis=0)
    cloud = cloud[np.isfinite(cloud).all(1)]
    up, ground_point = select_ground_plane(
        cloud, camera_centers(metric_cam_T_world), UP, distance_threshold=GROUND_DISTANCE_M, max_tilt_deg=MAX_GROUND_TILT_DEG
    )
    gravity_rotation: Float64[ndarray, "3 3"] = rotation_matrix_between(up, UP)
    shift: Float64[ndarray, "3"] = np.array([0.0, 0.0, -float((gravity_rotation @ ground_point)[2])])
    print(f"ground plane: up={np.round(up, 3)} ({np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))):.1f} deg from G3T gravity), ground z={ground_point[2]:.3f} m")

    cam_T_world: Float64[ndarray, "v 4 4"] = np.tile(np.eye(4), (len(mv), 1, 1))
    cam_T_world[:, :3, :3] = metric_cam_T_world[:, :3, :3] @ gravity_rotation.T
    cam_T_world[:, :3, 3] = metric_cam_T_world[:, :3, 3] - np.einsum("vij,j->vi", cam_T_world[:, :3, :3], shift)
    # Rescale the estimated K from the geometry pass resolution to the native video resolution.
    video_h, video_w = frames_rgb[0].shape[:2]
    intrinsics: Float64[ndarray, "v 3 3"] = np.stack(
        [np.diag([video_w / p.pinhole_param.intrinsics.width, video_h / p.pinhole_param.intrinsics.height, 1.0]) @ K for p, K in zip(mv, intrinsics_per_view, strict=True)]
    )
    return RigCameras(names, intrinsics, cam_T_world, np.tile(np.array([video_w, video_h], dtype=np.int64), (len(mv), 1)))


class InitNode:
    """Calibrate the rig from one frameset."""

    inputs: Mapping[str, PortType] = {"frames": MultiViewFrames}
    outputs: Mapping[str, PortType] = {"init": RigCameras}

    def reset(self, ctx: RunContext) -> None:
        self.g3t, self.moge = ctx.resources["g3t"], ctx.resources["moge"]

    def step(self, tick: Tick, out: Emit) -> None:
        frames = tick.value
        assert isinstance(frames, MultiViewFrames)
        frames_rgb: list[UInt8[ndarray, "h w 3"]] = list(frames.rgb.permute(0, 2, 3, 1).contiguous().cpu().numpy())
        print(f"Stage A: calibrating from frame {frames.frame_index} of {len(frames_rgb)} exo views")
        out.put("init", tick.t_ns, estimate_init_cameras(frames_rgb, frames.views, self.g3t, self.moge))

    def flush(self, out: Emit) -> None:
        pass
