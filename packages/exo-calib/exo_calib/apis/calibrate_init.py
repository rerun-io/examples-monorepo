"""Stage A: self-contained initial exo-camera calibration.

One synchronized frame from each exo video (read from the catalog, NVDEC) goes
through the gravity-aligned multi-view backend (G3T), MoGe-2 metric depth fixes
the global scale, and a RANSAC ground plane puts gravity on +Z with the floor at
z=0 — the ``monopriors.apis.calibrate_synced_videos`` metric route, catalog-native.
Ground truth is never read here; the result is evaluated against GT elsewhere.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d
import rerun as rr
from jaxtyping import Float64, UInt8
from numpy import ndarray
from simplecv.camera_orient_utils import rotation_matrix_between
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import log_pinhole

from exo_calib.catalog_io import (
    DEFAULT_CATALOG_URL,
    DEFAULT_DATASET_NAME,
    connect_dataset,
    exocalib_entity,
    new_layer_recording,
    only_segment_id,
    register_layer,
)
from exo_calib.video_io import ExoVideoStreams, open_exo_streams


@dataclass
class InitCalibrationConfig:
    """Config for the Stage A initial calibration tool."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """Rerun catalog server URL."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset holding the registered segment."""
    segment_id: str | None = None
    """Segment to calibrate; ``None`` uses the dataset's single segment."""
    frame_index: int = 0
    """Sample index of the synchronized frame used for calibration (rig is static)."""
    model: Literal["g3t", "vggt"] = "g3t"
    """Multi-view backend; ``g3t`` is gravity-aligned."""
    keep_top_percent: float = 30.0
    """Percentage of high-confidence pixels retained by the geometry pass."""
    ground_dist: float = 0.06
    """RANSAC inlier distance (meters) for the ground-plane fit."""
    unproject_stride: int = 24
    """Pixel stride when unprojecting depth to the ground-fit point cloud."""
    output_dir: Path = Path("data/outputs")
    """Directory for ``<segment>/init_cameras.npz`` and the layer RRD."""
    layer_name: str = "exocalib_init"
    """Catalog layer name for the logged initial cameras."""
    application_id: str = "exocalib"
    """Application id of generated layer recordings."""
    register: bool = True
    """Register the layer RRD into the catalog after writing it."""


@dataclass(slots=True)
class InitCameras:
    """Initial (estimated) exo camera parameters — metric, +Z-up, ground at z=0."""

    names: tuple[str, ...]
    """Camera entity names, e.g. ``rig_00/cam_00``."""
    k_v33: Float64[ndarray, "v 3 3"]
    """Estimated pinhole intrinsics at the native video resolution."""
    cam_T_world_v44: Float64[ndarray, "v 4 4"]
    """Estimated world-to-camera transforms."""
    metric_scale: float
    """Global scale applied to the up-to-scale multi-view translations."""

    def save(self, npz_path: Path) -> None:
        """Write the parameters to ``npz_path``."""
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            names=np.array(self.names),
            k_v33=self.k_v33,
            cam_T_world_v44=self.cam_T_world_v44,
            metric_scale=np.float64(self.metric_scale),
        )

    @classmethod
    def load(cls, npz_path: Path) -> "InitCameras":
        """Read parameters written by :meth:`save`."""
        data = np.load(npz_path)
        return cls(
            names=tuple(str(n) for n in data["names"]),
            k_v33=data["k_v33"],
            cam_T_world_v44=data["cam_T_world_v44"],
            metric_scale=float(data["metric_scale"]),
        )


def _unproject_world(
    depth_hw: Float64[ndarray, "h w"],
    k_33: Float64[ndarray, "3 3"],
    cam_T_world_44: Float64[ndarray, "4 4"],
    valid_hw: ndarray,
    stride: int,
) -> Float64[ndarray, "n 3"]:
    """Lift confident depth pixels of one view to world points."""
    from monopriors.models.multiview.multiview_pointcloud import unproject_selected_pixels

    ys, xs = np.where(valid_hw)
    ys, xs = ys[::stride], xs[::stride]
    world_T_cam_44: Float64[ndarray, "4 4"] = np.linalg.inv(cam_T_world_44)
    return unproject_selected_pixels(depth_hw, k_33, world_T_cam_44, ys, xs).astype(np.float64, copy=False)


def estimate_init_cameras(
    frames_rgb: list[UInt8[ndarray, "h w 3"]],
    names: tuple[str, ...],
    config: InitCalibrationConfig,
) -> InitCameras:
    """Estimate metric, gravity-aligned exo cameras from one synchronized frame set.

    Args:
        frames_rgb: One RGB frame per camera, native video resolution.
        names: Camera entity names, index-aligned with ``frames_rgb``.
        config: Stage configuration.

    Returns:
        Estimated cameras with intrinsics rescaled to the native video resolution.
    """
    import cv2
    from monopriors.apis.multiview_geometry import MultiviewGeometryConfig, MultiviewGeometryResult, run_multiview_geometry
    from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
    from monopriors.models.multiview.multiview_predictor import MultiviewPredictor, MultiviewPredictorConfig

    predictor: MultiviewPredictor = MultiviewPredictor(MultiviewPredictorConfig(model_name=config.model))
    geo: MultiviewGeometryResult = run_multiview_geometry(
        rgb_list=frames_rgb,
        multiview_predictor=predictor,
        config=MultiviewGeometryConfig(keep_top_percent=config.keep_top_percent),
    )
    mv = geo.mv_pred_list
    k_per: list[Float64[ndarray, "3 3"]] = [np.asarray(p.pinhole_param.intrinsics.k_matrix, dtype=np.float64) for p in mv]
    w2c_per: list[Float64[ndarray, "4 4"]] = [np.asarray(p.pinhole_param.extrinsics.cam_T_world, dtype=np.float64) for p in mv]

    # MoGe-2 metric depth -> one global scale (median over views of per-view median ratio).
    moge: MoGeV2MetricPredictor = MoGeV2MetricPredictor(device="cuda")
    metric_per: list[Float64[ndarray, "h w"]] = []
    valid_per: list[ndarray] = []
    per_view_scale: list[float] = []
    for i, pred in enumerate(mv):
        metric_hw: Float64[ndarray, "h w"] = moge(frames_rgb[i], K_33=k_per[i].astype(np.float32)).depth_meters.astype(np.float64)
        mv_depth_hw: Float64[ndarray, "h w"] = np.asarray(pred.depth_map, dtype=np.float64)
        if metric_hw.shape != mv_depth_hw.shape:
            metric_hw = cv2.resize(metric_hw, (mv_depth_hw.shape[1], mv_depth_hw.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_hw: ndarray = (
            (np.asarray(geo.depth_confidences[i]) > 0) & np.isfinite(metric_hw) & np.isfinite(mv_depth_hw) & (mv_depth_hw > 1e-6) & (metric_hw > 1e-6)
        )
        metric_per.append(metric_hw)
        valid_per.append(valid_hw)
        if int(valid_hw.sum()) >= 100:
            per_view_scale.append(float(np.median(metric_hw[valid_hw] / mv_depth_hw[valid_hw])))
    if not per_view_scale:
        raise RuntimeError("no view yielded enough confident pixels for the metric scale")
    scale: float = float(np.median(per_view_scale))
    print(f"metric scale s={scale:.4f} (per-view {min(per_view_scale):.3f}..{max(per_view_scale):.3f}, n={len(per_view_scale)})")

    w2c_metric: list[Float64[ndarray, "4 4"]] = []
    for w2c_44 in w2c_per:
        scaled_44: Float64[ndarray, "4 4"] = w2c_44.copy()
        scaled_44[:3, 3] *= scale
        w2c_metric.append(scaled_44)
    cam_centers_v3: Float64[ndarray, "v 3"] = np.array([-m[:3, :3].T @ m[:3, 3] for m in w2c_metric])

    # Ground-plane RANSAC on the metric cloud -> +Z up, ground at z=0.
    cloud_n3: Float64[ndarray, "n 3"] = np.concatenate(
        [_unproject_world(metric_per[i], k_per[i], w2c_metric[i], valid_per[i], config.unproject_stride) for i in range(len(mv))], axis=0
    )
    cloud_n3 = cloud_n3[np.isfinite(cloud_n3).all(1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_n3)
    plane, inliers = pcd.segment_plane(distance_threshold=config.ground_dist, ransac_n=3, num_iterations=2000)
    normal_3: Float64[ndarray, "3"] = np.array(plane[:3], dtype=np.float64)
    normal_3 /= np.linalg.norm(normal_3)
    ground_point_3: Float64[ndarray, "3"] = cloud_n3[inliers].mean(axis=0)
    up_3: Float64[ndarray, "3"] = normal_3 if float(np.dot(normal_3, cam_centers_v3.mean(0) - ground_point_3)) > 0 else -normal_3
    r_gravity_33: Float64[ndarray, "3 3"] = rotation_matrix_between(up_3, np.array([0.0, 0.0, 1.0]))
    shift_3: Float64[ndarray, "3"] = np.array([0.0, 0.0, -float((r_gravity_33 @ ground_point_3)[2])])
    print(f"ground plane: {len(inliers)}/{len(cloud_n3)} inliers, up={np.round(up_3, 3)}")

    cam_T_world_list: list[Float64[ndarray, "4 4"]] = []
    k_video_list: list[Float64[ndarray, "3 3"]] = []
    for i in range(len(mv)):
        r_new_33: Float64[ndarray, "3 3"] = w2c_metric[i][:3, :3] @ r_gravity_33.T
        t_new_3: Float64[ndarray, "3"] = w2c_metric[i][:3, 3] - r_new_33 @ shift_3
        cam_T_world_44: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        cam_T_world_44[:3, :3] = r_new_33
        cam_T_world_44[:3, 3] = t_new_3
        cam_T_world_list.append(cam_T_world_44)
        # Rescale the estimated K from the geometry pass resolution to the native video resolution.
        video_h, video_w = frames_rgb[i].shape[:2]
        pass_w: int = int(mv[i].pinhole_param.intrinsics.width)
        pass_h: int = int(mv[i].pinhole_param.intrinsics.height)
        scale_wh: Float64[ndarray, "3 3"] = np.diag([video_w / pass_w, video_h / pass_h, 1.0])
        k_video_list.append(scale_wh @ k_per[i])

    return InitCameras(names=names, k_v33=np.stack(k_video_list), cam_T_world_v44=np.stack(cam_T_world_list), metric_scale=scale)


def log_cameras_layer(
    cameras_k_v33: Float64[ndarray, "v 3 3"],
    cameras_cam_T_world_v44: Float64[ndarray, "v 4 4"],
    names: tuple[str, ...],
    resolution_wh: tuple[int, int],
    entity_prefix: str,
    recording: rr.RecordingStream,
    image_plane_distance: float = 0.1,
) -> None:
    """Log a set of cameras as static pinhole frusta under ``entity_prefix``.

    ``image_plane_distance`` defaults to the base layer's GT frustum size so
    estimated and GT frusta compare visually.
    """
    for i, name in enumerate(names):
        intrinsics: Intrinsics = Intrinsics(
            camera_conventions="RDF", k_matrix=cameras_k_v33[i], width=resolution_wh[0], height=resolution_wh[1]
        )
        extrinsics: Extrinsics = Extrinsics(cam_R_world=cameras_cam_T_world_v44[i][:3, :3], cam_t_world=cameras_cam_T_world_v44[i][:3, 3])
        pinhole: PinholeParameters = PinholeParameters(name=name.replace("/", "_"), intrinsics=intrinsics, extrinsics=extrinsics)
        log_pinhole(pinhole, Path(f"{entity_prefix}/{name}"), image_plane_distance=image_plane_distance, static=True, recording=recording)


def main(config: InitCalibrationConfig) -> None:
    """Run Stage A and (optionally) register the ``exocalib_init`` layer."""
    dataset = connect_dataset(config.catalog_url, config.dataset_name)
    segment_id: str = config.segment_id or only_segment_id(dataset)
    streams: ExoVideoStreams = open_exo_streams(dataset, segment_id)
    frames_rgb: list[UInt8[ndarray, "h w 3"]] = [streams.frame_rgb_hw3(i, config.frame_index) for i in range(len(streams.names))]
    print(f"segment {segment_id}: calibrating from frame {config.frame_index} of {len(frames_rgb)} exo views")

    init: InitCameras = estimate_init_cameras(frames_rgb, streams.names, config)
    npz_path: Path = config.output_dir / segment_id / "init_cameras.npz"
    init.save(npz_path)
    print(f"wrote {npz_path}")

    recording, rrd_path = new_layer_recording(config.application_id, segment_id, config.output_dir / segment_id / f"{config.layer_name}.rrd")
    video_h, video_w = frames_rgb[0].shape[:2]
    log_cameras_layer(init.k_v33, init.cam_T_world_v44, init.names, (video_w, video_h), exocalib_entity("init"), recording)
    recording.flush(timeout_sec=30.0)
    print(f"wrote {rrd_path}")
    if config.register:
        register_layer(dataset, rrd_path, config.layer_name)
        print(f"registered layer {config.layer_name}")
