"""Calibrate a directory of time-synced (but UNCALIBRATED) videos and write the
mamma ``meta/`` NPZ calibration so the directory becomes a drop-in mamma capture.

Pipeline (single synchronized timestamp; the rig is static):
  1. Multi-view geometry      -> per-camera intrinsics K + extrinsics (up-to-scale).
  2. MoGe-v2 metric depth     -> a single global scale s = median over views of
     median(metric_depth / multiview_depth) on confident pixels -> metric rig.
  3. Ground-plane gravity     -> unproject the metric depth to a scene point
     cloud, RANSAC the dominant plane (the floor); its normal is gravity (sign
     from "cameras sit above the ground") and its offset puts the ground at z=0.
     This is POSE-INDEPENDENT (it uses the scene, not the subject), so it works
     even when the subject is lying down / crouching / mid-jump — unlike a
     body-up heuristic, and unlike VGGT's own "mean camera up-vector" which is
     unreliable for handheld/phone rigs.
  4. Write ``<videos_dir>/meta/{<cam>.npz, global.npz}`` (metric, Z-up, ground at
     z=0) in the mamma ma_cap contract. The mamma pipeline then runs unchanged
     via ``load_mamma_sequence(<videos_dir>)``.

Usage:
    pixi run -e monoprior python tools/demos/calibrate_synced_videos.py --videos-dir DIR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import open3d as o3d
from jaxtyping import Float64
from numpy import ndarray
from simplecv.camera_orient_utils import rotation_matrix_between
from simplecv.video_io import MultiVideoReader

from monopriors.apis.multiview_geometry import (
    MultiviewGeometryConfig,
    MultiviewGeometryResult,
    run_multiview_geometry,
)
from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
from monopriors.models.multiview.multiview_pointcloud import unproject_selected_pixels
from monopriors.models.multiview.multiview_predictor import MultiviewPredictor, MultiviewPredictorConfig

_VIDEO_EXTS: tuple[str, ...] = (".mp4", ".mov", ".MP4", ".MOV")
_VIDEO_SUBDIRS: tuple[str, ...] = ("videos_light", "videos", "videos_crf24")


@dataclass
class CalibConfig:
    videos_dir: Path
    """Capture dir holding time-synced videos (one per camera, no calibration).
    Videos may sit directly inside, or in a ``videos_light/`` (etc.) subdir."""
    ts_idx: int = 0
    """Frame index used for the single-shot multiview calibration (rig is static)."""
    predictor_config: MultiviewPredictorConfig = field(default_factory=MultiviewPredictorConfig)
    """Construction settings for the selected multi-view backend."""
    geometry_config: MultiviewGeometryConfig = field(default_factory=MultiviewGeometryConfig)
    """Per-run multi-view preprocessing, centering, and confidence settings."""
    gravity: Literal["ground", "model"] = "ground"
    """How to set the world up-axis. ``ground``: RANSAC the reconstructed floor
    (pose-independent, recommended). ``model``: trust the selected backend's +Z orientation."""
    ground_dist: float = 0.06
    """RANSAC inlier distance (meters) for the ground-plane fit."""
    stride: int = 24
    """Pixel stride when unprojecting depth to the ground-fit point cloud."""
    force: bool = False
    """Recompute even if ``meta/`` already exists."""


def _unproject(depth: Float64[ndarray, "h w"], k_33: Float64[ndarray, "3 3"], w2c: Float64[ndarray, "4 4"], valid: ndarray, stride: int) -> Float64[ndarray, "n 3"]:
    """Lift confident depth pixels of one view to world points (metric w2c)."""
    r_cw: Float64[ndarray, "3 3"] = w2c[:3, :3]
    ys, xs = np.where(valid)
    ys, xs = ys[::stride], xs[::stride]
    world_T_cam: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    world_T_cam[:3, :3] = r_cw.T
    world_T_cam[:3, 3] = -r_cw.T @ w2c[:3, 3]
    return unproject_selected_pixels(depth, k_33, world_T_cam, ys, xs).astype(np.float64, copy=False)


def main(config: CalibConfig) -> int:
    videos_dir: Path = config.videos_dir
    video_root: Path = next((videos_dir / d for d in _VIDEO_SUBDIRS if (videos_dir / d).is_dir()), videos_dir)
    video_paths: list[Path] = sorted(p for p in video_root.iterdir() if p.suffix in _VIDEO_EXTS)
    if not video_paths:
        raise FileNotFoundError(f"no videos ({_VIDEO_EXTS}) under {video_root}")
    cam_names: list[str] = [p.stem for p in video_paths]
    meta_dir: Path = videos_dir / "meta"
    if meta_dir.is_dir() and not config.force:
        print(f"meta/ already exists at {meta_dir}; pass --force to recompute. Skipping.")
        return 0
    print(f"cameras ({len(cam_names)}): {cam_names}")

    reader: MultiVideoReader = MultiVideoReader(video_paths)
    fps: float = float(reader.video_readers[0].fps)
    frame_count: int = len(reader)
    rgb_list = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in reader[config.ts_idx]]
    print(f"loaded synced frame {config.ts_idx}: {len(rgb_list)} views @ {rgb_list[0].shape[1]}x{rgb_list[0].shape[0]}, fps={fps}, frames={frame_count}")

    # 1. Multi-view rig (up-to-scale).
    multiview_predictor: MultiviewPredictor = MultiviewPredictor(config.predictor_config)
    geo: MultiviewGeometryResult = run_multiview_geometry(
        rgb_list=rgb_list,
        multiview_predictor=multiview_predictor,
        config=config.geometry_config,
    )
    mv = geo.mv_pred_list
    k_per: list[Float64[ndarray, "3 3"]] = [np.asarray(p.pinhole_param.intrinsics.k_matrix, dtype=np.float64) for p in mv]
    w2c_per: list[Float64[ndarray, "4 4"]] = [np.asarray(p.pinhole_param.extrinsics.cam_T_world, dtype=np.float64) for p in mv]

    # 2. MoGe-v2 metric scale + keep per-view metric depth (for the ground fit).
    moge: MoGeV2MetricPredictor = MoGeV2MetricPredictor(device=config.predictor_config.device)
    metric_per: list[ndarray] = []
    valid_per: list[ndarray] = []
    per_view_s: list[float] = []
    for i, pred in enumerate(mv):
        metric: Float64[ndarray, "h w"] = moge(rgb_list[i], K_33=k_per[i].astype(np.float32)).depth_meters.astype(np.float64)
        multiview_depth: Float64[ndarray, "h w"] = np.asarray(pred.depth_map, dtype=np.float64)
        if metric.shape != multiview_depth.shape:
            metric = cv2.resize(
                metric,
                (multiview_depth.shape[1], multiview_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        valid: ndarray = (
            (np.asarray(geo.depth_confidences[i]) > 0)
            & np.isfinite(metric)
            & np.isfinite(multiview_depth)
            & (multiview_depth > 1e-6)
            & (metric > 1e-6)
        )
        metric_per.append(metric)
        valid_per.append(valid)
        if int(valid.sum()) >= 100:
            per_view_s.append(float(np.median(metric[valid] / multiview_depth[valid])))
            print(f"  {cam_names[i]}: scale={per_view_s[-1]:.4f}  ({int(valid.sum())} px)")
    if not per_view_s:
        raise RuntimeError("no view yielded enough confident pixels for metric scale")
    s: float = float(np.median(per_view_s))
    print(f"global metric scale s={s:.4f}  (per-view {min(per_view_s):.3f}..{max(per_view_s):.3f}, n={len(per_view_s)})")

    # metric extrinsics (translation scaled).
    w2c_metric: list[Float64[ndarray, "4 4"]] = []
    for w2c in w2c_per:
        m = w2c.copy()
        m[:3, 3] *= s
        w2c_metric.append(m)
    cam_centers: Float64[ndarray, "c 3"] = np.array([-m[:3, :3].T @ m[:3, 3] for m in w2c_metric])

    # 3. Gravity: RANSAC the reconstructed ground plane (pose-independent).
    r_grav: Float64[ndarray, "3 3"] = np.eye(3)
    shift: Float64[ndarray, "3"] = np.zeros(3)
    if config.gravity == "ground":
        cloud = np.concatenate([_unproject(metric_per[i], k_per[i], w2c_metric[i], valid_per[i], config.stride) for i in range(len(mv))], axis=0)
        cloud = cloud[np.isfinite(cloud).all(1)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        plane, inliers = pcd.segment_plane(distance_threshold=config.ground_dist, ransac_n=3, num_iterations=2000)
        normal: Float64[ndarray, "3"] = np.array(plane[:3], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        ground_pt: Float64[ndarray, "3"] = cloud[inliers].mean(axis=0)
        up: Float64[ndarray, "3"] = normal if float(np.dot(normal, cam_centers.mean(0) - ground_pt)) > 0 else -normal
        r_grav = rotation_matrix_between(up, np.array([0.0, 0.0, 1.0]))
        shift = np.array([0.0, 0, -float((r_grav @ ground_pt)[2])])
        cam_h_after = (r_grav @ cam_centers.T).T[:, 2] + shift[2]
        print(f"ground plane: {len(inliers)}/{len(cloud)} inliers, up={np.round(up, 3)}, cam heights above ground (m)={np.round(cam_h_after, 2)}")
    else:
        print(f"gravity=model: trusting {config.predictor_config.model_name.upper()}'s +Z orientation")

    # 4. Write the mamma meta/ contract: world_to_cam in metric, Z-up, ground at z=0.
    meta_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(mv)):
        r_c_new: Float64[ndarray, "3 3"] = w2c_metric[i][:3, :3] @ r_grav.T
        t_c_new: Float64[ndarray, "3"] = w2c_metric[i][:3, 3] - r_c_new @ shift
        w2c_out: Float64[ndarray, "4 4"] = np.eye(4)
        w2c_out[:3, :3] = r_c_new
        w2c_out[:3, 3] = t_c_new
        np.savez(
            meta_dir / f"{cam_names[i]}.npz",
            cam_name=cam_names[i],
            cam_int=k_per[i],
            cam_ext=w2c_out,
            cam_img_w=int(mv[i].pinhole_param.intrinsics.width),
            cam_img_h=int(mv[i].pinhole_param.intrinsics.height),
        )
    np.savez(meta_dir / "global.npz", seq_name=videos_dir.name, fps=fps, frame_start=0, frame_end=frame_count)
    print(f"wrote {len(mv)} camera NPZs + global.npz to {meta_dir}  (seq={videos_dir.name})")
    return 0
