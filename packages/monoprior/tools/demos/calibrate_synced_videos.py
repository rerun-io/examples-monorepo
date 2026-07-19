"""Calibrate a directory of time-synced (but UNCALIBRATED) videos and write the
mamma ``meta/`` NPZ calibration so the directory becomes a drop-in mamma capture.

Pipeline (single synchronized timestamp; the rig is static):
  1. VGGT multiview geometry -> per-camera intrinsics K + extrinsics (up-to-scale).
  2. MoGe-v2 metric depth     -> a single global scale s = median over views of
     median(metric_depth / vggt_depth) on confident pixels -> metric rig.
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

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import open3d as o3d
import tyro
from jaxtyping import Float64
from numpy import ndarray
from simplecv.video_io import MultiVideoReader

from monopriors.apis.multiview_geometry import (
    MultiviewGeometryConfig,
    MultiviewGeometryResult,
    run_multiview_geometry,
)
from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
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
    device: Literal["cuda", "cpu"] = "cuda"
    """Compute device for VGGT + MoGe-v2."""
    preprocessing_mode: Literal["crop", "pad"] = "pad"
    """VGGT image preprocessing."""
    gravity: Literal["ground", "vggt"] = "ground"
    """How to set the world up-axis. ``ground``: RANSAC the reconstructed floor
    (pose-independent, recommended). ``vggt``: trust VGGT's own +Z orientation."""
    ground_dist: float = 0.06
    """RANSAC inlier distance (meters) for the ground-plane fit."""
    stride: int = 24
    """Pixel stride when unprojecting depth to the ground-fit point cloud."""
    force: bool = False
    """Recompute even if ``meta/`` already exists."""


def _rot_between(a: Float64[ndarray, "3"], b: Float64[ndarray, "3"]) -> Float64[ndarray, "3 3"]:
    """Rotation R with R @ a aligned to b (both unit-normalized)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:  # antiparallel: 180deg about any axis perpendicular to a
        ax = np.cross(a, np.array([1.0, 0, 0]))
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, np.array([0, 1.0, 0]))
        ax /= np.linalg.norm(ax)
        k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2 * (k @ k)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + (k @ k) * (1.0 / (1.0 + c))


def _unproject(depth: Float64[ndarray, "h w"], k_33: Float64[ndarray, "3 3"], w2c: Float64[ndarray, "4 4"], valid: ndarray, stride: int) -> Float64[ndarray, "n 3"]:
    """Lift confident depth pixels of one view to world points (metric w2c)."""
    r_cw: Float64[ndarray, "3 3"] = w2c[:3, :3]
    cam_center: Float64[ndarray, "3"] = -r_cw.T @ w2c[:3, 3]
    ys, xs = np.where(valid)
    ys, xs = ys[::stride], xs[::stride]
    d = depth[ys, xs]
    pix: Float64[ndarray, "3 n"] = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    cam: Float64[ndarray, "3 n"] = (np.linalg.inv(k_33) @ pix) * d
    return (r_cw.T @ cam + cam_center[:, None]).T


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

    # 1. VGGT rig (up-to-scale).
    multiview_predictor: MultiviewPredictor = MultiviewPredictor(
        MultiviewPredictorConfig(device=config.device, preprocessing_mode=config.preprocessing_mode)
    )
    geo: MultiviewGeometryResult = run_multiview_geometry(
        rgb_list=rgb_list,
        multiview_predictor=multiview_predictor,
        config=MultiviewGeometryConfig(),
    )
    mv = geo.mv_pred_list
    k_per: list[Float64[ndarray, "3 3"]] = [np.asarray(p.pinhole_param.intrinsics.k_matrix, dtype=np.float64) for p in mv]
    w2c_per: list[Float64[ndarray, "4 4"]] = [np.asarray(p.pinhole_param.extrinsics.cam_T_world, dtype=np.float64) for p in mv]

    # 2. MoGe-v2 metric scale + keep per-view metric depth (for the ground fit).
    moge: MoGeV2MetricPredictor = MoGeV2MetricPredictor(device=config.device)
    metric_per: list[ndarray] = []
    valid_per: list[ndarray] = []
    per_view_s: list[float] = []
    for i, pred in enumerate(mv):
        metric: Float64[ndarray, "h w"] = moge(rgb_list[i], K_33=k_per[i].astype(np.float32)).depth_meters.astype(np.float64)
        vggt_d: Float64[ndarray, "h w"] = np.asarray(pred.depth_map, dtype=np.float64)
        if metric.shape != vggt_d.shape:
            metric = cv2.resize(metric, (vggt_d.shape[1], vggt_d.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid: ndarray = (np.asarray(geo.depth_confidences[i]) > 0) & np.isfinite(metric) & np.isfinite(vggt_d) & (vggt_d > 1e-6) & (metric > 1e-6)
        metric_per.append(metric)
        valid_per.append(valid)
        if int(valid.sum()) >= 100:
            per_view_s.append(float(np.median(metric[valid] / vggt_d[valid])))
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
        r_grav = _rot_between(up, np.array([0.0, 0, 1]))
        shift = np.array([0.0, 0, -float((r_grav @ ground_pt)[2])])
        cam_h_after = (r_grav @ cam_centers.T).T[:, 2] + shift[2]
        print(f"ground plane: {len(inliers)}/{len(cloud)} inliers, up={np.round(up, 3)}, cam heights above ground (m)={np.round(cam_h_after, 2)}")
    else:
        print("gravity=vggt: trusting VGGT's own +Z orientation")

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


if __name__ == "__main__":
    sys.exit(main(tyro.cli(CalibConfig)))
