"""Refine a *coarse* uncalibrated-recovery rig with human-keypoint bundle adjustment.

This is stage B of the calibration-free path and runs ONLY for captures that
have no ground-truth calibration (or when you explicitly want to re-derive it):

    stage A (monoprior env): tools/demos/calibrate_synced_videos.py
        VGGT + MoGe-v2 metric scale + ground-plane gravity  ->  coarse meta/
    stage B (this tool, mamma env):
        coarse meta/  ->  MammaNet 2D landmarks over N frames  ->  confidence-
        weighted keypoint bundle adjustment  ->  refined meta/

The normal calibrated path is untouched: a capture that already ships a GT
``meta/`` is run directly by the quality pipeline and never sees this tool.

The 2D landmarks MammaNet emits are independent of the camera calibration (it
only sees person crops), so we can collect them once with the coarse rig and use
them to refine that rig — exactly the "VGGT coarse init, then refine with human
keypoint BA" recipe from ``liris-xr/kineo`` issue #7. The BA is camera-only
(per-tick fitting is skipped here), so it is a cheap one-shot calibration cost
that does not touch the streaming pipeline's per-tick speed.

Usage:
    pixi run -e mamma python tools/refine_calibration_ba.py --data-dir DIR \
        --trt-engine .trt_cache/<plan> [--out-meta-dir DIR]
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import tyro
from jaxtyping import Float32, Float64
from numpy import ndarray

from mamma.calibration.keypoint_bundle_adjust import BundleAdjustConfig, BundleAdjustResult, bundle_adjust_cameras
from mamma.calibration.npz_contract import CameraCalibration, dump_camera_npz
from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import ResultCollector, StreamingPipeline, build_streaming_pipeline
from mamma.engine.presets import get_preset
from mamma.fitting.triangulation import triangulate_points
from mamma.landmarks.estimator import CameraLandmarks


@dataclass
class RefineConfig:
    """Bundle-adjustment calibration-refinement settings."""

    data_dir: Path = Path("data/inputs/uncalib/long_jump")
    """Capture dir with a *coarse* ``meta/`` (from calibrate_synced_videos.py) + videos."""
    out_meta_dir: Path | None = None
    """Where to write the refined per-camera NPZs; default = ``<data_dir>/meta``
    in place (the coarse meta is backed up to ``meta_coarse/`` first)."""
    mammanet_weights: Path = Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors")
    """Converted MammaNet weights."""
    trt_engine: Path | None = None
    """Optional MammaNet TensorRT engine plan (single-person 4-cam = 4 crops)."""
    resize_hw: tuple[int, int] = (720, 1280)
    """Engine resolution the 2D landmarks (and the BA) operate in."""
    max_calib_frames: int | None = 160
    """Cap track+landmark collection to the first N frames (calibration cost is
    fixed regardless of clip length); None = whole clip."""
    n_kp_samples: int = 4000
    """Multi-view correspondences fed to the BA (subsampled from all frames x
    landmarks); -1 keeps every candidate."""
    min_kp_score: float = 0.5
    """A landmark must exceed this visibility in >= ``min_views`` cameras to be a
    bundle-adjustment correspondence."""
    min_views: int = 2
    """Minimum cameras observing a correspondence."""
    optimize_focal: bool = True
    """Refine per-camera focal length."""
    optimize_distortion: bool = False
    """Refine Brown-Conrady distortion (off: iPhone footage ~ pinhole)."""
    anchor_rot_weight: float = 50.0
    """Drift penalty on rotation (rad^2) from the VGGT init."""
    anchor_trans_weight: float = 50.0
    """Drift penalty on translation (m^2) from the VGGT init."""
    anchor_focal_weight: float = 10.0
    """Drift penalty on fractional focal change from the VGGT/MoGe init."""
    n_iters: int = 40
    """LBFGS iteration budget."""
    reground: bool = True
    """Re-derive gravity + ground (z=0) from MammaNet floor-contact landmarks
    triangulated with the BA-refined rig. More reliable than stage A's scene-cloud
    RANSAC (the feet on the floor are a direct ground signal); guarded so it only
    overrides when it finds enough contacts and yields plausible camera heights."""
    floor_contact_thresh: float = 0.5
    """Min mean (over cameras) floor-contact probability to treat a triangulated
    landmark as a ground point."""
    device: str = "cuda"
    """Compute device."""
    seed: int = 0
    """Correspondence-subsampling seed (determinism)."""


def collect_correspondences(
    landmarks_per_tick: list[list[CameraLandmarks] | None],
    n_cams: int,
    n_landmarks: int,
) -> tuple[Float32[ndarray, "c s 3"], Float32[ndarray, "c s"]]:
    """Flatten collected per-tick landmarks into multi-view correspondences.

    Each ``(tick, person, landmark)`` becomes one correspondence ``s`` with a 2D
    observation ``[x, y, log_variance]`` and a visibility per camera (zeros where
    that camera did not see the person). MammaNet's landmark index ``j`` is the
    same SMPL-X surface sample in every camera, so index ``j`` across cameras is a
    valid correspondence without any cross-view matching.
    """
    xy_samples: list[Float32[ndarray, "j c 3"]] = []
    vis_samples: list[Float32[ndarray, "j c"]] = []
    for tick in landmarks_per_tick:
        if tick is None:
            continue
        obj_ids: set[int] = set().union(*[set(cam.keys()) for cam in tick]) if tick else set()
        for oid in sorted(obj_ids):
            if sum(oid in cam for cam in tick) < 2:
                continue
            xy: Float32[ndarray, "c j 3"] = np.zeros((n_cams, n_landmarks, 3), dtype=np.float32)
            vis: Float32[ndarray, "c j"] = np.zeros((n_cams, n_landmarks), dtype=np.float32)
            for c, cam in enumerate(tick):
                if oid in cam:
                    xy[c] = cam[oid].joints2d.detach().cpu().numpy()
                    vis[c] = cam[oid].visibility.detach().cpu().numpy()
            xy_samples.append(xy.transpose(1, 0, 2))  # (j, c, 3)
            vis_samples.append(vis.transpose(1, 0))  # (j, c)
    if not xy_samples:
        return np.zeros((n_cams, 0, 3), dtype=np.float32), np.zeros((n_cams, 0), dtype=np.float32)
    kps2d: Float32[ndarray, "c s 3"] = np.concatenate(xy_samples, axis=0).transpose(1, 0, 2)
    vis_all: Float32[ndarray, "c s"] = np.concatenate(vis_samples, axis=0).transpose(1, 0)
    return kps2d, vis_all


def _rot_up_to_z(up: Float64[ndarray, "3"]) -> Float64[ndarray, "3 3"]:
    """Rotation mapping unit vector ``up`` to +Z (Rodrigues; handles (anti)parallel)."""
    up = up / (np.linalg.norm(up) + 1e-12)
    z: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0])
    v: Float64[ndarray, "3"] = np.cross(up, z)
    c: float = float(np.dot(up, z))
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])  # 180deg flip about X
    k: Float64[ndarray, "3 3"] = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + (k @ k) * (1.0 / (1.0 + c))


def reground_from_floor_contact(
    landmarks_per_tick: list[list[CameraLandmarks] | None],
    k_engine: Float32[torch.Tensor, "c 3 3"],
    w2c_refined: Float32[torch.Tensor, "c 4 4"],
    thresh: float,
    device: str,
) -> tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3"]] | None:
    """Fit gravity + ground from triangulated floor-contact landmarks (RANSAC).

    Returns ``(R_grav, shift)`` mapping the BA world to a Z-up, ground-at-z=0 world
    (``x_new = R_grav @ x_old + shift``), or ``None`` if there is no reliable floor
    signal. The BA cameras are then re-expressed in that world.
    """
    n_cams: int = int(k_engine.shape[0])
    pts: list[Float64[ndarray, "n 3"]] = []
    for tick in landmarks_per_tick:
        if tick is None:
            continue
        obj_ids: set[int] = set().union(*[set(cam.keys()) for cam in tick]) if tick else set()
        for oid in sorted(obj_ids):
            present: list[int] = [c for c in range(n_cams) if c < len(tick) and oid in tick[c]]
            if len(present) < 2:
                continue
            j2d: list[Float32[torch.Tensor, "j 3"]] = []
            vis: list[Float32[torch.Tensor, "j"]] = []
            fc: list[Float32[torch.Tensor, "j"]] = []
            for c in range(n_cams):
                if oid in tick[c]:
                    lr = tick[c][oid]
                    j2d.append(lr.joints2d.to(device))
                    vis.append(lr.visibility.to(device))
                    fc.append(lr.floor_contact.to(device))
                else:
                    j2d.append(torch.zeros_like(tick[present[0]][oid].joints2d, device=device))
                    vis.append(torch.zeros_like(tick[present[0]][oid].visibility, device=device))
                    fc.append(torch.zeros_like(tick[present[0]][oid].floor_contact, device=device))
            pts3d, valid = triangulate_points(
                j2d, [k_engine[c] for c in range(n_cams)], [w2c_refined[c] for c in range(n_cams)], vis, vis_thresh=0.3
            )
            fc_mean: Float32[torch.Tensor, "j"] = torch.stack(fc, 0).mean(0)
            keep = valid & (fc_mean > thresh)
            if int(keep.sum()) > 0:
                pts.append(pts3d[keep].detach().cpu().numpy().astype(np.float64))
    if not pts:
        return None
    cloud: Float64[ndarray, "n 3"] = np.concatenate(pts, axis=0)
    cloud = cloud[np.isfinite(cloud).all(1)]
    if cloud.shape[0] < 50:
        return None

    # RANSAC plane fit (numpy): sample 3 points, score inliers within 5cm.
    rng = np.random.default_rng(0)
    best_inl: ndarray = np.array([], dtype=int)
    for _ in range(400):
        idx = rng.choice(cloud.shape[0], 3, replace=False)
        p0, p1, p2 = cloud[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        dist = np.abs((cloud - p0) @ nrm)
        inl = np.where(dist < 0.05)[0]
        if inl.size > best_inl.size:
            best_inl = inl
    if best_inl.size < 50:
        return None
    # refit normal on inliers (smallest-eigvec of centered covariance).
    inl_pts = cloud[best_inl]
    centroid = inl_pts.mean(0)
    _, _, Vt = np.linalg.svd(inl_pts - centroid)
    normal: Float64[ndarray, "3"] = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12)
    cam_centers: Float64[ndarray, "c 3"] = np.array(
        [(-w2c_refined[c, :3, :3].T @ w2c_refined[c, :3, 3]).cpu().numpy() for c in range(n_cams)], dtype=np.float64
    )
    up: Float64[ndarray, "3"] = normal if float(np.dot(normal, cam_centers.mean(0) - centroid)) > 0 else -normal
    r_grav: Float64[ndarray, "3 3"] = _rot_up_to_z(up)
    shift: Float64[ndarray, "3"] = np.array([0.0, 0.0, -float((r_grav @ centroid)[2])])
    cam_h: Float64[ndarray, "c"] = (r_grav @ cam_centers.T).T[:, 2] + shift[2]
    if not (np.all(cam_h > 0.2) and np.all(cam_h < 5.0)):
        print(f"  reground: implausible camera heights {np.round(cam_h, 2)} — keeping stage-A ground")
        return None
    print(f"  reground: {best_inl.size}/{cloud.shape[0]} floor inliers, cam heights (m)={np.round(cam_h, 2)}")
    return r_grav, shift


def main(config: RefineConfig) -> int:
    # The pipeline's StreamLogger logs to (and sends a blueprint to) the global
    # Rerun recording. This is a headless calibration pass with no viewer: init an
    # in-memory recording with spawn=False so nothing is rendered or streamed
    # (no sink => never blocks), bounded by the capped calibration frame count.
    import rerun as rr

    rr.init("mamma_ba_calib", spawn=False)

    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)
    n_cams: int = len(sequence.cameras)
    native_w: int = sequence.cameras[0].width
    native_h: int = sequence.cameras[0].height
    print(f"capture {sequence.name}: {n_cams} cams @ {native_w}x{native_h}, {sequence.frame_count} frames")

    # --- Stage B.1: collect MammaNet 2D landmarks with the coarse rig (no fit). ---
    preset = get_preset("quality")
    collector: ResultCollector = ResultCollector()
    pipeline: StreamingPipeline = build_streaming_pipeline(
        sequence,
        resize_hw=config.resize_hw,
        device=config.device,
        tracker_config=preset.tracker,
        fitter_config=None,  # camera-only BA: skip the per-tick SMPL-X fit
        mammanet_weights=config.mammanet_weights,
        trt_engine=config.trt_engine,
        collector=collector,
        hires_crops=preset.hires_crops,
    )
    try:
        stats = pipeline.run(chunk_size=32, max_frames=config.max_calib_frames)
    finally:
        pipeline.close()
    n_ticks: int = sum(1 for lm in collector.landmarks if lm is not None)
    print(f"collected landmarks on {n_ticks} ticks in {stats.elapsed_s:.1f}s ({stats.ticks_per_s:.1f} ticks/s)")

    # MammaNet emits a fixed 512-landmark grid; read the exact count from the first result.
    n_landmarks: int = 512
    for lm in collector.landmarks:
        if lm:
            for cam in lm:
                if cam:
                    n_landmarks = int(next(iter(cam.values())).joints2d.shape[0])
                    break
            break

    kps2d_np, vis_np = collect_correspondences(collector.landmarks, n_cams, n_landmarks)
    n_total: int = kps2d_np.shape[1]
    if n_total == 0:
        print("ERROR: no multi-view landmark correspondences collected; cannot refine.")
        return 1

    # --- Stage B.2: confidence-aware subsample of candidate correspondences. ---
    vis_t: Float32[torch.Tensor, "c s"] = torch.from_numpy(vis_np)
    in_img: Float32[torch.Tensor, "c s"] = (torch.from_numpy(kps2d_np[..., :2]).abs().sum(-1) > 0).float()
    candidate: torch.Tensor = ((vis_t > config.min_kp_score) & (in_img > 0)).sum(0) >= config.min_views
    cand_idx: torch.Tensor = torch.where(candidate)[0]
    if cand_idx.numel() == 0:
        print("ERROR: no correspondence met the confidence/visibility threshold.")
        return 1
    gen: torch.Generator = torch.Generator().manual_seed(config.seed)
    if config.n_kp_samples != -1 and cand_idx.numel() > config.n_kp_samples:
        pick: torch.Tensor = cand_idx[torch.randperm(cand_idx.numel(), generator=gen)[: config.n_kp_samples]]
    else:
        pick = cand_idx
    print(f"correspondences: {n_total} total -> {cand_idx.numel()} candidates -> {pick.numel()} used for BA")

    kps2d: Float32[torch.Tensor, "c s 3"] = torch.from_numpy(kps2d_np)[:, pick].to(config.device)
    vis: Float32[torch.Tensor, "c s"] = vis_t[:, pick].to(config.device)

    # --- Stage B.3: init 3D by weighted-DLT triangulation with the coarse rig. ---
    scaled: list[CameraCalibration] = sequence.scaled_cameras(config.resize_hw)
    k_engine: Float32[torch.Tensor, "c 3 3"] = torch.stack(
        [torch.tensor(c.k_matrix, dtype=torch.float32) for c in scaled]
    ).to(config.device)
    w2c: Float32[torch.Tensor, "c 4 4"] = torch.stack(
        [torch.tensor(c.world_to_cam, dtype=torch.float32) for c in scaled]
    ).to(config.device)
    pts3d_init, valid_init = triangulate_points(
        [kps2d[c] for c in range(n_cams)],
        [k_engine[c] for c in range(n_cams)],
        [w2c[c] for c in range(n_cams)],
        [vis[c] for c in range(n_cams)],
        vis_thresh=config.min_kp_score,
    )
    print(f"triangulated init: {int(valid_init.sum())}/{pts3d_init.shape[0]} valid points")

    # --- Stage B.4: bundle adjustment (cam 0 locked; anchored to the VGGT init). ---
    ba_cfg: BundleAdjustConfig = BundleAdjustConfig(
        optimize_focal=config.optimize_focal,
        optimize_distortion=config.optimize_distortion,
        anchor_rot_weight=config.anchor_rot_weight,
        anchor_trans_weight=config.anchor_trans_weight,
        anchor_focal_weight=config.anchor_focal_weight,
        n_iters=config.n_iters,
        min_obs_views=config.min_views,
    )
    result: BundleAdjustResult = bundle_adjust_cameras(k_engine, w2c, kps2d, vis, pts3d_init, ba_cfg)
    print(
        f"BA: {result.n_samples} samples, reprojection RMSE {result.rmse_px_before:.2f}px -> "
        f"{result.rmse_px_after:.2f}px over {len(result.loss_history)} LBFGS steps"
    )
    _report_camera_deltas(scaled, result, n_cams)

    # --- Stage B.4b: re-derive gravity + ground from floor-contact landmarks. ---
    # Stage A's scene-cloud RANSAC occasionally locks onto a non-floor plane (a
    # mis-oriented rig that leaves the body floating/sideways). The triangulated
    # floor-contact landmarks are a direct ground signal; re-orienting to them
    # fixes those cases and is a no-op-or-better on the rest (guarded).
    if config.reground:
        grav = reground_from_floor_contact(
            collector.landmarks, k_engine, result.world_to_cam_per_cam, config.floor_contact_thresh, config.device
        )
        if grav is not None:
            r_grav, shift = grav
            rg: Float32[torch.Tensor, "3 3"] = torch.tensor(r_grav, dtype=torch.float32, device=config.device)
            sh: Float32[torch.Tensor, "3"] = torch.tensor(shift, dtype=torch.float32, device=config.device)
            w2c_rg: Float32[torch.Tensor, "c 4 4"] = result.world_to_cam_per_cam.clone()
            for c in range(n_cams):
                r_cw_new: Float32[torch.Tensor, "3 3"] = result.world_to_cam_per_cam[c, :3, :3] @ rg.T
                w2c_rg[c, :3, :3] = r_cw_new
                w2c_rg[c, :3, 3] = result.world_to_cam_per_cam[c, :3, 3] - r_cw_new @ sh
            result = replace(result, world_to_cam_per_cam=w2c_rg)

    # --- Stage B.5: write refined meta/ (native K + refined extrinsics). ---
    out_meta: Path = config.out_meta_dir if config.out_meta_dir is not None else (config.data_dir / "meta")
    if config.out_meta_dir is None:
        coarse_backup: Path = config.data_dir / "meta_coarse"
        if not coarse_backup.exists():
            shutil.copytree(config.data_dir / "meta", coarse_backup)
            print(f"backed up coarse meta -> {coarse_backup}")
    out_meta.mkdir(parents=True, exist_ok=True)
    _write_refined_meta(sequence, result, config.resize_hw, native_h, native_w, out_meta)
    # carry the sequence-level global.npz over unchanged.
    if not (out_meta / "global.npz").exists():
        shutil.copy(config.data_dir / "meta" / "global.npz", out_meta / "global.npz")
    print(f"wrote refined calibration for {n_cams} cameras -> {out_meta}")
    return 0


def _report_camera_deltas(coarse: list[CameraCalibration], result: BundleAdjustResult, n_cams: int) -> None:
    """Print how far each refined camera moved from its coarse (VGGT) pose."""
    import math

    for c in range(n_cams):
        r_old: Float32[ndarray, "3 3"] = coarse[c].world_to_cam[:3, :3].astype(np.float32)
        r_new: Float32[torch.Tensor, "3 3"] = result.world_to_cam_per_cam[c, :3, :3].cpu()
        rel: ndarray = r_old @ r_new.numpy().T
        cos: float = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
        d_rot_deg: float = math.degrees(math.acos(cos))
        t_old: ndarray = coarse[c].world_to_cam[:3, 3].astype(np.float32)
        t_new: ndarray = result.world_to_cam_per_cam[c, :3, 3].cpu().numpy()
        d_trans_mm: float = float(np.linalg.norm(t_new - t_old) * 1000.0)
        f_old: float = float((coarse[c].k_matrix[0, 0] + coarse[c].k_matrix[1, 1]) * 0.5)
        f_new: float = float((result.k_per_cam[c, 0, 0] + result.k_per_cam[c, 1, 1]).cpu() * 0.5)
        print(f"  cam {c}: dRot {d_rot_deg:5.2f}deg  dTrans {d_trans_mm:6.1f}mm  focal {f_old:.1f}->{f_new:.1f}px ({100 * (f_new / f_old - 1):+.1f}%)")


def _write_refined_meta(
    sequence: MultiViewSequence,
    result: BundleAdjustResult,
    resize_hw: tuple[int, int],
    native_h: int,
    native_w: int,
    out_meta: Path,
) -> None:
    """Rescale BA-refined engine-res K back to native res and write per-camera NPZs."""
    for c, cam in enumerate(sequence.cameras):
        k_engine: Float32[ndarray, "3 3"] = result.k_per_cam[c].cpu().numpy().astype(np.float64)
        engine_cam: CameraCalibration = CameraCalibration(
            name=cam.name, k_matrix=k_engine, world_to_cam=result.world_to_cam_per_cam[c].cpu().numpy().astype(np.float64),
            width=resize_hw[1], height=resize_hw[0],
        )
        native_cam: CameraCalibration = engine_cam.scaled_to(height=native_h, width=native_w)
        dump_camera_npz(native_cam, out_meta / f"{cam.name}.npz")


if __name__ == "__main__":
    sys.exit(main(tyro.cli(RefineConfig)))
