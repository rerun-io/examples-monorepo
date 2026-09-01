"""Hill-climb sweep: run many refine configurations in-process and score each.

Loads Stage A/B outputs once, then for every experiment: joint selection →
AssemblyHands-X post-processing → Kineo correspondences → robust triangulation
→ kornia BA → SE(3) evaluation against GT. GT touches only the scoring.
Results append to ``sweep_results.csv`` next to ``eval.json``.
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.apis.calibrate_init import InitCameras
from exo_calib.apis.keypoints2d import CameraKeypoints, postprocess_confidences
from exo_calib.apis.refine import load_stage_b
from exo_calib.catalog_io import (
    DEFAULT_CATALOG_URL,
    DEFAULT_DATASET_NAME,
    EXO_CAMERA_NAMES,
    GtCameras,
    connect_dataset,
    only_segment_id,
    read_gt_cameras,
)

JOINT_SUBSETS: dict[str, tuple[tuple[int, int], ...]] = {
    "all": ((0, 133),),
    "no-face": ((0, 23), (91, 133)),
    "body-feet": ((0, 23),),
    "body": ((0, 17),),
    "hands": ((91, 133),),
}


@dataclass
class Experiment:
    """One refine configuration to score."""

    name: str
    """Row label in the results table."""
    joints: str = "all"
    """Joint subset key from ``JOINT_SUBSETS``."""
    median_window: int = 5
    """Temporal median filter window (frames)."""
    margin_px: float = 32.0
    """AssemblyHands-X crop-edge margin (crop pixels)."""
    conf_tau: float = 0.15
    """AssemblyHands-X confidence threshold."""
    min_pair_conf: float = 0.75
    """Kineo pair rule bound."""
    frame_stride: int = 3
    """Temporal stride over Stage B frames."""
    max_points: int | None = 6000
    """3D point budget."""
    reproj_threshold_px: float = 10.0
    """Robust triangulation threshold."""
    ba_rounds: int = 2
    """BA / re-triangulation rounds."""
    ba_max_iterations: int = 50
    """LM iterations per round."""
    robust: str = "huber"
    """Robust kernel."""
    robust_scale_px: float = 2.0
    """Robust scale in pixels."""
    pose_prior_sigma_m: float | None = 0.10
    """Camera-center prior sigma (meters); None disables."""
    conf_gamma: float = 1.0
    """Exponent applied to observation confidences before the weighted DLT."""
    min_views: int = 2
    """Minimum surviving views per point."""
    seed: int = 0
    """Point-budget subsampling seed."""
    chain: str = ""
    """Optional second BA stage kernel, e.g. ``cauchy:1.0`` (re-anchors priors to stage-1 output)."""


@dataclass
class SweepConfig:
    """Config for the sweep tool."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """Rerun catalog server URL."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset holding the registered segment."""
    segment_id: str | None = None
    """Segment to sweep; ``None`` uses the dataset's single segment."""
    output_dir: Path = Path("data/outputs")
    """Directory holding Stage A/B outputs."""
    batch: str = "baseline"
    """Name of the experiment batch defined in ``BATCHES``."""


def run_experiment(
    experiment: Experiment,
    per_camera: dict[str, CameraKeypoints],
    init: InitCameras,
    gt: GtCameras,
    postprocess_cache: dict[tuple, tuple],
) -> dict:
    """Run one configuration end to end and return its metrics row."""
    from exo_calib.ba import refine_extrinsics
    from exo_calib.correspondences import build_observations
    from exo_calib.eval import evaluate_rig
    from exo_calib.triangulation import triangulate_robust

    cache_key: tuple = (experiment.median_window, experiment.margin_px, experiment.conf_tau)
    if cache_key not in postprocess_cache:
        num_frames: int = min(len(cam.sample_indices) for cam in per_camera.values())
        kp_list, conf_list = [], []
        for name in EXO_CAMERA_NAMES:
            kp_xy, conf = postprocess_confidences(
                per_camera[name], experiment.median_window, experiment.margin_px, experiment.conf_tau, crop_wh=(768, 1024)
            )
            kp_list.append(kp_xy[:num_frames])
            conf_list.append(conf[:num_frames])
        postprocess_cache[cache_key] = (np.stack(kp_list).astype(np.float64), np.stack(conf_list))
    kp_xy_vtj2, conf_vtj = postprocess_cache[cache_key]

    joint_mask_j: Float64[ndarray, "133"] = np.zeros(133, dtype=np.float64)
    for start, stop in JOINT_SUBSETS[experiment.joints]:
        joint_mask_j[start:stop] = 1.0
    conf_masked_vtj: Float64[ndarray, "v t 133"] = conf_vtj * joint_mask_j[None, None, :]

    obs = build_observations(
        kp_xy_vtj2,
        conf_masked_vtj,
        frame_stride=experiment.frame_stride,
        min_pair_conf=experiment.min_pair_conf,
        min_views=experiment.min_views,
        max_points=experiment.max_points,
        seed=experiment.seed,
    )
    if experiment.conf_gamma != 1.0:
        obs = type(obs)(
            point_frame_idx=obs.point_frame_idx,
            point_joint_idx=obs.point_joint_idx,
            obs_point_idx=obs.obs_point_idx,
            obs_view_idx=obs.obs_view_idx,
            obs_xy=obs.obs_xy,
            obs_conf=obs.obs_conf**experiment.conf_gamma,
        )
    points_init, obs = triangulate_robust(
        obs, init.k_v33, init.cam_T_world_v44, reproj_threshold_px=experiment.reproj_threshold_px
    )
    result = refine_extrinsics(
        init.cam_T_world_v44,
        init.k_v33,
        obs,
        points_init,
        rounds=experiment.ba_rounds,
        robust=experiment.robust,  # type: ignore[arg-type]
        robust_scale_px=experiment.robust_scale_px,
        max_iterations=experiment.ba_max_iterations,
        pose_prior_sigma_m=experiment.pose_prior_sigma_m,
        fixed_pose_indices=(),
    )
    if experiment.chain:
        chain_kernel, chain_scale = experiment.chain.split(":")
        result = refine_extrinsics(
            result.cam_T_world_v44,
            init.k_v33,
            obs,
            result.points_xyz_n3,
            rounds=experiment.ba_rounds,
            robust=chain_kernel,  # type: ignore[arg-type]
            robust_scale_px=float(chain_scale),
            max_iterations=experiment.ba_max_iterations,
            pose_prior_sigma_m=experiment.pose_prior_sigma_m,
            fixed_pose_indices=(),
        )
    errors = evaluate_rig(result.cam_T_world_v44, gt.cam_T_world_v44, mode="se3")
    errors_sim3 = evaluate_rig(result.cam_T_world_v44, gt.cam_T_world_v44, mode="sim3")
    return {
        **asdict(experiment),
        "n_points": int(obs.point_frame_idx.size),
        "n_obs": int(obs.obs_point_idx.size),
        "reproj_px": round(result.mean_reproj_px_per_round[-1], 3),
        "trans_cm_mean": round(float(errors.translation_cm_v.mean()), 3),
        "trans_cm_max": round(float(errors.translation_cm_v.max()), 3),
        "rot_deg_mean": round(float(errors.rotation_error_deg_v.mean()), 3),
        "rot_deg_max": round(float(errors.rotation_error_deg_v.max()), 3),
        "sim3_trans_cm_mean": round(float(errors_sim3.translation_cm_v.mean()), 3),
        "sim3_scale": round(float(errors_sim3.scale), 4),
    }


BATCHES: dict[str, list[Experiment]] = {
    "baseline": [
        Experiment(name="v1-default"),
        Experiment(name="seed1", seed=1),
        Experiment(name="seed2", seed=2),
    ],
    "joints": [
        Experiment(name="all"),
        Experiment(name="no-face", joints="no-face"),
        Experiment(name="body-feet", joints="body-feet"),
        Experiment(name="body", joints="body"),
        Experiment(name="hands", joints="hands"),
    ],
    "pair-conf": [Experiment(name=f"pair{v}", min_pair_conf=v) for v in (0.5, 0.65, 0.75, 0.85)],
    "pair-hi": [Experiment(name=f"pair{v}", min_pair_conf=v) for v in (0.88, 0.9, 0.92, 0.95)],
    "robust": [
        Experiment(name=f"{kernel}-{scale}", robust=kernel, robust_scale_px=scale)
        for kernel in ("huber", "cauchy")
        for scale in (1.0, 2.0, 4.0, 6.0)
    ],
    "prior": [
        Experiment(name=f"sigma{v}", pose_prior_sigma_m=v) for v in (0.02, 0.05, 0.10, 0.15, 0.25, 0.50)
    ],
    "combo": [
        Experiment(name="cauchy1-pair85", robust="cauchy", robust_scale_px=1.0, min_pair_conf=0.85),
        Experiment(name="cauchy1-pair95", robust="cauchy", robust_scale_px=1.0, min_pair_conf=0.95),
        Experiment(name="cauchy1-12k", robust="cauchy", robust_scale_px=1.0, frame_stride=1, max_points=12000),
        Experiment(name="cauchy1-mv3", robust="cauchy", robust_scale_px=1.0, min_views=3),
        Experiment(name="cauchy05", robust="cauchy", robust_scale_px=0.5),
        Experiment(name="cauchy1-sigma5", robust="cauchy", robust_scale_px=1.0, pose_prior_sigma_m=0.5),
        Experiment(name="hub2>cau1", chain="cauchy:1.0"),
        Experiment(name="hub2>cau1-pair95", chain="cauchy:1.0", min_pair_conf=0.95),
        Experiment(name="hub2>cau05", chain="cauchy:0.5"),
        Experiment(name="cau1>hub2", robust="cauchy", robust_scale_px=1.0, chain="huber:2.0"),
    ],
    "budget": [
        Experiment(name="stride1", frame_stride=1),
        Experiment(name="stride1-12k", frame_stride=1, max_points=12000),
        Experiment(name="stride1-uncapped", frame_stride=1, max_points=None),
        Experiment(name="minviews3", min_views=3),
        Experiment(name="thr5", reproj_threshold_px=5.0),
        Experiment(name="thr20", reproj_threshold_px=20.0),
        Experiment(name="rounds4", ba_rounds=4),
        Experiment(name="iters200", ba_max_iterations=200),
        Experiment(name="gamma2", conf_gamma=2.0),
        Experiment(name="gamma05", conf_gamma=0.5),
    ],
}


def main(config: SweepConfig) -> None:
    """Run one experiment batch and append rows to ``sweep_results.csv``."""
    dataset = connect_dataset(config.catalog_url, config.dataset_name)
    segment_id: str = config.segment_id or only_segment_id(dataset)
    segment_dir: Path = config.output_dir / segment_id
    init: InitCameras = InitCameras.load(segment_dir / "init_cameras.npz")
    gt: GtCameras = read_gt_cameras(dataset, segment_id)
    per_camera: dict[str, CameraKeypoints] = load_stage_b(config.output_dir, segment_id, EXO_CAMERA_NAMES)

    postprocess_cache: dict[tuple, tuple] = {}
    rows: list[dict] = []
    for experiment in BATCHES[config.batch]:
        row: dict = run_experiment(experiment, per_camera, init, gt, postprocess_cache)
        rows.append(row)
        print(
            f"{config.batch}/{row['name']:>16s}: trans {row['trans_cm_mean']:6.2f} cm (max {row['trans_cm_max']:6.2f}) | "
            f"rot {row['rot_deg_mean']:5.2f} deg (max {row['rot_deg_max']:5.2f}) | sim3 {row['sim3_trans_cm_mean']:5.2f} cm "
            f"scale {row['sim3_scale']:.4f} | reproj {row['reproj_px']:5.2f} px | {row['n_points']} pts"
        )

    csv_path: Path = segment_dir / "sweep_results.csv"
    write_header: bool = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch", *rows[0].keys()])
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({"batch": config.batch, **row})
    print(f"appended {len(rows)} rows to {csv_path}")


