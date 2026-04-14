"""DPV-SLAM evaluation: run on EuRoC sequences and compute ATE.

Uses ``run_dpvo_pipeline`` with Rerun visualization and multiprocessing
frame reading.  Computes Absolute Trajectory Error (ATE) via ``evo``
after Sim(3) alignment (required for monocular SLAM with unknown scale).
"""

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import rerun as rr
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from jaxtyping import Float64
from numpy import ndarray

from dpvo.api.euroc import EuRoCSequence
from dpvo.api.inference import DPVOPipelineHandle, run_dpvo_pipeline
from dpvo.config import DPVOConfig

try:
    import evo.main_ape as main_ape
except ImportError:
    main_ape = None


@dataclass
class EvalResult:
    """Result of a single evaluation run."""

    sequence: str
    mode: str
    ate_rmse: float
    ate_median: float
    ate_mean: float
    num_frames: int


def poses_to_evo_traj(
    poses: ndarray,
    timestamps: ndarray,
) -> PoseTrajectory3D:
    """Convert lietorch output poses to an evo PoseTrajectory3D.

    lietorch format: ``[tx, ty, tz, qx, qy, qz, qw]`` (scalar-last).
    evo expects: ``orientations_quat_wxyz`` = ``[qw, qx, qy, qz]``.

    Reorder via ``[:, [6, 3, 4, 5]]`` — matches upstream evaluate_euroc.py.
    """
    return PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=timestamps,
    )


def compute_ate(
    pred_traj: PoseTrajectory3D,
    gt_traj: PoseTrajectory3D,
) -> dict[str, float]:
    """Compute ATE after Sim(3) alignment and timestamp association.

    Returns dict with 'rmse', 'mean', 'median', 'std', 'min', 'max'.
    """
    assert main_ape is not None, "evo.main_ape not available"

    # Associate by timestamps (nearest-neighbor matching)
    gt_assoc, pred_assoc = sync.associate_trajectories(gt_traj, pred_traj)

    result = main_ape.ape(
        gt_assoc,
        pred_assoc,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    return result.stats


def run_euroc_eval(
    seq: EuRoCSequence,
    config: DPVOConfig,
    network_path: str,
    stride: int = 2,
    mode_name: str = "vo",
    parent_log_path: Path = Path("world"),
) -> EvalResult:
    """Run DPV-SLAM on a single EuRoC sequence and compute ATE.

    Uses ``run_dpvo_pipeline`` with multiprocessing frame reader and
    Rerun visualization.  Provides EuRoC calibration directly to skip
    DUSt3R auto-estimation.
    """
    # Write calibration file for the pipeline
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        calib_path = f.name
        np.savetxt(f, seq.intrinsics.reshape(1, -1), fmt="%.6f")

    try:
        handle = DPVOPipelineHandle()

        # Run the pipeline
        for _msg in run_dpvo_pipeline(
            dpvo_config=config,
            network_path=network_path,
            imagedir=seq.image_dir,
            calib=calib_path,
            stride=stride,
            handle=handle,
            parent_log_path=parent_log_path,
        ):
            pass

        assert handle.prediction is not None, f"Pipeline produced no prediction for {seq.name}"

        # Post-hoc reconstruct real timestamps from sorted filenames
        # image_stream uses enumerate(0,1,2,...) but we need nanosecond
        # timestamps for ground truth association
        images_list = sorted(glob.glob(os.path.join(seq.image_dir, "*.png")))[::stride]
        real_timestamps: Float64[ndarray, "n"] = np.array([float(Path(f).stem) for f in images_list])

        # The pipeline returns n_frames poses (one per frame processed).
        # Timestamps from real filenames must match.
        n_pred = handle.prediction.final_poses.shape[0]
        assert n_pred <= len(real_timestamps), f"More poses ({n_pred}) than images ({len(real_timestamps)})"
        real_timestamps = real_timestamps[:n_pred]

        # Convert to seconds for evo association
        timestamps_sec = real_timestamps / 1e9

        # Convert poses to evo trajectory with correct quaternion order
        pred_traj = poses_to_evo_traj(handle.prediction.final_poses, timestamps_sec)

        # Compute ATE
        stats = compute_ate(pred_traj, seq.gt_traj)
        ate_rmse = stats["rmse"]
        ate_median = stats["median"]
        ate_mean = stats["mean"]

        # Log metrics to Rerun
        rr.log(f"{parent_log_path}/metrics/ate_rmse", rr.Scalars(ate_rmse))
        rr.log(
            f"{parent_log_path}/metrics/summary",
            rr.TextLog(f"{seq.name} [{mode_name}]: ATE RMSE={ate_rmse:.4f}m, median={ate_median:.4f}m"),
        )

        # Log ground truth trajectory as a line strip
        gt_assoc, _ = sync.associate_trajectories(seq.gt_traj, pred_traj)
        gt_positions = gt_assoc.positions_xyz
        rr.log(
            f"{parent_log_path}/ground_truth",
            rr.LineStrips3D(strips=[gt_positions.tolist()], colors=[0, 255, 0]),
        )

        return EvalResult(
            sequence=seq.name,
            mode=mode_name,
            ate_rmse=ate_rmse,
            ate_median=ate_median,
            ate_mean=ate_mean,
            num_frames=n_pred,
        )

    finally:
        os.unlink(calib_path)


MODE_CONFIGS: dict[str, DPVOConfig] = {
    "vo": DPVOConfig.accurate(),
    "slam": DPVOConfig.slam(),
    "slam++": DPVOConfig.slam_classic(),
}


def print_results_table(results: list[EvalResult], modes: list[str]) -> None:
    """Print a summary table of evaluation results."""
    # Group by sequence
    by_seq: dict[str, dict[str, float]] = {}
    for r in results:
        if r.sequence not in by_seq:
            by_seq[r.sequence] = {}
        by_seq[r.sequence][r.mode] = r.ate_rmse

    # Header
    mode_headers = "".join(f"{m:>12}" for m in modes)
    print(f"\n{'Sequence':<20}{mode_headers}")
    print("-" * (20 + 12 * len(modes)))

    # Rows
    mode_means: dict[str, list[float]] = {m: [] for m in modes}
    for seq_name in by_seq:
        row = f"{seq_name:<20}"
        for m in modes:
            ate = by_seq[seq_name].get(m, float("nan"))
            row += f"{ate:>12.4f}"
            if not np.isnan(ate):
                mode_means[m].append(ate)
        print(row)

    # Mean row
    print("-" * (20 + 12 * len(modes)))
    mean_row = f"{'Mean':<20}"
    for m in modes:
        vals = mode_means[m]
        mean_val = np.mean(vals) if vals else float("nan")
        mean_row += f"{mean_val:>12.4f}"
    print(mean_row)
    print()
