"""Evaluate DPV-SLAM on EuRoC MAV sequences.

Runs the DPVO pipeline in different modes (VO, SLAM, SLAM++) on EuRoC
Machine Hall sequences and reports ATE (Absolute Trajectory Error) after
Sim(3) alignment.

Usage::

    # Quick test: VO + SLAM on all MH sequences
    python tools/eval_euroc.py

    # Single sequence, all modes
    python tools/eval_euroc.py --sequences MH_01_easy --modes vo slam slam++

    # Multiple trials (paper uses 5, reports median)
    python tools/eval_euroc.py --trials 5
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import tyro

from dpvo.api.euroc import EUROC_MH_SEQUENCES, load_euroc_sequence
from dpvo.api.evaluate import MODE_CONFIGS, EvalResult, print_results_table, run_euroc_eval


@dataclass
class EvalConfig:
    """Configuration for EuRoC evaluation."""

    data_root: Path = Path("data/euroc")
    """Root directory containing EuRoC sequences."""
    network_path: str = "checkpoints/dpvo.pth"
    """Path to DPVO network weights."""
    sequences: list[str] = field(default_factory=lambda: list(EUROC_MH_SEQUENCES))
    """EuRoC sequences to evaluate."""
    stride: int = 2
    """Frame stride (upstream uses 2 for EuRoC)."""
    trials: int = 1
    """Number of trials per sequence (paper uses 5, reports median)."""
    modes: list[str] = field(default_factory=lambda: ["vo", "slam"])
    """Evaluation modes: 'vo', 'slam', 'slam++'. Default omits slam++ since it requires dpretrieval."""
    seed: int = 1234
    """Random seed for reproducibility."""


def main(config: EvalConfig) -> None:
    """Run EuRoC evaluation."""
    rr.init("dpvslam-eval", spawn=True)

    torch.manual_seed(config.seed)

    # Validate modes
    for mode in config.modes:
        assert mode in MODE_CONFIGS, f"Unknown mode '{mode}'. Must be one of: {list(MODE_CONFIGS.keys())}"

    all_results: list[EvalResult] = []

    for seq_name in config.sequences:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_name}")
        print(f"{'='*60}")

        seq = load_euroc_sequence(config.data_root, seq_name)
        print(f"  Images: {len(seq.image_files)}, GT poses: {seq.gt_traj.num_poses}")

        for mode in config.modes:
            dpvo_config = MODE_CONFIGS[mode]
            print(f"\n  Mode: {mode} (loop_closure={dpvo_config.loop_closure}, classic={dpvo_config.classic_loop_closure})")

            trial_ates: list[float] = []
            best_result: EvalResult | None = None

            for trial in range(config.trials):
                if config.trials > 1:
                    torch.manual_seed(config.seed + trial)

                log_path = Path(f"{seq_name}/{mode}/trial_{trial}")

                result = run_euroc_eval(
                    seq=seq,
                    config=dpvo_config,
                    network_path=config.network_path,
                    stride=config.stride,
                    mode_name=mode,
                    parent_log_path=log_path,
                )

                trial_ates.append(result.ate_rmse)
                print(f"    Trial {trial + 1}/{config.trials}: ATE RMSE = {result.ate_rmse:.4f}m")

                if best_result is None or result.ate_rmse <= np.median(trial_ates):
                    best_result = result

            # Report median across trials
            median_ate = float(np.median(trial_ates))
            print(f"  → {mode} median ATE: {median_ate:.4f}m ({config.trials} trials)")

            assert best_result is not None
            # Use median ATE for the summary
            all_results.append(
                EvalResult(
                    sequence=seq_name,
                    mode=mode,
                    ate_rmse=median_ate,
                    ate_median=best_result.ate_median,
                    ate_mean=best_result.ate_mean,
                    num_frames=best_result.num_frames,
                )
            )

    # Print summary table
    print_results_table(all_results, config.modes)


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    main(config)
