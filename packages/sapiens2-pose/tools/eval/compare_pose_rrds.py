"""Compare two Sapiens2 pose Rerun recordings by querying stored component values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from sapiens2_pose.api.rerun_pose_artifact import (
    RerunPoseArtifactComparisonConfig,
    RerunPoseArtifactComparisonMetrics,
    compare_rerun_pose_artifacts,
    load_rerun_pose_artifact,
)


@dataclass(frozen=True, slots=True)
class ComparePoseRrdsCli:
    """CLI arguments for comparing pose values stored in Rerun recordings."""

    baseline_rrd_path: Path
    """Path to the baseline `.rrd` pose recording."""
    candidate_rrd_path: Path
    """Path to the candidate `.rrd` pose recording."""
    max_relative_difference: float = 0.05
    """Maximum allowed relative difference for Rerun bbox scalar values."""
    max_keypoint_bbox_fraction: float = 0.05
    """Maximum visible-keypoint distance as a fraction of the person-box diagonal."""
    absolute_floor: float = 1e-6
    """Minimum denominator used for relative differences and bbox diagonals."""


def main(args: ComparePoseRrdsCli) -> None:
    """Run Rerun recording value comparison."""
    metrics: RerunPoseArtifactComparisonMetrics = compare_rerun_pose_artifacts(
        load_rerun_pose_artifact(args.baseline_rrd_path),
        load_rerun_pose_artifact(args.candidate_rrd_path),
        RerunPoseArtifactComparisonConfig(
            max_relative_difference=args.max_relative_difference,
            max_keypoint_bbox_fraction=args.max_keypoint_bbox_fraction,
            absolute_floor=args.absolute_floor,
        ),
    )
    print(
        "rrd comparison passed "
        f"max_bbox_rel={metrics.max_bbox_relative_difference:.6f} "
        f"mean_bbox_rel={metrics.mean_bbox_relative_difference:.6f} "
        f"max_bbox_abs={metrics.max_bbox_absolute_difference:.6f} "
        f"mean_bbox_abs={metrics.mean_bbox_absolute_difference:.6f} "
        f"bboxes={metrics.compared_bboxes} "
        f"max_keypoint_bbox_fraction={metrics.max_keypoint_bbox_fraction:.6f} "
        f"mean_keypoint_bbox_fraction={metrics.mean_keypoint_bbox_fraction:.6f} "
        f"keypoints={metrics.compared_keypoints}"
    )


if __name__ == "__main__":
    main(tyro.cli(ComparePoseRrdsCli))
