"""Compare two Sapiens2 pose numeric artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from sapiens2_pose.api.pose_artifact import (
    PoseArtifactComparisonConfig,
    PoseArtifactComparisonMetrics,
    compare_pose_prediction_artifacts,
    load_pose_prediction_artifact,
)


@dataclass(frozen=True, slots=True)
class ComparePoseArtifactsCli:
    """CLI arguments for comparing pose artifacts."""

    baseline_path: Path
    """Path to the baseline numeric `.npz` pose artifact."""
    candidate_path: Path
    """Path to the candidate numeric `.npz` pose artifact."""
    max_relative_difference: float = 0.05
    """Maximum allowed relative difference for any compared finite value."""
    absolute_floor: float = 1e-6
    """Minimum denominator used when computing relative differences."""
    keypoint_score_threshold: float | None = None
    """Optional score threshold for bbox-normalized keypoint location checks."""
    max_keypoint_bbox_fraction: float | None = None
    """Optional maximum keypoint distance as a fraction of the person-box diagonal."""


def main(args: ComparePoseArtifactsCli) -> None:
    """Run pose artifact comparison."""
    metrics: PoseArtifactComparisonMetrics = compare_pose_prediction_artifacts(
        load_pose_prediction_artifact(args.baseline_path),
        load_pose_prediction_artifact(args.candidate_path),
        PoseArtifactComparisonConfig(
            max_relative_difference=args.max_relative_difference,
            absolute_floor=args.absolute_floor,
            keypoint_score_threshold=args.keypoint_score_threshold,
            max_keypoint_bbox_fraction=args.max_keypoint_bbox_fraction,
        ),
    )
    message: str = (
        "comparison passed "
        f"max_rel={metrics.max_relative_difference:.6f} mean_rel={metrics.mean_relative_difference:.6f} "
        f"max_abs={metrics.max_absolute_difference:.6f} mean_abs={metrics.mean_absolute_difference:.6f} "
        f"values={metrics.compared_values}"
    )
    if metrics.max_keypoint_bbox_fraction is not None:
        message += (
            f" max_keypoint_bbox_fraction={metrics.max_keypoint_bbox_fraction:.6f} "
            f"mean_keypoint_bbox_fraction={metrics.mean_keypoint_bbox_fraction:.6f} "
            f"keypoints={metrics.compared_keypoints}"
        )
    print(message)


if __name__ == "__main__":
    main(tyro.cli(ComparePoseArtifactsCli))
