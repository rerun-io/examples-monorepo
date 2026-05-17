"""Pose prediction artifacts and numeric comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float32
from numpy import ndarray


@dataclass(frozen=True, slots=True)
class PosePredictionArtifact:
    """Serializable pose prediction arrays used for baseline parity checks."""

    bboxes: Float32[ndarray, "n 4"]
    """Detected person bounding boxes in XYXY image coordinates."""
    keypoints: Float32[ndarray, "n k 2"]
    """Per-person keypoints in image coordinates."""
    scores: Float32[ndarray, "n k"]
    """Per-person keypoint confidence scores."""


@dataclass(frozen=True, slots=True)
class PoseArtifactComparisonConfig:
    """Numeric tolerance configuration for pose artifact comparisons."""

    max_relative_difference: float = 0.05
    """Maximum allowed relative difference for any compared finite value."""
    absolute_floor: float = 1e-6
    """Minimum denominator used when computing relative differences."""
    keypoint_score_threshold: float | None = None
    """Optional score threshold for bbox-normalized keypoint location checks."""
    max_keypoint_bbox_fraction: float | None = None
    """Optional maximum keypoint Euclidean distance as a fraction of the person-box diagonal."""


@dataclass(frozen=True, slots=True)
class PoseArtifactComparisonMetrics:
    """Summary of numeric differences between two pose prediction artifacts."""

    max_relative_difference: float
    """Maximum relative difference across boxes, keypoints, and scores."""
    mean_relative_difference: float
    """Mean relative difference across boxes, keypoints, and scores."""
    max_absolute_difference: float
    """Maximum absolute difference across boxes, keypoints, and scores."""
    mean_absolute_difference: float
    """Mean absolute difference across boxes, keypoints, and scores."""
    compared_values: int
    """Number of finite scalar values compared."""
    max_keypoint_bbox_fraction: float | None = None
    """Maximum visible-keypoint distance as a fraction of the person-box diagonal."""
    mean_keypoint_bbox_fraction: float | None = None
    """Mean visible-keypoint distance as a fraction of the person-box diagonal."""
    compared_keypoints: int | None = None
    """Number of visible keypoints compared with bbox-normalized distance."""


def _as_float32_array(value: Float32[ndarray, "..."]) -> Float32[ndarray, "..."]:
    """Return a contiguous float32 NumPy view/copy."""
    array: Float32[ndarray, "..."] = np.asarray(value, dtype=np.float32)
    return array


def _flatten_artifact(artifact: PosePredictionArtifact) -> Float32[ndarray, "values"]:
    """Flatten all numeric pose artifact fields into one vector."""
    bboxes: Float32[ndarray, "bbox_values"] = _as_float32_array(artifact.bboxes).reshape(-1)
    keypoints: Float32[ndarray, "keypoint_values"] = _as_float32_array(artifact.keypoints).reshape(-1)
    scores: Float32[ndarray, "score_values"] = _as_float32_array(artifact.scores).reshape(-1)
    flattened: Float32[ndarray, "values"] = np.concatenate([bboxes, keypoints, scores], axis=0).astype(np.float32, copy=False)
    return flattened


def _flatten_scalar_fields(
    artifact: PosePredictionArtifact,
    *,
    include_keypoints: bool,
) -> Float32[ndarray, "values"]:
    """Flatten artifact fields that should use scalar relative-difference checks."""
    bboxes: Float32[ndarray, "bbox_values"] = _as_float32_array(artifact.bboxes).reshape(-1)
    scores: Float32[ndarray, "score_values"] = _as_float32_array(artifact.scores).reshape(-1)
    if include_keypoints:
        keypoints: Float32[ndarray, "keypoint_values"] = _as_float32_array(artifact.keypoints).reshape(-1)
        return np.concatenate([bboxes, keypoints, scores], axis=0).astype(np.float32, copy=False)
    return np.concatenate([bboxes, scores], axis=0).astype(np.float32, copy=False)


def _compute_relative_difference_metrics(
    baseline_values: Float32[ndarray, "values"],
    candidate_values: Float32[ndarray, "values"],
    config: PoseArtifactComparisonConfig,
) -> PoseArtifactComparisonMetrics:
    """Compute scalar relative-difference metrics."""
    if baseline_values.shape != candidate_values.shape:
        raise AssertionError(f"Artifact shapes differ: {baseline_values.shape} != {candidate_values.shape}")

    finite_mask: ndarray = np.isfinite(baseline_values) & np.isfinite(candidate_values)
    if not bool(np.all(np.isfinite(baseline_values) == np.isfinite(candidate_values))):
        raise AssertionError("Artifact finite-value masks differ.")
    compared_values: int = int(finite_mask.sum())
    if compared_values == 0:
        raise AssertionError("No finite artifact values were available to compare.")

    baseline_finite: Float32[ndarray, "finite"] = baseline_values[finite_mask]
    candidate_finite: Float32[ndarray, "finite"] = candidate_values[finite_mask]
    absolute_difference: Float32[ndarray, "finite"] = np.abs(candidate_finite - baseline_finite).astype(np.float32, copy=False)
    denominator: Float32[ndarray, "finite"] = np.maximum(np.abs(baseline_finite), np.float32(config.absolute_floor))
    relative_difference: Float32[ndarray, "finite"] = (absolute_difference / denominator).astype(np.float32, copy=False)

    metrics: PoseArtifactComparisonMetrics = PoseArtifactComparisonMetrics(
        max_relative_difference=float(relative_difference.max()),
        mean_relative_difference=float(relative_difference.mean()),
        max_absolute_difference=float(absolute_difference.max()),
        mean_absolute_difference=float(absolute_difference.mean()),
        compared_values=compared_values,
    )
    if metrics.max_relative_difference > config.max_relative_difference:
        raise AssertionError(
            f"max_relative_difference {metrics.max_relative_difference:.6f} > {config.max_relative_difference:.6f}; "
            f"max_absolute_difference={metrics.max_absolute_difference:.6f}"
        )
    return metrics


def compare_pose_prediction_artifacts(
    baseline: PosePredictionArtifact,
    candidate: PosePredictionArtifact,
    config: PoseArtifactComparisonConfig,
) -> PoseArtifactComparisonMetrics:
    """Compare two pose prediction artifacts and assert they meet tolerance."""
    use_bbox_keypoint_metric: bool = config.keypoint_score_threshold is not None or config.max_keypoint_bbox_fraction is not None
    keypoint_score_threshold: float | None = config.keypoint_score_threshold
    max_keypoint_bbox_fraction: float | None = config.max_keypoint_bbox_fraction
    if use_bbox_keypoint_metric and (keypoint_score_threshold is None or max_keypoint_bbox_fraction is None):
        raise ValueError("keypoint_score_threshold and max_keypoint_bbox_fraction must be set together.")
    if not use_bbox_keypoint_metric:
        baseline_values: Float32[ndarray, "values"] = _flatten_artifact(baseline)
        candidate_values: Float32[ndarray, "values"] = _flatten_artifact(candidate)
        return _compute_relative_difference_metrics(baseline_values, candidate_values, config)
    if keypoint_score_threshold is None or max_keypoint_bbox_fraction is None:
        raise ValueError("keypoint_score_threshold and max_keypoint_bbox_fraction must be set together.")

    baseline_values = _flatten_scalar_fields(baseline, include_keypoints=False)
    candidate_values = _flatten_scalar_fields(candidate, include_keypoints=False)
    scalar_metrics: PoseArtifactComparisonMetrics = _compute_relative_difference_metrics(baseline_values, candidate_values, config)
    keypoint_fractions: Float32[ndarray, "visible"] = _compute_visible_keypoint_bbox_fractions(
        baseline,
        candidate,
        keypoint_score_threshold=keypoint_score_threshold,
        absolute_floor=config.absolute_floor,
    )
    max_keypoint_fraction: float = float(keypoint_fractions.max())
    mean_keypoint_fraction: float = float(keypoint_fractions.mean())
    if max_keypoint_fraction > max_keypoint_bbox_fraction:
        raise AssertionError(f"max_keypoint_bbox_fraction {max_keypoint_fraction:.6f} > {max_keypoint_bbox_fraction:.6f}")
    return PoseArtifactComparisonMetrics(
        max_relative_difference=scalar_metrics.max_relative_difference,
        mean_relative_difference=scalar_metrics.mean_relative_difference,
        max_absolute_difference=scalar_metrics.max_absolute_difference,
        mean_absolute_difference=scalar_metrics.mean_absolute_difference,
        compared_values=scalar_metrics.compared_values,
        max_keypoint_bbox_fraction=max_keypoint_fraction,
        mean_keypoint_bbox_fraction=mean_keypoint_fraction,
        compared_keypoints=int(keypoint_fractions.shape[0]),
    )


def _compute_visible_keypoint_bbox_fractions(
    baseline: PosePredictionArtifact,
    candidate: PosePredictionArtifact,
    *,
    keypoint_score_threshold: float,
    absolute_floor: float,
) -> Float32[ndarray, "visible"]:
    """Return visible keypoint distances normalized by each person-box diagonal."""
    baseline_bboxes: Float32[ndarray, "n 4"] = _as_float32_array(baseline.bboxes)
    candidate_bboxes: Float32[ndarray, "n 4"] = _as_float32_array(candidate.bboxes)
    baseline_keypoints: Float32[ndarray, "n k 2"] = _as_float32_array(baseline.keypoints)
    candidate_keypoints: Float32[ndarray, "n k 2"] = _as_float32_array(candidate.keypoints)
    baseline_scores: Float32[ndarray, "n k"] = _as_float32_array(baseline.scores)
    candidate_scores: Float32[ndarray, "n k"] = _as_float32_array(candidate.scores)
    if baseline_bboxes.shape != candidate_bboxes.shape:
        raise AssertionError(f"Artifact bbox shapes differ: {baseline_bboxes.shape} != {candidate_bboxes.shape}")
    if baseline_keypoints.shape != candidate_keypoints.shape:
        raise AssertionError(f"Artifact keypoint shapes differ: {baseline_keypoints.shape} != {candidate_keypoints.shape}")
    if baseline_scores.shape != candidate_scores.shape:
        raise AssertionError(f"Artifact score shapes differ: {baseline_scores.shape} != {candidate_scores.shape}")

    visible_mask: Bool[ndarray, "n k"] = np.asarray(
        (baseline_scores >= keypoint_score_threshold) & (candidate_scores >= keypoint_score_threshold),
        dtype=np.bool_,
    )
    keypoint_finite_mask: Bool[ndarray, "n k"] = np.asarray(
        np.isfinite(baseline_keypoints).all(axis=-1) & np.isfinite(candidate_keypoints).all(axis=-1),
        dtype=np.bool_,
    )
    visible_mask = visible_mask & keypoint_finite_mask
    compared_keypoints: int = int(visible_mask.sum())
    if compared_keypoints == 0:
        raise AssertionError("No visible keypoints were available to compare.")

    keypoint_distances: Float32[ndarray, "n k"] = np.linalg.norm(candidate_keypoints - baseline_keypoints, axis=-1).astype(np.float32)
    bbox_widths: Float32[ndarray, "n"] = np.abs(baseline_bboxes[:, 2] - baseline_bboxes[:, 0]).astype(np.float32)
    bbox_heights: Float32[ndarray, "n"] = np.abs(baseline_bboxes[:, 3] - baseline_bboxes[:, 1]).astype(np.float32)
    bbox_diagonals: Float32[ndarray, "n"] = np.maximum(
        np.sqrt(np.square(bbox_widths) + np.square(bbox_heights)).astype(np.float32),
        np.float32(absolute_floor),
    )
    fractions: Float32[ndarray, "n k"] = (keypoint_distances / bbox_diagonals[:, None]).astype(np.float32)
    visible_fractions: Float32[ndarray, "visible"] = fractions[visible_mask].astype(np.float32, copy=False)
    return visible_fractions


def save_pose_prediction_artifact(artifact: PosePredictionArtifact, artifact_path: Path) -> Path:
    """Write a pose prediction artifact to a compressed NumPy archive."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        bboxes=_as_float32_array(artifact.bboxes),
        keypoints=_as_float32_array(artifact.keypoints),
        scores=_as_float32_array(artifact.scores),
    )
    return artifact_path


def load_pose_prediction_artifact(artifact_path: Path) -> PosePredictionArtifact:
    """Load a pose prediction artifact from a compressed NumPy archive."""
    with np.load(artifact_path) as data:
        bboxes: Float32[ndarray, "n 4"] = np.asarray(data["bboxes"], dtype=np.float32)
        keypoints: Float32[ndarray, "n k 2"] = np.asarray(data["keypoints"], dtype=np.float32)
        scores: Float32[ndarray, "n k"] = np.asarray(data["scores"], dtype=np.float32)
    return PosePredictionArtifact(bboxes=bboxes, keypoints=keypoints, scores=scores)
