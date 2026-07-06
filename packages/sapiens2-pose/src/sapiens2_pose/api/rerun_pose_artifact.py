"""Rerun RRD pose artifact query and comparison helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rerun.experimental as rrx
from jaxtyping import Bool, Float32
from numpy import ndarray

_POSE_ENTITY_PATTERN: re.Pattern[str] = re.compile(r"^/image/person_(?P<person_idx>\d+)/(?P<kind>bbox|keypoints)$")
_POINTS_POSITION_COMPONENT: str = "Points2D:positions"
_BOX_CENTERS_COMPONENT: str = "Boxes2D:centers"
_BOX_HALF_SIZES_COMPONENT: str = "Boxes2D:half_sizes"


@dataclass(frozen=True, slots=True)
class RerunPoseArtifact:
    """Pose values queried directly from a Rerun `.rrd` recording."""

    bboxes: Float32[ndarray, "n 4"]
    """Per-person bounding boxes reconstructed from Rerun `Boxes2D` centers and half-sizes in XYXY image coordinates."""
    keypoints: Float32[ndarray, "n k 2"]
    """Per-person keypoints queried from Rerun `Points2D` positions, with hidden points represented as NaN."""


@dataclass(frozen=True, slots=True)
class RerunPoseArtifactComparisonConfig:
    """Tolerance configuration for Rerun pose artifact comparisons."""

    max_relative_difference: float = 0.05
    """Maximum allowed relative difference for Rerun bbox scalar values."""
    max_keypoint_bbox_fraction: float = 0.05
    """Maximum visible-keypoint distance as a fraction of the person-box diagonal."""
    absolute_floor: float = 1e-6
    """Minimum denominator used for relative differences and bbox diagonals."""


@dataclass(frozen=True, slots=True)
class RerunPoseArtifactComparisonMetrics:
    """Summary of value differences between two Rerun pose recordings."""

    max_bbox_relative_difference: float
    """Maximum relative difference across queried bbox scalar values."""
    mean_bbox_relative_difference: float
    """Mean relative difference across queried bbox scalar values."""
    max_bbox_absolute_difference: float
    """Maximum absolute difference across queried bbox scalar values."""
    mean_bbox_absolute_difference: float
    """Mean absolute difference across queried bbox scalar values."""
    compared_bboxes: int
    """Number of finite bbox scalar values compared."""
    max_keypoint_bbox_fraction: float
    """Maximum visible-keypoint distance as a fraction of the person-box diagonal."""
    mean_keypoint_bbox_fraction: float
    """Mean visible-keypoint distance as a fraction of the person-box diagonal."""
    compared_keypoints: int
    """Number of visible Rerun keypoints compared."""


def _parse_pose_entity_path(entity_path: str) -> tuple[int, Literal["bbox", "keypoints"]] | None:
    """Parse a Sapiens2 pose entity path written by the Rerun demos."""
    match: re.Match[str] | None = _POSE_ENTITY_PATTERN.fullmatch(entity_path)
    if match is None:
        return None
    person_idx: int = int(match.group("person_idx"))
    kind: Literal["bbox", "keypoints"] = "bbox" if match.group("kind") == "bbox" else "keypoints"
    return person_idx, kind


def _load_one_recording_chunks(rrd_path: Path) -> list[Any]:
    """Load chunks from a single-recording Rerun archive."""
    reader: rrx.RrdReader = rrx.RrdReader(str(rrd_path))
    recordings: list[Any] = list(reader.recordings())
    if len(recordings) != 1:
        raise ValueError(f"Expected exactly one recording in {rrd_path}, found {len(recordings)}.")
    chunks: list[Any] = list(reader.stream(store=recordings[0]).to_chunks())
    return chunks


def _single_row_value(batch_data: dict[str, list[Any]], component_name: str, entity_path: str) -> Any:
    """Return one component value from a one-row Rerun chunk."""
    if component_name not in batch_data:
        raise ValueError(f"Missing Rerun component {component_name!r} on entity {entity_path!r}.")
    rows: list[Any] = batch_data[component_name]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for Rerun component {component_name!r} on entity {entity_path!r}, found {len(rows)}.")
    row_value: Any = rows[0]
    return row_value


def _load_keypoints_from_chunk(chunk: Any) -> Float32[ndarray, "k 2"]:
    """Load `Points2D:positions` from one Rerun chunk."""
    batch_data: dict[str, list[Any]] = chunk.to_record_batch().to_pydict()
    row_value: Any = _single_row_value(batch_data, _POINTS_POSITION_COMPONENT, str(chunk.entity_path))
    keypoints: Float32[ndarray, "k 2"] = np.asarray(row_value, dtype=np.float32).reshape(-1, 2)
    return keypoints


def _load_bbox_from_chunk(chunk: Any) -> Float32[ndarray, "4"]:
    """Load `Boxes2D` centers and half-sizes from one Rerun chunk as XYXY."""
    batch_data: dict[str, list[Any]] = chunk.to_record_batch().to_pydict()
    centers: Float32[ndarray, "boxes 2"] = np.asarray(
        _single_row_value(batch_data, _BOX_CENTERS_COMPONENT, str(chunk.entity_path)),
        dtype=np.float32,
    ).reshape(-1, 2)
    half_sizes: Float32[ndarray, "boxes 2"] = np.asarray(
        _single_row_value(batch_data, _BOX_HALF_SIZES_COMPONENT, str(chunk.entity_path)),
        dtype=np.float32,
    ).reshape(-1, 2)
    if centers.shape != half_sizes.shape:
        raise ValueError(f"Rerun bbox centers/half-sizes shapes differ on {chunk.entity_path!r}: {centers.shape} != {half_sizes.shape}.")
    if centers.shape[0] != 1:
        raise ValueError(f"Expected one box on entity {chunk.entity_path!r}, found {centers.shape[0]}.")
    xyxy: Float32[ndarray, "4"] = np.concatenate([centers[0] - half_sizes[0], centers[0] + half_sizes[0]], axis=0).astype(
        np.float32,
        copy=False,
    )
    return xyxy


def load_rerun_pose_artifact(rrd_path: Path) -> RerunPoseArtifact:
    """Query person boxes and visible keypoint positions directly from a Sapiens2 pose RRD file."""
    bboxes_by_person: dict[int, Float32[ndarray, "4"]] = {}
    keypoints_by_person: dict[int, Float32[ndarray, "k 2"]] = {}

    for chunk in _load_one_recording_chunks(rrd_path):
        entity_path: str = str(chunk.entity_path)
        parsed_path: tuple[int, Literal["bbox", "keypoints"]] | None = _parse_pose_entity_path(entity_path)
        if parsed_path is None:
            continue
        person_idx: int = parsed_path[0]
        kind: Literal["bbox", "keypoints"] = parsed_path[1]
        if kind == "bbox":
            bboxes_by_person[person_idx] = _load_bbox_from_chunk(chunk)
        else:
            keypoints_by_person[person_idx] = _load_keypoints_from_chunk(chunk)

    person_indices: list[int] = sorted(bboxes_by_person)
    if not person_indices:
        raise ValueError(f"No `/image/person_*/bbox` entities found in {rrd_path}.")
    if person_indices != sorted(keypoints_by_person):
        raise ValueError(
            f"Rerun bbox/keypoint person sets differ in {rrd_path}: "
            f"bboxes={person_indices}, keypoints={sorted(keypoints_by_person)}."
        )

    bboxes: Float32[ndarray, "n 4"] = np.stack([bboxes_by_person[idx] for idx in person_indices], axis=0).astype(np.float32, copy=False)
    keypoints: Float32[ndarray, "n k 2"] = np.stack([keypoints_by_person[idx] for idx in person_indices], axis=0).astype(
        np.float32,
        copy=False,
    )
    return RerunPoseArtifact(bboxes=bboxes, keypoints=keypoints)


def _compute_bbox_metrics(
    baseline_bboxes: Float32[ndarray, "n 4"],
    candidate_bboxes: Float32[ndarray, "n 4"],
    config: RerunPoseArtifactComparisonConfig,
) -> tuple[float, float, float, float, int]:
    """Compute relative and absolute difference metrics for Rerun bboxes."""
    if baseline_bboxes.shape != candidate_bboxes.shape:
        raise AssertionError(f"Rerun bbox shapes differ: {baseline_bboxes.shape} != {candidate_bboxes.shape}.")
    baseline_values: Float32[ndarray, "values"] = np.asarray(baseline_bboxes, dtype=np.float32).reshape(-1)
    candidate_values: Float32[ndarray, "values"] = np.asarray(candidate_bboxes, dtype=np.float32).reshape(-1)
    finite_mask: Bool[ndarray, "values"] = np.asarray(np.isfinite(baseline_values) & np.isfinite(candidate_values), dtype=np.bool_)
    if not bool(np.all(np.isfinite(baseline_values) == np.isfinite(candidate_values))):
        raise AssertionError("Rerun bbox finite-value masks differ.")
    compared_values: int = int(finite_mask.sum())
    if compared_values == 0:
        raise AssertionError("No finite Rerun bbox values were available to compare.")

    baseline_finite: Float32[ndarray, "finite"] = baseline_values[finite_mask]
    candidate_finite: Float32[ndarray, "finite"] = candidate_values[finite_mask]
    absolute_difference: Float32[ndarray, "finite"] = np.abs(candidate_finite - baseline_finite).astype(np.float32, copy=False)
    denominator: Float32[ndarray, "finite"] = np.maximum(np.abs(baseline_finite), np.float32(config.absolute_floor))
    relative_difference: Float32[ndarray, "finite"] = (absolute_difference / denominator).astype(np.float32, copy=False)

    max_relative_difference: float = float(relative_difference.max())
    mean_relative_difference: float = float(relative_difference.mean())
    max_absolute_difference: float = float(absolute_difference.max())
    mean_absolute_difference: float = float(absolute_difference.mean())
    if max_relative_difference > config.max_relative_difference:
        raise AssertionError(
            f"Rerun max_bbox_relative_difference {max_relative_difference:.6f} > {config.max_relative_difference:.6f}; "
            f"max_bbox_absolute_difference={max_absolute_difference:.6f}"
        )
    return max_relative_difference, mean_relative_difference, max_absolute_difference, mean_absolute_difference, compared_values


def _compute_keypoint_bbox_fractions(
    baseline: RerunPoseArtifact,
    candidate: RerunPoseArtifact,
    config: RerunPoseArtifactComparisonConfig,
) -> Float32[ndarray, "visible"]:
    """Compute visible-keypoint distances normalized by person-box diagonal."""
    if baseline.keypoints.shape != candidate.keypoints.shape:
        raise AssertionError(f"Rerun keypoint shapes differ: {baseline.keypoints.shape} != {candidate.keypoints.shape}.")
    if baseline.bboxes.shape[0] != baseline.keypoints.shape[0]:
        raise AssertionError(f"Rerun baseline person counts differ between boxes and keypoints: {baseline.bboxes.shape[0]} != {baseline.keypoints.shape[0]}.")
    if candidate.bboxes.shape[0] != candidate.keypoints.shape[0]:
        raise AssertionError(f"Rerun candidate person counts differ between boxes and keypoints: {candidate.bboxes.shape[0]} != {candidate.keypoints.shape[0]}.")

    baseline_visibility: Bool[ndarray, "n k"] = np.asarray(np.isfinite(baseline.keypoints).all(axis=-1), dtype=np.bool_)
    candidate_visibility: Bool[ndarray, "n k"] = np.asarray(np.isfinite(candidate.keypoints).all(axis=-1), dtype=np.bool_)
    if not bool(np.array_equal(baseline_visibility, candidate_visibility)):
        changed_values: int = int(np.count_nonzero(baseline_visibility != candidate_visibility))
        raise AssertionError(f"Rerun keypoint visibility masks differ at {changed_values} positions.")
    compared_keypoints: int = int(baseline_visibility.sum())
    if compared_keypoints == 0:
        raise AssertionError("No visible Rerun keypoints were available to compare.")

    keypoint_distances: Float32[ndarray, "n k"] = np.linalg.norm(candidate.keypoints - baseline.keypoints, axis=-1).astype(np.float32)
    bbox_widths: Float32[ndarray, "n"] = np.abs(baseline.bboxes[:, 2] - baseline.bboxes[:, 0]).astype(np.float32)
    bbox_heights: Float32[ndarray, "n"] = np.abs(baseline.bboxes[:, 3] - baseline.bboxes[:, 1]).astype(np.float32)
    bbox_diagonals: Float32[ndarray, "n"] = np.maximum(
        np.sqrt(np.square(bbox_widths) + np.square(bbox_heights)).astype(np.float32),
        np.float32(config.absolute_floor),
    )
    fractions: Float32[ndarray, "n k"] = (keypoint_distances / bbox_diagonals[:, None]).astype(np.float32)
    visible_fractions: Float32[ndarray, "visible"] = fractions[baseline_visibility].astype(np.float32, copy=False)
    return visible_fractions


def compare_rerun_pose_artifacts(
    baseline: RerunPoseArtifact,
    candidate: RerunPoseArtifact,
    config: RerunPoseArtifactComparisonConfig,
) -> RerunPoseArtifactComparisonMetrics:
    """Compare pose values queried from two Rerun recordings."""
    bbox_metrics: tuple[float, float, float, float, int] = _compute_bbox_metrics(baseline.bboxes, candidate.bboxes, config)
    keypoint_fractions: Float32[ndarray, "visible"] = _compute_keypoint_bbox_fractions(baseline, candidate, config)
    max_keypoint_fraction: float = float(keypoint_fractions.max())
    mean_keypoint_fraction: float = float(keypoint_fractions.mean())
    if max_keypoint_fraction > config.max_keypoint_bbox_fraction:
        raise AssertionError(f"Rerun max_keypoint_bbox_fraction {max_keypoint_fraction:.6f} > {config.max_keypoint_bbox_fraction:.6f}")
    return RerunPoseArtifactComparisonMetrics(
        max_bbox_relative_difference=bbox_metrics[0],
        mean_bbox_relative_difference=bbox_metrics[1],
        max_bbox_absolute_difference=bbox_metrics[2],
        mean_bbox_absolute_difference=bbox_metrics[3],
        compared_bboxes=bbox_metrics[4],
        max_keypoint_bbox_fraction=max_keypoint_fraction,
        mean_keypoint_bbox_fraction=mean_keypoint_fraction,
        compared_keypoints=int(keypoint_fractions.shape[0]),
    )
