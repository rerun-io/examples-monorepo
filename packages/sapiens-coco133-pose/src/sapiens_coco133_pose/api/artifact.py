"""COCO-133 video pose artifacts and Rerun logging helpers."""
# ruff: noqa: I001

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Int, UInt8
from numpy import ndarray
from sapiens2_pose.api.metadata import get_pose_schema, log_annotation_context
from sapiens2_pose.api.pose_artifact import PoseArtifactComparisonConfig, PoseArtifactComparisonMetrics, PosePredictionArtifact, compare_pose_prediction_artifacts
from sapiens2_pose.api.rerun_pose_artifact import _load_one_recording_chunks
from simplecv.rerun_custom_types import Points2DWithConfidence
from simplecv.rerun_log_utils import log_video

_LEGACY_POSE_ENTITY_PATTERN = re.compile(r"^/?video/frame_(?P<frame>\d+)/person_(?P<person>\d+)/(?P<kind>bbox|keypoints)$")
_STABLE_POSE_ENTITY_PATTERN = re.compile(r"^/?video/person_(?P<person>\d+)/(?P<kind>bbox|keypoints)$")
_BOX_CENTERS_COMPONENT = "Boxes2D:centers"
_BOX_HALF_SIZES_COMPONENT = "Boxes2D:half_sizes"
_POINTS_POSITION_COMPONENT = "Points2D:positions"
POINTS_CONFIDENCE_COMPONENT = "simplecv.KeypointConfidence2D:confidences"
POINTS_AVERAGE_CONFIDENCE_COMPONENT = "simplecv.KeypointConfidence2D:average_confidence"
_FRAME_INDEX_COLUMN = "frame"
_VIDEO_TIME_COLUMN = "video_time"
_VIDEO_TIMELINE = "video_time"
_FRAME_TIMELINE = "frame"


@dataclass(frozen=True, slots=True)
class Coco133PoseFrame:
    """Pose predictions for one video frame in COCO WholeBody order."""

    frame_index: int
    """Zero-based video frame index."""
    timestamp_ns: int
    """Frame timestamp in nanoseconds on the video timeline."""
    image_rgb: UInt8[ndarray, "h w 3"] | None
    """Optional RGB image for image-frame logging when video media is not logged separately."""
    bboxes: Float32[ndarray, "n 4"]
    """Person bounding boxes in image-space ``xyxy`` order."""
    keypoints: Float32[ndarray, "n 133 2"]
    """COCO-133 keypoint coordinates in image-space ``xy`` order."""
    scores: Float32[ndarray, "n 133"]
    """Per-keypoint confidence scores aligned with ``keypoints``."""


class Coco133VideoPoseArtifact(NamedTuple):
    """Flat, frame-aware COCO-133 pose artifact loaded from an RRD."""

    frame_indices: Int[ndarray, "n_frames"]
    """Frame ids present in the artifact."""
    timestamps_ns: Int[ndarray, "n_frames"]
    """Frame timestamps in nanoseconds."""
    person_frame_indices: Int[ndarray, "n_persons"]
    """Frame id for each person row."""
    bboxes: Float32[ndarray, "n_persons 4"]
    """Person boxes in ``xyxy`` order."""
    keypoints: Float32[ndarray, "n_persons 133 2"]
    """COCO-133 keypoints in image coordinates."""
    scores: Float32[ndarray, "n_persons 133"]
    """COCO-133 per-keypoint confidence scores."""


def _empty_rows() -> tuple[Float32[ndarray, "0 4"], Float32[ndarray, "0 133 2"], Float32[ndarray, "0 133"]]:
    return np.empty((0, 4), np.float32), np.empty((0, 133, 2), np.float32), np.empty((0, 133), np.float32)


def compare_video_pose_artifacts(baseline: Coco133VideoPoseArtifact, candidate: Coco133VideoPoseArtifact, config: PoseArtifactComparisonConfig) -> PoseArtifactComparisonMetrics:
    """Compare two COCO-133 video artifacts.

    Args:
        baseline: Reference video pose artifact.
        candidate: Candidate video pose artifact to compare against ``baseline``.
        config: Numeric tolerances and keypoint visibility thresholds.

    Returns:
        Aggregate comparison metrics over matching bbox and keypoint rows.

    Raises:
        AssertionError: If frame or person row identities differ before value comparison.
    """
    if not np.array_equal(baseline.frame_indices, candidate.frame_indices):
        raise AssertionError("Video artifact frame_indices differ.")
    if not np.array_equal(baseline.person_frame_indices, candidate.person_frame_indices):
        raise AssertionError("Video artifact person_frame_indices differ.")
    return compare_pose_prediction_artifacts(PosePredictionArtifact(np.asarray(baseline.bboxes, np.float32), np.asarray(baseline.keypoints, np.float32), np.asarray(baseline.scores, np.float32)), PosePredictionArtifact(np.asarray(candidate.bboxes, np.float32), np.asarray(candidate.keypoints, np.float32), np.asarray(candidate.scores, np.float32)), config)


def _frame_timestamp_ns(frame: Coco133PoseFrame, video_timestamps_ns: Int[ndarray, "num_video_frames"] | None) -> int:
    """Return the timestamp used for overlaying a predicted frame on the video timeline."""
    if video_timestamps_ns is None or frame.frame_index < 0 or frame.frame_index >= int(video_timestamps_ns.shape[0]):
        return int(frame.timestamp_ns)
    return int(video_timestamps_ns[int(frame.frame_index)])


def coco133_points2d_with_confidence(
    keypoints: Float32[ndarray, "133 2"],
    scores: Float32[ndarray, "133"],
    *,
    keypoint_threshold: float,
    keypoint_ids: list[int],
    class_id: int = 0,
) -> Points2DWithConfidence:
    """Build the package-wide COCO-133 Rerun keypoint archetype.

    Args:
        keypoints: Image-space COCO-133 keypoint coordinates.
        scores: Per-keypoint confidence scores.
        keypoint_threshold: Minimum confidence for visible keypoints. Lower-confidence points are logged as ``NaN``.
        keypoint_ids: COCO-133 keypoint ids used by Rerun annotation context.
        class_id: Rerun class id for the person instance.

    Returns:
        ``Points2DWithConfidence`` containing positions, confidence values, keypoint ids, and class ids;
        colors derive from the confidences inside the archetype.
    """
    keypoints_f32: Float32[ndarray, "133 2"] = np.asarray(keypoints, dtype=np.float32).reshape(133, 2).copy()
    scores_f32: Float32[ndarray, "133"] = np.asarray(scores, dtype=np.float32).reshape(133).copy()
    visible_mask: Bool[ndarray, "133"] = np.asarray(scores_f32 >= np.float32(keypoint_threshold), dtype=np.bool_)
    keypoints_f32[~visible_mask] = np.nan
    return Points2DWithConfidence(positions=keypoints_f32, confidences=scores_f32, class_ids=class_id, keypoint_ids=keypoint_ids, show_labels=False)


def log_video_pose_frames(frames: list[Coco133PoseFrame], *, keypoint_threshold: float, recording: rr.RecordingStream, video_path: Path | None = None, video_log_path: Path = Path("video")) -> rr.RecordingStream:
    """Log frame-major COCO-133 pose predictions into an existing Rerun recording.

    Args:
        frames: Frame-major pose predictions to log.
        keypoint_threshold: Minimum confidence for visible keypoints.
        recording: Rerun recording stream that receives video, boxes, and keypoints.
        video_path: Optional encoded video path to log once as media.
        video_log_path: Rerun entity path for the video and pose overlays.

    Returns:
        The same recording stream after logging.
    """
    schema = get_pose_schema("coco133")
    video_entity: str = str(video_log_path)
    video_timestamps_ns: Int[ndarray, "num_video_frames"] | None = None
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial2DView(name="COCO-133 Pose", origin=f"/{video_entity}"), collapse_panels=True), recording=recording)
    log_annotation_context(schema, recording=recording)
    if video_path is not None:
        video_timestamps_ns = log_video(video_path, video_log_path=video_log_path, timeline=_VIDEO_TIMELINE, recording=recording)
    previous_person_count: int = 0
    for frame in frames:
        timestamp_ns: int = _frame_timestamp_ns(frame, video_timestamps_ns)
        rr.set_time(_FRAME_TIMELINE, sequence=int(frame.frame_index), recording=recording)
        rr.set_time(_VIDEO_TIMELINE, duration=float(timestamp_ns) * 1e-9, recording=recording)
        if video_path is None and frame.image_rgb is not None:
            rr.log(f"{video_entity}/image", rr.Image(np.asarray(frame.image_rgb, dtype=np.uint8), color_model=rr.ColorModel.RGB), recording=recording)
        person_count: int = min(int(frame.bboxes.shape[0]), int(frame.keypoints.shape[0]), int(frame.scores.shape[0]))
        for person_idx, bbox in enumerate(np.asarray(frame.bboxes[:person_count], dtype=np.float32).reshape(-1, 4)):
            rr.log(f"{video_entity}/person_{person_idx}/bbox", rr.Boxes2D(array=bbox.reshape(1, 4), array_format=rr.Box2DFormat.XYXY, class_ids=0, labels=f"person_{person_idx}", colors=(0, 255, 0)), recording=recording)
            rr.log(f"{video_entity}/person_{person_idx}/keypoints", coco133_points2d_with_confidence(frame.keypoints[person_idx], frame.scores[person_idx], keypoint_threshold=keypoint_threshold, keypoint_ids=schema.keypoint_ids), recording=recording)
        for stale_person_idx in range(person_count, previous_person_count):
            rr.log(f"{video_entity}/person_{stale_person_idx}", rr.Clear(recursive=True), recording=recording)
        previous_person_count = person_count
    return recording


def write_video_pose_rrd(frames: list[Coco133PoseFrame], *, rrd_path: Path, keypoint_threshold: float, application_id: str = "sapiens_coco133_pose", video_path: Path | None = None) -> Path:
    """Write frame-major COCO-133 pose predictions to an RRD file.

    Args:
        frames: Frame-major pose predictions to serialize.
        rrd_path: Output RRD path.
        keypoint_threshold: Minimum confidence for visible keypoints.
        application_id: Rerun application id stored in the recording.
        video_path: Optional encoded video path to log once as media.

    Returns:
        The written RRD path.
    """
    recording = rr.RecordingStream(application_id=application_id)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    with recording:
        log_video_pose_frames(frames, keypoint_threshold=keypoint_threshold, recording=recording, video_path=video_path)
    recording.save(str(rrd_path))
    return rrd_path


def _parse_pose_chunk_path(entity_path: str) -> tuple[int | None, int, str] | None:
    legacy_match: re.Match[str] | None = _LEGACY_POSE_ENTITY_PATTERN.fullmatch(entity_path)
    if legacy_match is not None:
        return int(legacy_match.group("frame")), int(legacy_match.group("person")), legacy_match.group("kind")
    stable_match: re.Match[str] | None = _STABLE_POSE_ENTITY_PATTERN.fullmatch(entity_path)
    if stable_match is None:
        return None
    return None, int(stable_match.group("person")), stable_match.group("kind")


def _duration_ns(value: Any) -> int:
    if hasattr(value, "value"):
        return int(value.value)
    if isinstance(value, np.timedelta64):
        return int(value.astype("timedelta64[ns]").astype(np.int64))
    if isinstance(value, int | np.integer):
        return int(value)
    return int(float(value) * 1_000_000_000)


def _component_rows(batch_data: dict[str, list[Any]], component_name: str, entity_path: str) -> list[Any]:
    if component_name not in batch_data:
        raise ValueError(f"Missing Rerun component {component_name!r} on entity {entity_path!r}.")
    return batch_data[component_name]


def _row_frame_indices(batch_data: dict[str, list[Any]], path_frame: int | None, row_count: int, entity_path: str) -> list[int]:
    if path_frame is not None:
        return [path_frame] * row_count
    if _FRAME_INDEX_COLUMN not in batch_data:
        raise ValueError(f"Missing Rerun frame timeline on stable video pose entity {entity_path!r}.")
    return [int(frame_idx) for frame_idx in batch_data[_FRAME_INDEX_COLUMN][:row_count]]


def _row_timestamps_ns(batch_data: dict[str, list[Any]], row_count: int) -> list[int | None]:
    if _VIDEO_TIME_COLUMN not in batch_data:
        return [None] * row_count
    return [_duration_ns(value) for value in batch_data[_VIDEO_TIME_COLUMN][:row_count]]


def _load_bbox_row(batch_data: dict[str, list[Any]], row_idx: int, entity_path: str) -> Float32[ndarray, "4"]:
    centers: Float32[ndarray, "boxes 2"] = np.asarray(_component_rows(batch_data, _BOX_CENTERS_COMPONENT, entity_path)[row_idx], dtype=np.float32).reshape(-1, 2)
    half_sizes: Float32[ndarray, "boxes 2"] = np.asarray(_component_rows(batch_data, _BOX_HALF_SIZES_COMPONENT, entity_path)[row_idx], dtype=np.float32).reshape(-1, 2)
    if centers.shape != half_sizes.shape:
        raise ValueError(f"Rerun bbox centers/half-sizes shapes differ on {entity_path!r}: {centers.shape} != {half_sizes.shape}.")
    if centers.shape[0] != 1:
        raise ValueError(f"Expected one box on entity {entity_path!r}, found {centers.shape[0]}.")
    return np.concatenate([centers[0] - half_sizes[0], centers[0] + half_sizes[0]], axis=0).astype(np.float32, copy=False)


def _load_keypoints_row(batch_data: dict[str, list[Any]], row_idx: int, entity_path: str) -> Float32[ndarray, "133 2"]:
    return np.asarray(_component_rows(batch_data, _POINTS_POSITION_COMPONENT, entity_path)[row_idx], dtype=np.float32).reshape(133, 2)


def _load_scores_row(batch_data: dict[str, list[Any]], row_idx: int, entity_path: str, keypoints: Float32[ndarray, "133 2"]) -> Float32[ndarray, "133"]:
    if POINTS_CONFIDENCE_COMPONENT not in batch_data:
        return np.asarray(np.isfinite(keypoints).all(axis=-1), dtype=np.float32)
    scores: Float32[ndarray, "n"] = np.asarray(batch_data[POINTS_CONFIDENCE_COMPONENT][row_idx], dtype=np.float32).reshape(-1)
    if scores.shape[0] != 133:
        raise ValueError(f"Expected 133 keypoint confidence values on entity {entity_path!r}, found {scores.shape[0]}.")
    return scores.astype(np.float32, copy=False)


def load_rerun_video_pose_artifact(rrd_path: Path) -> Coco133VideoPoseArtifact:
    """Load bbox and keypoint rows from a COCO-133 video pose RRD.

    Args:
        rrd_path: Rerun recording containing COCO-133 video pose entities.

    Returns:
        A row-major artifact with frame ids, timestamps, person frame ids, bboxes, keypoints, and scores.

    Raises:
        ValueError: If bbox and keypoint rows do not line up by frame and person id.
    """
    frame_indices: set[int] = set()
    timestamps_by_frame: dict[int, int] = {}
    bboxes: dict[tuple[int, int], Float32[ndarray, "4"]] = {}
    keypoints: dict[tuple[int, int], Float32[ndarray, "133 2"]] = {}
    scores: dict[tuple[int, int], Float32[ndarray, "133"]] = {}
    for chunk in _load_one_recording_chunks(rrd_path):
        entity_path: str = str(chunk.entity_path)
        parsed_path: tuple[int | None, int, str] | None = _parse_pose_chunk_path(entity_path)
        if parsed_path is None:
            continue
        path_frame, person_idx, kind = parsed_path
        batch_data: dict[str, list[Any]] = chunk.to_record_batch().to_pydict()
        component_name: str = _BOX_CENTERS_COMPONENT if kind == "bbox" else _POINTS_POSITION_COMPONENT
        row_count: int = len(_component_rows(batch_data, component_name, entity_path))
        frame_rows: list[int] = _row_frame_indices(batch_data, path_frame, row_count, entity_path)
        timestamp_rows: list[int | None] = _row_timestamps_ns(batch_data, row_count)
        for row_idx, frame_idx in enumerate(frame_rows):
            key = (int(frame_idx), int(person_idx))
            frame_indices.add(key[0])
            timestamp_ns: int | None = timestamp_rows[row_idx]
            if timestamp_ns is not None:
                timestamps_by_frame[key[0]] = int(timestamp_ns)
            if kind == "bbox":
                bboxes[key] = _load_bbox_row(batch_data, row_idx, entity_path)
            else:
                keypoint_row: Float32[ndarray, "133 2"] = _load_keypoints_row(batch_data, row_idx, entity_path)
                keypoints[key] = keypoint_row
                scores[key] = _load_scores_row(batch_data, row_idx, entity_path, keypoint_row)
    rows = sorted(bboxes)
    if rows != sorted(keypoints):
        raise ValueError(f"Rerun bbox/keypoint row sets differ in {rrd_path}: bboxes={rows}, keypoints={sorted(keypoints)}.")
    sorted_frame_indices: Int[ndarray, "n_frames"] = np.asarray(sorted(frame_indices), np.int64)
    timestamps_ns: Int[ndarray, "n_frames"] = np.asarray([timestamps_by_frame.get(int(frame_idx), 0) for frame_idx in sorted_frame_indices], np.int64)
    if not rows:
        empty_bboxes, empty_keypoints, empty_scores = _empty_rows()
        return Coco133VideoPoseArtifact(sorted_frame_indices, timestamps_ns, np.empty((0,), np.int64), empty_bboxes, empty_keypoints, empty_scores)
    stacked_keypoints = np.stack([keypoints[row] for row in rows], axis=0).astype(np.float32, copy=False)
    stacked_scores: Float32[ndarray, "n_persons 133"] = np.stack([scores[row] for row in rows], axis=0).astype(np.float32, copy=False)
    return Coco133VideoPoseArtifact(sorted_frame_indices, timestamps_ns, np.asarray([frame for frame, _ in rows], np.int64), np.stack([bboxes[row] for row in rows], axis=0).astype(np.float32, copy=False), stacked_keypoints, stacked_scores)


def compare_video_pose_rrds(baseline_rrd_path: Path, candidate_rrd_path: Path, config: PoseArtifactComparisonConfig) -> PoseArtifactComparisonMetrics:
    """Compare two COCO-133 video pose RRDs.

    Args:
        baseline_rrd_path: Reference RRD path.
        candidate_rrd_path: Candidate RRD path.
        config: Numeric tolerances and keypoint visibility thresholds.

    Returns:
        Aggregate comparison metrics from the loaded RRD artifacts.
    """
    return compare_video_pose_artifacts(load_rerun_video_pose_artifact(baseline_rrd_path), load_rerun_video_pose_artifact(candidate_rrd_path), config)
