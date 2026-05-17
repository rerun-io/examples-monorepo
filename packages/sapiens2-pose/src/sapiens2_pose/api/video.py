"""Video-to-Rerun Sapiens2 pose pipeline used by the CLI and Gradio UI."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import cv2
import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Int, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import log_video
from tqdm import tqdm

from sapiens2_pose.api.metadata import (
    KeypointSchemaName,
    PoseSchema,
    get_pose_schema,
    log_annotation_context,
    project_instances_to_schema,
)
from sapiens2_pose.api.pose_artifact import PosePredictionArtifact
from sapiens2_pose.api.rerun_logging import (
    build_video_blueprint,
    clear_initial_detr_boxes,
    log_initial_detr_boxes,
    log_video_pose_frame,
    log_video_segmentation_annotation_context,
    save_recording,
)
from sapiens2_pose.api.runtime import (
    DEFAULT_BBOX_THR,
    DEFAULT_MODEL_SIZE,
    DEFAULT_NMS_THR,
    DeviceChoice,
    ModelSize,
    PoseEstimation,
    detect_persons,
    estimate_frame_pose,
    estimate_frame_pose_from_bboxes,
)
from sapiens2_pose.api.sam3_tracking import (
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_SAM3_MEMORY_RETENTION_FRAMES,
    DEFAULT_SAM3_MIN_MASK_AREA_PX,
    Sam3BoxPromptVideoTracker,
    Sam3TrackerConfig,
    TrackedDetections,
)
from sapiens2_pose.api.tensorrt_pose import TensorRtPoseHeatmapRunner, estimate_sapiens_pose_tensorrt

TrackingBackend = Literal["detr_per_frame", "sam3_tracking"]
PoseBackend = Literal["pytorch", "tensorrt"]
Sam3DTypeChoice = Literal["bfloat16", "float16", "float32"]


class VideoMetadata(NamedTuple):
    """Video file metadata from OpenCV probing."""

    width: int
    height: int
    fps: float
    total_frames: int


@dataclass(frozen=True, slots=True)
class SapiensVideoPoseConfig:
    """CLI/API configuration for Sapiens2 video pose inference."""

    video_path: Path
    """Path to the input video file."""
    rrd_path: Path
    """Path to the output Rerun recording file."""
    model_size: ModelSize = DEFAULT_MODEL_SIZE
    """Sapiens2 pose model size."""
    keypoint_schema: KeypointSchemaName = "coco133"
    """Keypoint schema to log and expose in outputs."""
    max_frames: int | None = None
    """Optional cap on processed/logged frames after stride filtering."""
    frame_stride: int = 1
    """Process every nth input frame."""
    bbox_thr: float = DEFAULT_BBOX_THR
    """DETR person-detection confidence threshold."""
    nms_thr: float = DEFAULT_NMS_THR
    """Person-box NMS IoU threshold."""
    kpt_thr: float = 0.3
    """Pose keypoint visibility threshold for Rerun rendering."""
    device: DeviceChoice = "auto"
    """Compute device selection; auto prefers CUDA when available."""
    tracking_backend: TrackingBackend = "detr_per_frame"
    """Video person-box source."""
    pose_backend: PoseBackend = "pytorch"
    """Pose heatmap backend used after person boxes are available."""
    tensorrt_engine_path: Path | None = None
    """TensorRT engine path used when pose_backend is tensorrt."""
    sam3_checkpoint: str = DEFAULT_SAM3_CHECKPOINT
    """SAM3 checkpoint used when tracking_backend is sam3_tracking."""
    sam3_min_mask_area_px: int = DEFAULT_SAM3_MIN_MASK_AREA_PX
    """Minimum SAM3 mask area before a propagated track is used for pose."""
    sam3_memory_retention_frames: int = DEFAULT_SAM3_MEMORY_RETENTION_FRAMES
    """Recent SAM3 tracker-state window retained for long videos."""
    sam3_dtype: Sam3DTypeChoice = "bfloat16"
    """Torch dtype for SAM3 inference; half precision falls back to float32 on CPU."""


@dataclass(frozen=True, slots=True)
class VideoPoseSummary:
    """Summary returned after writing a video pose RRD."""

    rrd_path: Path
    """Path to the written Rerun recording."""
    video_metadata: VideoMetadata
    """Metadata probed from the source video."""
    processed_frames: int
    """Number of frames that were processed and logged."""
    keypoint_schema: KeypointSchemaName
    """Keypoint schema used for the output recording."""
    model_size: ModelSize
    """Sapiens2 pose model size used for inference."""
    pose_backend: PoseBackend
    """Pose heatmap backend used for inference."""


@dataclass(slots=True)
class VideoPosePipelineHandle:
    """Mutable result handle populated by the incremental video pipeline."""

    summary: VideoPoseSummary | None = None
    """Final video pose summary after the recording has been written."""


FramePoseFn = Callable[..., PoseEstimation]
BboxPoseFn = Callable[..., PoseEstimation]
DetectPersonsFn = Callable[..., Float32[ndarray, "n 4"]]
LogVideoFn = Callable[..., Int[ndarray, "num_frames"]]
TrackerFactory = Callable[[Sam3TrackerConfig, DeviceChoice], Any]


def _default_tracker_factory(config: Sam3TrackerConfig, device: DeviceChoice) -> Sam3BoxPromptVideoTracker:
    """Create the default SAM3 box-prompt video tracker."""
    return Sam3BoxPromptVideoTracker(config, device=device)


def probe_video(video_path: Path) -> VideoMetadata:
    """Probe video file metadata without loading all frames."""
    cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps: float = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return VideoMetadata(width=width, height=height, fps=fps, total_frames=total_frames)


def iter_video_frames(
    video_path: Path,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[tuple[int, UInt8[ndarray, "h w 3"]]]:
    """Load selected RGB frames from a video.

    Args:
        video_path: Path to a video readable by OpenCV.
        frame_stride: Process every nth frame.
        max_frames: Optional cap on returned frames after stride filtering.

    Returns:
        List of `(frame_idx, frame_rgb)` pairs.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1 when provided.")

    frames: list[tuple[int, UInt8[ndarray, "h w 3"]]] = list(
        iter_video_frame_stream(video_path, frame_stride=frame_stride, max_frames=max_frames)
    )

    if len(frames) == 0:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return frames


def iter_video_frame_stream(
    video_path: Path,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, UInt8[ndarray, "h w 3"]]]:
    """Yield selected RGB frames from a video without loading the whole video."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1 when provided.")

    cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    emitted_frames: int = 0
    frame_idx: int = 0
    try:
        while True:
            ok: bool
            frame_bgr: UInt8[ndarray, "h w 3"]
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride == 0:
                frame_rgb: UInt8[ndarray, "h w 3"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_idx, frame_rgb
                emitted_frames += 1
                if max_frames is not None and emitted_frames >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()


def _selected_frame_count(metadata: VideoMetadata, *, frame_stride: int, max_frames: int | None) -> int:
    """Estimate how many frames will be processed after stride and cap."""
    if metadata.total_frames <= 0:
        return int(max_frames or 0)
    selected_frames: int = (metadata.total_frames + frame_stride - 1) // frame_stride
    if max_frames is not None:
        selected_frames = min(selected_frames, max_frames)
    return selected_frames


def _timestamp_for_frame(
    frame_idx: int,
    frame_timestamps_ns: Int[ndarray, "num_frames"],
    metadata: VideoMetadata,
) -> int | None:
    """Return the best available nanosecond timestamp for a frame index."""
    if frame_idx < frame_timestamps_ns.shape[0]:
        return int(frame_timestamps_ns[frame_idx])
    if metadata.fps > 0.0:
        return int(frame_idx / metadata.fps * 1_000_000_000)
    return None


def _pose_artifact_to_estimation(artifact: PosePredictionArtifact) -> PoseEstimation:
    """Convert a dense pose artifact into the video pipeline's list-backed result."""
    bboxes: Float32[ndarray, "n 4"] = np.asarray(artifact.bboxes, dtype=np.float32).reshape(-1, 4)
    keypoints: list[Float32[ndarray, "sapiens_k 2"]] = [
        np.asarray(artifact.keypoints[idx], dtype=np.float32)
        for idx in range(int(artifact.keypoints.shape[0]))
    ]
    scores: list[Float32[ndarray, "sapiens_k"]] = [
        np.asarray(artifact.scores[idx], dtype=np.float32).reshape(-1)
        for idx in range(int(artifact.scores.shape[0]))
    ]
    return PoseEstimation(bboxes=bboxes, keypoints=keypoints, scores=scores)


def estimate_frame_pose_from_bboxes_tensorrt(
    image_rgb: UInt8[ndarray, "h w 3"],
    bboxes: Float32[ndarray, "n 4"],
    *,
    engine_path: Path,
    model_size: ModelSize = DEFAULT_MODEL_SIZE,
    device: DeviceChoice = "cuda",
    heatmap_runner: TensorRtPoseHeatmapRunner | None = None,
) -> PoseEstimation:
    """Estimate native Sapiens2 pose for externally supplied person boxes with TensorRT."""
    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if bboxes_f32.shape[0] == 0:
        return PoseEstimation(bboxes=bboxes_f32, keypoints=[], scores=[])

    runner: TensorRtPoseHeatmapRunner = heatmap_runner or TensorRtPoseHeatmapRunner(
        engine_path,
        device=device,
    )
    artifact: PosePredictionArtifact = estimate_sapiens_pose_tensorrt(
        image_rgb,
        bboxes_f32,
        engine_path=engine_path,
        model_size=model_size,
        device=device,
        heatmap_runner=runner,
    )
    return _pose_artifact_to_estimation(artifact)


def _make_effective_bbox_pose_fn(
    config: SapiensVideoPoseConfig,
    bbox_pose_fn: BboxPoseFn,
    *,
    tensorrt_bbox_pose_fn: BboxPoseFn | None = None,
) -> BboxPoseFn:
    """Return the box-pose function selected by the video pose backend."""
    if config.pose_backend != "tensorrt":
        return bbox_pose_fn

    if tensorrt_bbox_pose_fn is not None:
        return tensorrt_bbox_pose_fn

    if config.tensorrt_engine_path is None:
        raise ValueError("pose_backend='tensorrt' requires tensorrt_engine_path.")

    heatmap_runner: TensorRtPoseHeatmapRunner = TensorRtPoseHeatmapRunner(
        config.tensorrt_engine_path,
        device=config.device,
    )

    def tensorrt_bbox_pose_fn(
        image_rgb: UInt8[ndarray, "h w 3"],
        bboxes: Float32[ndarray, "n 4"],
        **_kwargs: object,
    ) -> PoseEstimation:
        return estimate_frame_pose_from_bboxes_tensorrt(
            image_rgb,
            bboxes,
            engine_path=config.tensorrt_engine_path,
            model_size=config.model_size,
            device=config.device,
            heatmap_runner=heatmap_runner,
        )

    return tensorrt_bbox_pose_fn


def run_video_pose_pipeline(
    config: SapiensVideoPoseConfig,
    *,
    frame_pose_fn: FramePoseFn = estimate_frame_pose,
    bbox_pose_fn: BboxPoseFn = estimate_frame_pose_from_bboxes,
    tensorrt_bbox_pose_fn: BboxPoseFn | None = None,
    detect_persons_fn: DetectPersonsFn = detect_persons,
    tracker_factory: TrackerFactory = _default_tracker_factory,
    log_video_fn: LogVideoFn = log_video,
    recording: rr.RecordingStream | None = None,
    handle: VideoPosePipelineHandle | None = None,
    save_rrd: bool = True,
) -> Generator[str, None, None]:
    """Run Sapiens2 pose inference on a video, logging incrementally to Rerun.

    Yields:
        Status messages after setup and after each processed frame.
    """
    if not config.video_path.exists():
        raise FileNotFoundError(f"Video not found: {config.video_path}")

    effective_bbox_pose_fn: BboxPoseFn = _make_effective_bbox_pose_fn(
        config,
        bbox_pose_fn,
        tensorrt_bbox_pose_fn=tensorrt_bbox_pose_fn,
    )
    metadata: VideoMetadata = probe_video(config.video_path)
    schema: PoseSchema = get_pose_schema(config.keypoint_schema)
    selected_frame_count: int = _selected_frame_count(
        metadata,
        frame_stride=config.frame_stride,
        max_frames=config.max_frames,
    )

    rec: rr.RecordingStream = recording or rr.RecordingStream(application_id="sapiens2_pose_video", send_properties=True)
    config.rrd_path.parent.mkdir(parents=True, exist_ok=True)

    processed_frames: int = 0
    previous_instance_count: int = 0
    previous_track_ids: set[int] = set()
    previous_segmentation_track_ids: set[int] = set()
    tracker: Any | None = None
    with rec if recording is None else nullcontext(rec):
        rr.send_blueprint(build_video_blueprint(), recording=rec)
        log_annotation_context(schema, recording=rec)
        if config.tracking_backend == "sam3_tracking":
            log_video_segmentation_annotation_context(recording=rec)
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video_fn(
            video_source=config.video_path,
            video_log_path=Path("video/raw"),
            timeline="video_time",
            recording=rec,
        )
        rec.send_property(
            "video_pose",
            rr.AnyValues(
                model_size=config.model_size,
                keypoint_schema=config.keypoint_schema,
                fps=float(metadata.fps),
                num_frames=int(metadata.total_frames),
                selected_frames=int(selected_frame_count),
                width=int(metadata.width),
                height=int(metadata.height),
                tracking_backend=config.tracking_backend,
                pose_backend=config.pose_backend,
            ),
        )
        action: str = "write" if save_rrd else "stream"
        yield f"Loaded video: {selected_frame_count} frame overlays to {action}."

        frame_stream: Iterator[tuple[int, UInt8[ndarray, "h w 3"]]] = iter_video_frame_stream(
            config.video_path,
            frame_stride=config.frame_stride,
            max_frames=config.max_frames,
        )
        for frame_idx, frame_rgb in tqdm(frame_stream, total=selected_frame_count or None, desc="Sapiens2 video pose"):
            timestamp_ns: int | None = _timestamp_for_frame(frame_idx, frame_timestamps_ns, metadata)
            masks_to_log: Bool[ndarray, "n h w"] | None = None
            if config.tracking_backend == "sam3_tracking":
                if tracker is None:
                    initial_bboxes: Float32[ndarray, "n 4"] = detect_persons_fn(
                        frame_rgb,
                        bbox_thr=config.bbox_thr,
                        nms_thr=config.nms_thr,
                        device=config.device,
                    )
                    log_initial_detr_boxes(frame_idx=frame_idx, timestamp_ns=timestamp_ns, bboxes=initial_bboxes, recording=rec)
                    next_frame_idx: int = frame_idx + 1
                    clear_initial_detr_boxes(
                        frame_idx=next_frame_idx,
                        timestamp_ns=_timestamp_for_frame(next_frame_idx, frame_timestamps_ns, metadata),
                        recording=rec,
                    )
                    tracker_config: Sam3TrackerConfig = Sam3TrackerConfig(
                        checkpoint=config.sam3_checkpoint,
                        min_mask_area_px=config.sam3_min_mask_area_px,
                        memory_retention_frames=config.sam3_memory_retention_frames,
                        dtype=config.sam3_dtype,
                    )
                    tracker = tracker_factory(tracker_config, config.device)
                    tracked: TrackedDetections = tracker.start(frame_idx=frame_idx, frame_rgb=frame_rgb, initial_bboxes=initial_bboxes)
                else:
                    tracked = tracker.track(frame_idx=frame_idx, frame_rgb=frame_rgb)

                pose_result = effective_bbox_pose_fn(
                    frame_rgb,
                    tracked.bboxes,
                    model_size=config.model_size,
                    device=config.device,
                )
                track_ids: Int[ndarray, "n"] = tracked.track_ids
                masks_to_log = tracked.masks
            else:
                if config.pose_backend == "tensorrt":
                    frame_bboxes: Float32[ndarray, "n 4"] = detect_persons_fn(
                        frame_rgb,
                        bbox_thr=config.bbox_thr,
                        nms_thr=config.nms_thr,
                        device=config.device,
                    )
                    pose_result = effective_bbox_pose_fn(
                        frame_rgb,
                        frame_bboxes,
                        model_size=config.model_size,
                        device=config.device,
                    )
                else:
                    pose_result = frame_pose_fn(
                        frame_rgb,
                        model_size=config.model_size,
                        bbox_thr=config.bbox_thr,
                        nms_thr=config.nms_thr,
                        device=config.device,
                    )
                instance_count: int = int(pose_result.bboxes.shape[0])
                track_ids = np.arange(instance_count, dtype=np.int32)

            keypoints: list[Float32[ndarray, "schema_k 2"]]
            scores: list[Float32[ndarray, "schema_k"]]
            keypoints, scores = project_instances_to_schema(
                pose_result.keypoints,
                pose_result.scores,
                config.keypoint_schema,
            )
            current_instance_count: int = min(int(pose_result.bboxes.shape[0]), len(keypoints), len(scores), int(track_ids.shape[0]))
            keypoints_to_log: list[Float32[ndarray, "schema_k 2"]] = keypoints[:current_instance_count]
            scores_to_log: list[Float32[ndarray, "schema_k"]] = scores[:current_instance_count]
            track_ids_to_log: Int[ndarray, "m"] = track_ids[:current_instance_count]
            log_video_pose_frame(
                frame_idx=frame_idx,
                timestamp_ns=timestamp_ns,
                bboxes=pose_result.bboxes,
                keypoints=keypoints_to_log,
                scores=scores_to_log,
                schema=schema,
                kpt_thr=config.kpt_thr,
                previous_instance_count=previous_instance_count,
                track_ids=track_ids,
                previous_track_ids=previous_track_ids,
                previous_segmentation_track_ids=previous_segmentation_track_ids if masks_to_log is not None else None,
                masks=masks_to_log,
                recording=rec,
            )
            previous_instance_count = current_instance_count
            previous_track_ids = {int(track_id) for track_id in track_ids_to_log.tolist()}
            if masks_to_log is not None:
                segmentation_count: int = min(int(track_ids.shape[0]), int(pose_result.bboxes.shape[0]), int(masks_to_log.shape[0]))
                previous_segmentation_track_ids = {int(track_id) for track_id in track_ids[:segmentation_count].tolist()}
            processed_frames += 1
            if selected_frame_count > 0:
                if config.tracking_backend == "sam3_tracking":
                    yield f"Frame {processed_frames}/{selected_frame_count}: {current_instance_count} poses, {len(previous_segmentation_track_ids)} SAM3 tracks"
                else:
                    yield f"Frame {processed_frames}/{selected_frame_count}"
            else:
                if config.tracking_backend == "sam3_tracking":
                    yield f"Frame {processed_frames}: {current_instance_count} poses, {len(previous_segmentation_track_ids)} SAM3 tracks"
                else:
                    yield f"Frame {processed_frames}"

    if processed_frames == 0:
        raise RuntimeError(f"No frames loaded from {config.video_path}")

    if save_rrd:
        save_recording(rec, config.rrd_path)
        summary: VideoPoseSummary = VideoPoseSummary(
            rrd_path=config.rrd_path,
            video_metadata=metadata,
            processed_frames=processed_frames,
            keypoint_schema=config.keypoint_schema,
            model_size=config.model_size,
            pose_backend=config.pose_backend,
        )
        if handle is not None:
            handle.summary = summary
        yield f"Complete: {processed_frames} frame overlays written to {config.rrd_path.name}."
        return

    yield f"Complete: {processed_frames} frame overlays streamed."


def write_video_pose_rrd(
    config: SapiensVideoPoseConfig,
    *,
    frame_pose_fn: FramePoseFn = estimate_frame_pose,
    bbox_pose_fn: BboxPoseFn = estimate_frame_pose_from_bboxes,
    tensorrt_bbox_pose_fn: BboxPoseFn | None = None,
    detect_persons_fn: DetectPersonsFn = detect_persons,
    tracker_factory: TrackerFactory = _default_tracker_factory,
    log_video_fn: LogVideoFn = log_video,
) -> VideoPoseSummary:
    """Run Sapiens2 pose inference on a video and write an annotated RRD."""
    handle: VideoPosePipelineHandle = VideoPosePipelineHandle()
    for _status in run_video_pose_pipeline(
        config,
        frame_pose_fn=frame_pose_fn,
        bbox_pose_fn=bbox_pose_fn,
        tensorrt_bbox_pose_fn=tensorrt_bbox_pose_fn,
        detect_persons_fn=detect_persons_fn,
        tracker_factory=tracker_factory,
        log_video_fn=log_video_fn,
        handle=handle,
    ):
        pass
    if handle.summary is None:
        raise RuntimeError("Video pose pipeline finished without a summary.")
    return handle.summary
