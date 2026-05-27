"""Benchmark iterable PyTorch and batched TensorRT COCO-133 pose pipelines."""
# ruff: noqa: I001

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import torch
from sapiens2_pose.api.pose_artifact import PoseArtifactComparisonConfig, PoseArtifactComparisonMetrics
from simplecv.rerun_log_utils import RerunTyroConfig

from sapiens_coco133_pose.api.artifact import Coco133PoseFrame, compare_video_pose_rrds, write_video_pose_rrd
from sapiens_coco133_pose.api.batched_tensorrt import (
    AnnotatedPoseTensorRtConfig,
    BatchedTensorRtVideoConfig,
    BatchedTensorRtVideoSummary,
    SapiensTensorRtPoseConfig,
    TensorRtDetectorConfig,
    TensorRtRuntimeConfig,
    create_batched_tensorrt_video_frame_predictor,
)
from sapiens_coco133_pose.api.iterable import IterableVideoPoseConfig, IterableVideoPoseSummary, iter_video_pose_coco133


@dataclass(frozen=True, slots=True)
class PoseBenchmarkConfig:
    """Configuration for benchmarking iterable versus batched TensorRT pose."""

    video_path: Path
    """Input video path."""
    detector: TensorRtDetectorConfig = field(default_factory=TensorRtDetectorConfig)
    """TensorRT detector engine configuration."""
    pose: AnnotatedPoseTensorRtConfig = field(default_factory=SapiensTensorRtPoseConfig)
    """TensorRT pose backend configuration selected by Tyro subcommand."""
    runtime: TensorRtRuntimeConfig = field(default_factory=TensorRtRuntimeConfig)
    """Runtime batch sizes and detector postprocessing thresholds."""
    output_dir: Path = Path("/tmp/sapiens-coco133-benchmark")
    """Directory where iterable and batched RRD artifacts are written."""
    max_frames: int | None = 100
    """Maximum number of frames to benchmark. If None, benchmark the full video."""
    keypoint_threshold: float = 0.3
    """Confidence threshold used for visible keypoints during RRD comparison."""
    required_speedup: float = 10.0
    """Minimum required batched TensorRT speedup over the iterable path."""
    max_relative_difference: float = 10.0
    """Maximum allowed relative numeric difference for RRD comparison."""
    max_keypoint_bbox_fraction: float = 0.06
    """Maximum allowed keypoint difference as a fraction of bbox size."""


class PoseBenchmarkSummary(NamedTuple):
    """Summary returned after benchmarking iterable and batched TensorRT pose paths."""

    iterable_summary: IterableVideoPoseSummary
    """Iterable path artifact summary."""
    batched_summary: BatchedTensorRtVideoSummary
    """Batched TensorRT path artifact summary."""
    iterable_seconds: float
    """Timed iterable inference seconds."""
    batched_seconds: float
    """Timed batched TensorRT inference seconds."""
    iterable_fps: float
    """Iterable inference frames per second."""
    batched_fps: float
    """Batched TensorRT inference frames per second."""
    speedup: float
    """Batched FPS divided by iterable FPS."""
    passes_speed_target: bool
    """Whether ``speedup`` met ``required_speedup``."""
    comparison_metrics: PoseArtifactComparisonMetrics
    """RRD comparison metrics between iterable and batched artifacts."""


class BenchmarkArtifactCounts(NamedTuple):
    """Counts returned after writing an RRD benchmark artifact."""

    frame_count: int
    """Number of frames written to the artifact."""
    person_count: int
    """Number of person rows written to the artifact."""


def run_pose_benchmark(config: PoseBenchmarkConfig) -> PoseBenchmarkSummary:
    """Benchmark iterable and batched TensorRT COCO-133 pose inference.

    Args:
        config: Benchmark input video, TensorRT configs, runtime settings, and accuracy/speed thresholds.

    Returns:
        Inference timings, FPS values, speedup status, artifact summaries, and RRD comparison metrics.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = "" if config.pose.backend == "sapiens" else f"{config.pose.backend}_"
    iterable_rrd_path = config.output_dir / f"{artifact_prefix}iterable_golden.rrd"
    batched_rrd_path = config.output_dir / f"{artifact_prefix}batched_tensorrt.rrd"
    iterable_config = IterableVideoPoseConfig(
        video_path=config.video_path,
        rrd_path=iterable_rrd_path,
        max_frames=config.max_frames,
        pose_backend=config.pose.backend,
        keypoint_threshold=config.keypoint_threshold,
        log_images=False,
    )
    batched_config = BatchedTensorRtVideoConfig(
        rr_config=RerunTyroConfig(application_id="sapiens_coco133_pose_benchmark", headless=True),
        video_path=config.video_path,
        detector=config.detector,
        pose=config.pose,
        runtime=config.runtime,
        max_frames=config.max_frames,
        keypoint_threshold=config.keypoint_threshold,
    )
    _synchronize_cuda()
    start = time.perf_counter()
    iterable_frames = list(iter_video_pose_coco133(iterable_config))
    _synchronize_cuda()
    iterable_seconds = time.perf_counter() - start
    if not iterable_frames:
        raise ValueError("Iterable benchmark produced no frames; cannot compute FPS or compare artifacts.")
    iterable_counts = _write_benchmark_artifacts(iterable_frames, rrd_path=iterable_rrd_path, keypoint_threshold=config.keypoint_threshold)
    iterable_summary = IterableVideoPoseSummary(iterable_rrd_path, iterable_counts.frame_count, iterable_counts.person_count)
    _release_iterable_gpu_caches()
    predictor = create_batched_tensorrt_video_frame_predictor(batched_config)
    _synchronize_cuda()
    start = time.perf_counter()
    batched_frames = predictor.predict_frames()
    _synchronize_cuda()
    batched_seconds = time.perf_counter() - start
    if not batched_frames:
        raise ValueError("Batched TensorRT benchmark produced no frames; cannot compute FPS or compare artifacts.")
    batched_counts = _write_benchmark_artifacts(batched_frames, rrd_path=batched_rrd_path, keypoint_threshold=config.keypoint_threshold)
    batched_summary = BatchedTensorRtVideoSummary(batched_rrd_path, batched_counts.frame_count, batched_counts.person_count)
    comparison = compare_video_pose_rrds(
        iterable_rrd_path,
        batched_rrd_path,
        PoseArtifactComparisonConfig(
            config.max_relative_difference,
            keypoint_score_threshold=config.keypoint_threshold,
            max_keypoint_bbox_fraction=config.max_keypoint_bbox_fraction,
        ),
    )
    frame_count = iterable_summary.frame_count
    iterable_fps, batched_fps = frame_count / max(iterable_seconds, 1e-9), frame_count / max(batched_seconds, 1e-9)
    speedup = batched_fps / max(iterable_fps, 1e-9)
    return PoseBenchmarkSummary(
        iterable_summary,
        batched_summary,
        iterable_seconds,
        batched_seconds,
        iterable_fps,
        batched_fps,
        speedup,
        speedup >= config.required_speedup,
        comparison,
    )


def _write_benchmark_artifacts(frames: list[Coco133PoseFrame], *, rrd_path: Path, keypoint_threshold: float) -> BenchmarkArtifactCounts:
    """Write benchmark RRD outside timed inference.

    Args:
        frames: Pose frames to serialize.
        rrd_path: Destination RRD path.
        keypoint_threshold: Confidence threshold used for visible keypoints.

    Returns:
        Frame and person counts written to the RRD.
    """
    write_video_pose_rrd(frames, rrd_path=rrd_path, keypoint_threshold=keypoint_threshold)
    return BenchmarkArtifactCounts(len(frames), sum(int(frame.bboxes.shape[0]) for frame in frames))


def _synchronize_cuda() -> None:
    """Synchronize CUDA if available so timing includes queued GPU work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _release_iterable_gpu_caches() -> None:
    """Release cached PyTorch baseline models before constructing TensorRT engines."""
    import sapiens2_pose.api.runtime as sapiens_runtime

    sapiens_runtime.clear_pose_model_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
