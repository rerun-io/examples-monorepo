"""
Text-prompt SAM3 video demo with chunk-based inference.

This module provides memory-efficient video segmentation by processing in chunks:
- Loads video in chunks of ~30 seconds (configurable based on resolution).
- Uses overlapping boundaries to preserve hotstart quality benefits.
- Uses SAM3's propagate_in_video_iterator within each chunk.
- Logs results to Rerun for interactive visualization.

This is a middle ground between:
- video_batch.py (full video in RAM - best quality, highest memory)
- video_stream.py (frame-by-frame - lowest memory, reduced temporal context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import cv2
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Int, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from simplecv.video_io import TorchCodecVideoReader
from torch import Tensor
from tqdm import tqdm
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

from sam3.api.video_logging import log_frame_outputs
from sam3.viz_constants import BOX_PALETTE, SEG_OVERLAY_ALPHA

if TYPE_CHECKING:
    from transformers.models.sam3_video.processing_sam3_video import Sam3VideoInferenceSession

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

"""Rerun draw order for bounding boxes (above segmentation)."""

DEFAULT_CHUNK_MEMORY_GB: float = 4.0
"""Default target memory per chunk in GB. Actual chunk size adapts to video resolution."""

HOTSTART_OVERLAP_FRAMES: int = 10
"""Number of overlap frames between chunks to preserve hotstart quality benefits."""


# ──────────────────────────────────────────────────────────────────────────────
# Video Metadata
# ──────────────────────────────────────────────────────────────────────────────


class VideoMetadata(NamedTuple):
    """Video file metadata from OpenCV probe."""

    width: int
    height: int
    fps: float
    total_frames: int
    bytes_per_frame: int


def probe_video(video_path: Path) -> VideoMetadata:
    """
    Probe video file for metadata without loading frames.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoMetadata containing resolution, fps, and frame count.

    Raises:
        RuntimeError: If video cannot be opened.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    bytes_per_frame: int = width * height * 3  # RGB uint8
    return VideoMetadata(width, height, fps, total_frames, bytes_per_frame)


def compute_chunk_size(meta: VideoMetadata, target_memory_gb: float) -> int:
    """
    Compute optimal chunk size in frames based on target memory budget.

    Args:
        meta: Video metadata from probe_video.
        target_memory_gb: Target memory budget per chunk in GB.

    Returns:
        Number of frames per chunk.
    """
    target_bytes: int = int(target_memory_gb * (1024**3))
    chunk_frames: int = max(1, target_bytes // meta.bytes_per_frame)
    return chunk_frames


# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
"""Mapping from string dtype names to torch.dtype."""


def safe_dtype(dtype_pref: str, device: torch.device) -> torch.dtype:
    """Return dtype, falling back to float32 on CPU for half-precision types."""
    dtype: torch.dtype = DTYPE_MAP[dtype_pref]
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


@dataclass(slots=True)
class Sam3VideoModelConfig:
    """
    Checkpoint, device, and dtype settings for the SAM3 video model.

    Controls compute placement and detection thresholds for chunk-based inference.
    """

    checkpoint: str = "facebook/sam3"
    """Model identifier passed to ``Sam3VideoModel.from_pretrained``."""

    device: Literal["auto", "cpu", "cuda"] = "auto"
    """Compute device selection; ``auto`` prefers CUDA when available."""

    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    """Torch dtype used for model weights and inference."""

    processing_device: Literal["cpu", "cuda"] = "cpu"
    """Device used by the processor for per-frame preprocessing."""

    video_storage_device: Literal["cpu", "cuda"] = "cpu"
    """Device that caches video tensors within the inference session."""

    score_threshold_detection: float = 0.85
    """Minimum detection score before tracking."""

    new_det_thresh: float | None = None
    """Optional stricter threshold for new detections (falls back to detection threshold if None)."""


@dataclass(slots=True)
class Sam3ChunkVideoDemoConfig:
    """
    CLI options for running text-prompt SAM3 video segmentation with chunk-based inference.

    This is the top-level configuration passed to ``main()``.
    """

    video_path: Path = Path()
    """Path to the input video file (any format supported by OpenCV)."""

    prompt: str = "person"
    """Text concept to detect and track across the video."""

    max_frames: int | None = None
    """Optional cap on total frames to process across all chunks."""

    chunk_memory_gb: float = DEFAULT_CHUNK_MEMORY_GB
    """Target memory per chunk in GB. Chunk size in frames is computed from this and video resolution."""

    overlap_frames: int = HOTSTART_OVERLAP_FRAMES
    """Number of overlap frames between chunks for hotstart quality preservation."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer/runtime options for Rerun (window layout, recording, etc.)."""

    model_config: Sam3VideoModelConfig = field(default_factory=Sam3VideoModelConfig)
    """Checkpoint, device, and dtype settings for the SAM3 video model."""


# ──────────────────────────────────────────────────────────────────────────────
# Rerun Visualization Helpers
# ──────────────────────────────────────────────────────────────────────────────


def log_annotation_context() -> None:
    """
    Register annotation context with background + instance color palette.

    This sets up class IDs for segmentation masks and bounding boxes,
    using colors from ``BOX_PALETTE``.
    """
    class_descriptions: list[rr.ClassDescription] = [
        rr.ClassDescription(
            info=rr.AnnotationInfo(id=0, label="Background", color=(64, 64, 64, 0))
        )
    ]
    for idx, color_rgb in enumerate(BOX_PALETTE[:, :3].tolist(), start=1):
        color_rgba: tuple[int, int, int, int] = (
            int(color_rgb[0]),
            int(color_rgb[1]),
            int(color_rgb[2]),
            SEG_OVERLAY_ALPHA,
        )
        class_descriptions.append(
            rr.ClassDescription(info=rr.AnnotationInfo(id=idx, label=f"Object-{idx}", color=color_rgba))
        )
    rr.log("/", rr.AnnotationContext(class_descriptions), static=True)


def build_blueprint() -> rrb.Blueprint:
    """
    Create a Rerun blueprint with a 2D view for video + segmentation overlay.

    Returns:
        Blueprint configured to show video, segmentation, and boxes.
    """
    view: rrb.Spatial2DView = rrb.Spatial2DView(
        name="Video + Segmentation",
        contents=["video/raw", "video/segmentation", "video/boxes"],
    )
    return rrb.Blueprint(view, collapse_panels=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_sam3_model(
    config: Sam3VideoModelConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Sam3VideoModel, Sam3VideoProcessor]:
    """
    Load SAM3 video model and processor from checkpoint.

    Args:
        config: Model configuration with checkpoint and thresholds.
        device: Target compute device.
        dtype: Target dtype for model weights.

    Returns:
        Tuple of (model, processor) ready for inference.
    """
    model_cfg: Sam3VideoConfig = Sam3VideoConfig.from_pretrained(
        config.checkpoint,
        score_threshold_detection=config.score_threshold_detection,
        new_det_thresh=config.new_det_thresh or config.score_threshold_detection,
    )
    model: Sam3VideoModel = Sam3VideoModel.from_pretrained(
        config.checkpoint, config=model_cfg
    ).to(device=device, dtype=dtype)

    processor: Sam3VideoProcessor = Sam3VideoProcessor.from_pretrained(config.checkpoint)

    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────


def main(cfg: Sam3ChunkVideoDemoConfig) -> None:
    """
    Run text-prompt SAM3 video segmentation using chunk-based inference.

    This function:
    1. Loads the SAM3 model once.
    2. Probes video to compute optimal chunk size based on memory budget.
    3. Processes video in overlapping chunks using propagate_in_video_iterator.
    4. Logs results to Rerun for visualization.

    Memory usage is O(chunk_size) instead of O(video_length).

    Args:
        cfg: Full configuration including video path, prompt, and chunking settings.

    Raises:
        FileNotFoundError: If the video file does not exist.
    """
    if not cfg.video_path.exists():
        raise FileNotFoundError(f"Video not found: {cfg.video_path}")

    # Probe video metadata
    meta: VideoMetadata = probe_video(cfg.video_path)

    # Compute chunk size based on memory budget
    chunk_frames: int = compute_chunk_size(meta, cfg.chunk_memory_gb)
    chunk_duration: float = chunk_frames / meta.fps

    # Resolve device and dtype
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = safe_dtype(cfg.model_config.dtype, device)

    # One NVDEC reader for the whole run: each chunk is an exact frame range on the GPU, where the
    # processor takes it, instead of a new cv2 capture, an inexact seek and a host colour conversion.
    # Its frame count comes from the stream index, not the container header cv2 reports.
    reader: TorchCodecVideoReader = TorchCodecVideoReader(cfg.video_path, device=device, seek_mode="exact")

    # Apply max_frames limit
    total_frames: int = len(reader)
    if cfg.max_frames is not None:
        total_frames = min(total_frames, cfg.max_frames)

    print(f"[Chunk config] {meta.width}x{meta.height} @ {meta.fps:.1f}fps")
    print(f"[Chunk config] Total: {total_frames} frames, Chunk: {chunk_frames} frames (~{chunk_duration:.1f}s)")
    print(f"[Chunk config] Overlap: {cfg.overlap_frames} frames")

    # Load model and processor once
    model, processor = load_sam3_model(cfg.model_config, device, dtype)

    # Setup Rerun visualization
    rr.send_blueprint(build_blueprint())
    log_annotation_context()

    # Log video asset once (much more efficient than per-frame images)
    frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        video_source=cfg.video_path,
        video_log_path=Path("video/raw"),
        timeline="video_time",
    )

    # Compute chunk boundaries with overlap
    # Step size is chunk_frames - overlap to create overlapping windows
    step_size: int = max(1, chunk_frames - cfg.overlap_frames)
    chunk_starts: list[int] = list(range(0, total_frames, step_size))

    # Track which frames we've already yielded (to skip overlap frames)
    frames_yielded: set[int] = set()
    total_processed: int = 0

    pbar: tqdm = tqdm(total=total_frames, desc="Processing chunks")

    with torch.inference_mode():
        for _chunk_idx, chunk_start in enumerate(chunk_starts):
            # Calculate actual chunk end
            chunk_end: int = min(chunk_start + chunk_frames, total_frames)
            actual_chunk_size: int = chunk_end - chunk_start

            if actual_chunk_size <= 0:
                break

            chunk_video: UInt8[Tensor, "t 3 h w"] = reader.get_frames_in_range(chunk_start, chunk_end)

            # Initialize inference session for this chunk
            inference_session: Sam3VideoInferenceSession = processor.init_video_session(
                video=chunk_video,
                inference_device=device,
                processing_device=cfg.model_config.processing_device,
                video_storage_device=cfg.model_config.video_storage_device,
                dtype=dtype,
            )
            processor.add_text_prompt(inference_session=inference_session, text=cfg.prompt)

            # Process chunk using propagate_in_video_iterator
            for model_outputs in model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=actual_chunk_size,
            ):
                # Convert chunk-local frame index to global frame index
                local_frame_idx: int = int(model_outputs.frame_idx)
                global_frame_idx: int = chunk_start + local_frame_idx

                # Skip frames we've already yielded (overlap from previous chunk)
                if global_frame_idx in frames_yielded:
                    continue

                # Skip if beyond max_frames
                if global_frame_idx >= total_frames:
                    continue

                frames_yielded.add(global_frame_idx)

                # Postprocess and log
                processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
                log_frame_outputs(
                    global_frame_idx,
                    frame_timestamps_ns,
                    processed_outputs,
                    frame_shape=(chunk_video.shape[2], chunk_video.shape[3]),
                )

                pbar.update(1)
                total_processed += 1

            # Free chunk memory
            del chunk_video
            del inference_session

    reader.close()
    pbar.close()

    # Summary
    print(f"[done] Processed {total_processed} frames in {len(chunk_starts)} chunks (prompt='{cfg.prompt}').")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__: list[str] = [
    "Sam3VideoModelConfig",
    "Sam3ChunkVideoDemoConfig",
    "main",
]
