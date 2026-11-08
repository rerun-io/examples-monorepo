"""
Text-prompt SAM3 video demo with batch-mode inference.

This module provides video segmentation using SAM3's propagate_in_video_iterator:
- Loads entire video into memory for batch processing.
- Uses SAM3's text-prompt interface for zero-shot object detection and tracking.
- Logs results to Rerun for interactive visualization.

For memory-constrained scenarios with long videos, use video_stream.py instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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

MAX_VIDEO_MEMORY_GB: float = 4.0
"""Maximum estimated raw video memory (GB) for batch loading. Larger videos should use video_stream.py."""


# ──────────────────────────────────────────────────────────────────────────────
# Video Checks
# ──────────────────────────────────────────────────────────────────────────────


def check_video_constraints(video_path: Path, max_frames: int | None = None) -> None:
    """
    Check if video is too large for batch loading based on estimated memory usage.

    Memory is estimated as: frames x width x height x 3 bytes (RGB uint8).

    Args:
        video_path: Path to the video file.
        max_frames: Optional cap on frame count (uses this for memory estimate if set).

    Raises:
        RuntimeError: If video cannot be opened.
        MemoryError: If estimated video memory exceeds MAX_VIDEO_MEMORY_GB.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Use max_frames if specified
    effective_frames: int = min(total_frames, max_frames) if max_frames else total_frames
    duration_seconds: float = total_frames / fps

    # Estimate raw memory: frames x width x height x 3 (RGB uint8)
    bytes_per_frame: int = width * height * 3
    estimated_bytes: int = bytes_per_frame * effective_frames
    estimated_gb: float = estimated_bytes / (1024**3)

    if estimated_gb > MAX_VIDEO_MEMORY_GB:
        raise MemoryError(
            f"Video too large for batch loading.\n"
            f"  Resolution:  {width}x{height}\n"
            f"  Duration:    {duration_seconds:.1f}s ({effective_frames} frames)\n"
            f"  Est. memory: {estimated_gb:.1f} GB (max: {MAX_VIDEO_MEMORY_GB:.0f} GB)\n"
            f"  Suggestion:  Use video_stream.py for streaming inference (constant memory),\n"
            f"               or use --max-frames to limit frames loaded."
        )

    print(f"[Video check] {width}x{height}, {duration_seconds:.1f}s, {effective_frames} frames (~{estimated_gb:.1f} GB)")


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

    Controls compute placement and detection thresholds for batch inference.
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
class Sam3VideoDemoConfig:
    """
    CLI options for running text-prompt SAM3 video segmentation with batch inference.

    This is the top-level configuration passed to ``main()``.
    """

    video_path: Path = Path()
    """Path to the input video file (any format supported by transformers ``load_video``)."""

    prompt: str = "person"
    """Text concept to detect and track across the video."""

    max_frames: int | None = None
    """Optional cap on the number of frames to load and propagate."""

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


def main(cfg: Sam3VideoDemoConfig) -> None:
    """
    Run text-prompt SAM3 video segmentation using batch inference.

    This function:
    1. Loads the SAM3 model and the entire video into memory.
    2. Initializes an inference session with the video tensor.
    3. Propagates masks through the video using text prompts.
    4. Logs results to Rerun for visualization.

    For long videos, consider using video_stream.py which has constant memory usage.

    Args:
        cfg: Full configuration including video path, prompt, and model settings.

    Raises:
        FileNotFoundError: If the video file does not exist.
    """
    if not cfg.video_path.exists():
        raise FileNotFoundError(f"Video not found: {cfg.video_path}")

    # Resolve device and dtype
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = safe_dtype(cfg.model_config.dtype, device)

    # Load model and processor
    model, processor = load_sam3_model(cfg.model_config, device, dtype)

    # Check video constraints (duration + resolution)
    check_video_constraints(cfg.video_path, cfg.max_frames)

    # Decode the first ``max_frames`` frames (all by default) on NVDEC, into one device tensor the
    # processor preprocesses where it lies. Session frame i is then video frame i, which the logged
    # VideoStream's timestamps assume; transformers' load_video(num_frames=N) sampled N frames spread
    # over the whole clip and put the masks on the wrong frames.
    reader: TorchCodecVideoReader = TorchCodecVideoReader(cfg.video_path, device=device)
    video_fps: float = reader.fps
    total_frames: int = len(reader) if cfg.max_frames is None else min(cfg.max_frames, len(reader))
    video_frames: UInt8[Tensor, "t 3 h w"] = reader.get_frames_in_range(0, total_frames)
    reader.close()

    # Initialize inference session with full video
    inference_session: Sam3VideoInferenceSession = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device=cfg.model_config.processing_device,
        video_storage_device=cfg.model_config.video_storage_device,
        dtype=dtype,
    )
    processor.add_text_prompt(inference_session=inference_session, text=cfg.prompt)

    # Setup Rerun visualization
    rr.send_blueprint(build_blueprint())
    log_annotation_context()

    # Log video asset once (much more efficient than per-frame images)
    frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        video_source=cfg.video_path,
        video_log_path=Path("video/raw"),
        timeline="video_time",
    )

    # Run batch inference using propagate_in_video_iterator
    with torch.inference_mode():
        for model_outputs in tqdm(
            model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=cfg.max_frames,
            ),
            total=total_frames,
            desc="Propagating masks",
        ):
            frame_idx: int = int(model_outputs.frame_idx)
            processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
            log_frame_outputs(
                frame_idx,
                frame_timestamps_ns,
                processed_outputs,
                frame_shape=(video_frames.shape[2], video_frames.shape[3]),
            )

    # Summary
    print(f"[done] Processed {total_frames} frames @ {video_fps:.2f} fps (prompt='{cfg.prompt}').")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__: list[str] = [
    "Sam3VideoModelConfig",
    "Sam3VideoDemoConfig",
    "main",
]
