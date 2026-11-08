"""
Text-prompt SAM3 video demo with streaming frame processing.

This module provides a memory-efficient video segmentation pipeline that:
- Streams frames from disk to minimize RAM usage (no full video buffering).
- Uses SAM3's text-prompt interface for zero-shot object detection and tracking.
- Logs results to Rerun for interactive visualization.

Uses OpenCV's VideoCapture (backed by FFmpeg) for broad codec support.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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


# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Sam3VideoModelConfig:
    """
    Checkpoint, device, and dtype settings for the SAM3 video model.

    Controls compute placement and detection thresholds for streaming inference.
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
class Sam3StreamDemoConfig:
    """
    CLI options for running text-prompt SAM3 video segmentation in streaming mode.

    This is the top-level configuration passed to ``main()``.
    """

    video_path: Path = Path()
    """Path to the input video file (any format supported by OpenCV or FFmpeg)."""

    prompt: str = "person"
    """Text concept to detect and track across the video."""

    max_frames: int | None = None
    """Optional cap on the number of frames to decode and propagate."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer/runtime options for Rerun (window layout, recording, etc.)."""

    model_config: Sam3VideoModelConfig = field(default_factory=Sam3VideoModelConfig)
    """Checkpoint, device, and dtype settings for the SAM3 video model."""


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


# ──────────────────────────────────────────────────────────────────────────────
# Video Frame Iteration (Streaming)
# ──────────────────────────────────────────────────────────────────────────────

FrameData = tuple[int, UInt8[Tensor, "3 h w"], float, int, int]
"""Type alias for frame iteration: (frame_idx, rgb_chw, fps, width, height)."""


def iter_video_frames(path: Path, device: torch.device) -> Generator[FrameData, None, None]:
    """
    Yield RGB frames from a video file, decoded in small batches on ``device`` (NVDEC on CUDA).

    The processor preprocesses each frame on the inference device, so the frames stay there
    instead of taking a host colour conversion and an upload each.

    Args:
        path: Path to the video file.
        device: Decode device.

    Yields:
        Tuple of (frame_idx, rgb_chw, fps, width, height) for each frame.
    """
    reader: TorchCodecVideoReader = TorchCodecVideoReader(path, device=device)
    try:
        for frame_idx, frame_rgb in enumerate(reader):
            yield frame_idx, frame_rgb, reader.fps, reader.width, reader.height
    finally:
        reader.close()


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


def main(cfg: Sam3StreamDemoConfig) -> None:
    """
    Run text-prompt SAM3 video segmentation using streaming inference.

    This function:
    1. Loads the SAM3 model and initializes an inference session.
    2. Streams video frames from disk (constant memory usage).
    3. Runs detection + tracking on each frame using the text prompt.
    4. Logs results to Rerun for visualization.

    Args:
        cfg: Full configuration including video path, prompt, and model settings.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video contains no frames.
    """
    if not cfg.video_path.exists():
        raise FileNotFoundError(f"Video not found: {cfg.video_path}")

    # Resolve device and dtype
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = safe_dtype(cfg.model_config.dtype, device)

    # Load model and processor
    model, processor = load_sam3_model(cfg.model_config, device, dtype)

    # Initialize frame iterator
    frame_iter: Generator[FrameData, None, None] = iter_video_frames(cfg.video_path, device)

    try:
        # Get first frame to validate video
        first: FrameData | None = next(frame_iter, None)
        if first is None:
            raise RuntimeError(f"Video contains no frames: {cfg.video_path}")

        first_idx, first_frame, fps, width, height = first

        # Initialize inference session
        inference_session: Sam3VideoInferenceSession = processor.init_video_session(
            video=None,
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

        # Frame processing function
        def process_frame(frame_idx: int, frame_rgb: UInt8[Tensor, "3 h w"]) -> None:
            """Process a single frame through the model and log to Rerun."""
            inputs = processor(images=frame_rgb, device=device, return_tensors="pt")
            model_outputs = model(
                inference_session=inference_session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )
            processed_outputs = processor.postprocess_outputs(
                inference_session,
                model_outputs,
                original_sizes=inputs.original_sizes,
            )
            log_frame_outputs(
                frame_idx,
                frame_timestamps_ns,
                processed_outputs,
                frame_shape=(frame_rgb.shape[1], frame_rgb.shape[2]),
            )

        # Run streaming inference
        total_frames: int = 0
        with torch.inference_mode():
            progress_total: int | None = cfg.max_frames
            pbar: tqdm = tqdm(total=progress_total, desc="Streaming masks")

            # Process first frame
            process_frame(first_idx, first_frame)
            pbar.update(1)
            total_frames += 1

            # Process remaining frames
            for frame_idx, frame_rgb, *_ in frame_iter:
                if cfg.max_frames is not None and frame_idx >= cfg.max_frames:
                    break
                process_frame(frame_idx, frame_rgb)
                pbar.update(1)
                total_frames += 1

            pbar.close()

        # Summary
        fps_msg: str = f" @ {fps:.2f} fps" if fps else ""
        print(f"[done] Stream-processed {total_frames} frames{fps_msg} (prompt='{cfg.prompt}').")

    finally:
        if hasattr(frame_iter, "close"):
            frame_iter.close()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__: list[str] = [
    "Sam3VideoModelConfig",
    "Sam3StreamDemoConfig",
    "main",
]
