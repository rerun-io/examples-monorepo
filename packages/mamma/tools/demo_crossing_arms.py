"""Streaming pipeline demo on a MAMMA-format capture (crossing_arms by default).

M1: NVDEC 4K decode -> GPU resize to 720p -> NVENC H.264 -> per-camera Rerun
VideoStreams + posed pinholes. M2: + causal person tracking (YOLO bootstrap,
CLIP+epipolar identity, streaming SAM2 masks). Landmarks/fitting land in M3+.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro
from simplecv.rerun_log_utils import RerunTyroConfig

from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import PipelineStats, StreamingPipeline, build_streaming_pipeline
from mamma.fitting.window_fitter import FitterConfig
from mamma.tracking.tracker import TrackerConfig


@dataclass
class DemoConfig:
    rr_config: RerunTyroConfig
    """Rerun viewer/save/connect behavior."""
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    """Streaming tracker settings (SAM2/YOLO checkpoints, re-detect cadence)."""
    fitter: FitterConfig = field(default_factory=FitterConfig)
    """Sliding-window SMPL-X fitter settings."""
    data_dir: Path = Path("data/inputs/indoors/crossing_arms")
    """MAMMA-format capture directory (meta/ + videos_light/)."""
    resize_hw: tuple[int, int] = (720, 1280)
    """(height, width) the pipeline works at; 4K sources are NVDEC-decoded and GPU-resized."""
    chunk_size: int = 32
    """Frames decoded per camera per chunk."""
    max_frames: int | None = None
    """Optional frame cap for quick runs."""
    start_frame: int = 0
    """First frame to process."""
    no_tracking: bool = False
    """Disable tracking (M1 video-only mode)."""
    mammanet_weights: Path = Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors")
    """Converted MammaNet state-dict; landmarks are skipped if tracking is off."""
    no_landmarks: bool = False
    """Disable dense 2D landmark estimation."""
    no_fitting: bool = False
    """Disable triangulation + SMPL-X fitting."""
    trt_engine: Path | None = None
    """Optional MammaNet TensorRT engine (.trt_cache/*.plan from tools/build_trt_engine.py)."""
    device: str = "cuda"
    """Decode/compute device."""


def main(config: DemoConfig) -> None:
    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)
    print(f"sequence {sequence.name}: {len(sequence.cameras)} cameras, {sequence.frame_count} frames @ {sequence.fps} fps")

    pipeline: StreamingPipeline = build_streaming_pipeline(
        sequence,
        resize_hw=config.resize_hw,
        device=config.device,
        tracker_config=None if config.no_tracking else config.tracker,
        fitter_config=None if config.no_fitting else config.fitter,
        mammanet_weights=None if config.no_landmarks else config.mammanet_weights,
        trt_engine=config.trt_engine,
    )
    stats: PipelineStats = pipeline.run(
        chunk_size=config.chunk_size, max_frames=config.max_frames, start_frame=config.start_frame
    )

    clip_seconds: float = stats.ticks / sequence.fps
    print(f"processed {stats.ticks} ticks x {len(sequence.cameras)} cams in {stats.elapsed_s:.2f}s (clip {clip_seconds:.2f}s, {stats.ticks_per_s:.1f} ticks/s)")
    print(stats.profiler.report())


if __name__ == "__main__":
    main(tyro.cli(DemoConfig))
