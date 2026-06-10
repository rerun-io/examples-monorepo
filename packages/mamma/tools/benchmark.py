"""Realtime benchmark: process the full crossing_arms clip (363 frames x 4 cams)
through the complete streaming pipeline, including Rerun logging, and gate wall
time against the gate (15 s = at worst 80% of realtime for the 12.1 s clip;
Pablo 2026-06-10: track_stride=1 mask quality is worth the slack).

Exit code 0 = PASS, 1 = FAIL. Run in the non-dev env (beartype off).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from simplecv.rerun_log_utils import RerunTyroConfig

from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import StreamingPipeline, build_streaming_pipeline
from mamma.fitting.window_fitter import FitterConfig
from mamma.tracking.tracker import TrackerConfig


@dataclass
class BenchmarkConfig:
    rr_config: RerunTyroConfig
    """Rerun behavior; logging cost is part of the gate, so the default
    headless+in-memory sink still exercises the full logging path."""
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(expected_subjects=1))
    """Tracker settings."""
    fitter: FitterConfig = field(default_factory=FitterConfig)
    """Fitter settings."""
    data_dir: Path = Path("data/inputs/indoors/crossing_arms")
    """Benchmark capture."""
    mammanet_weights: Path = Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors")
    """Converted MammaNet weights."""
    resize_hw: tuple[int, int] = (720, 1280)
    """Engine resolution."""
    trt_engine: Path | None = None
    """Optional MammaNet TensorRT engine plan."""
    gate_seconds: float = 15.0
    """PASS bound on wall time for the full clip (>=80% of realtime; the
    12.1 s clip duration would be exact realtime)."""
    warmup_frames: int = 33
    """Frames run once beforehand to absorb model load/compile/cudnn autotune."""
    device: str = "cuda"
    """Compute device."""


def build_pipeline(config: BenchmarkConfig, sequence: MultiViewSequence) -> StreamingPipeline:
    return build_streaming_pipeline(
        sequence,
        resize_hw=config.resize_hw,
        device=config.device,
        tracker_config=config.tracker,
        fitter_config=config.fitter,
        mammanet_weights=config.mammanet_weights,
        trt_engine=config.trt_engine,
    )


def main(config: BenchmarkConfig) -> int:
    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)

    # Warmup pass: fresh pipeline, short slice; absorbs weight load + autotune.
    # Its Rerun output goes to a sink-less throwaway recording — with a --save
    # sink, warmup + timed samples on one timeline duplicate VideoStream PTS
    # and corrupt H.264 decode in the viewer.
    import rerun as rr

    main_recording = rr.get_global_data_recording()
    rr.set_global_data_recording(rr.RecordingStream(application_id="mamma-benchmark-warmup"))
    warmup: StreamingPipeline = build_pipeline(config, sequence)
    warmup.run(max_frames=config.warmup_frames)
    if main_recording is not None:
        rr.set_global_data_recording(main_recording)
    print(f"warmup done ({config.warmup_frames} frames)")
    # Release the warmup pipeline's models + NVDEC contexts before building the
    # timed one — two resident copies exhaust CUDA memory (scale_cuda OOM).
    import gc

    import torch

    warmup.close()  # terminate the warmup decode workers (NVDEC + CUDA ctx each)
    del warmup
    gc.collect()
    torch.cuda.empty_cache()

    timed: StreamingPipeline = build_pipeline(config, sequence)
    stats = timed.run()
    timed.close()
    clip_seconds: float = stats.ticks / sequence.fps
    cam_fps: float = stats.ticks_per_s * len(sequence.cameras)

    print(f"\nBENCHMARK ({stats.ticks} frames x {len(sequence.cameras)} cams @ {sequence.fps:.0f} fps, incl. Rerun logging):")
    print(stats.profiler.report())
    print(f"\n  wall      {stats.elapsed_s:7.2f} s   (clip duration {clip_seconds:.2f} s)")
    print(f"  throughput {stats.ticks_per_s:6.1f} ticks/s ({cam_fps:.0f} cam-fps); realtime needs {sequence.fps:.0f} ticks/s")
    ok: bool = stats.elapsed_s <= config.gate_seconds
    print(f"  gate      {config.gate_seconds} s  ->  {'PASS' if ok else 'FAIL'}")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(tyro.cli(BenchmarkConfig)))
