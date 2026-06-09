"""Realtime benchmark: process the full crossing_arms clip (363 frames x 4 cams)
through the complete streaming pipeline, including Rerun logging, and gate wall
time against the clip duration (12.1 s = 30 fps sustained).

Exit code 0 = PASS, 1 = FAIL. Run in the non-dev env (beartype off).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.video_io import TorchCodecMultiVideoReader

from mamma.calibration.npz_contract import CameraCalibration
from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import StreamingPipeline
from mamma.fitting.stage import FittingStage
from mamma.fitting.window_fitter import FitterConfig
from mamma.landmarks.estimator import LandmarkEstimator
from mamma.tracking.tracker import MultiViewTracker, TrackerConfig
from mamma.viz.stream_logger import StreamLogger


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
    gate_seconds: float = 12.1
    """PASS bound on wall time for the full clip (clip duration = realtime)."""
    warmup_frames: int = 33
    """Frames run once beforehand to absorb model load/compile/cudnn autotune."""
    device: str = "cuda"
    """Compute device."""


def build_pipeline(config: BenchmarkConfig, sequence: MultiViewSequence) -> StreamingPipeline:
    scaled_cams: list[CameraCalibration] = [
        cam.scaled_to(height=config.resize_hw[0], width=config.resize_hw[1]) for cam in sequence.cameras
    ]
    reader = TorchCodecMultiVideoReader(list(sequence.video_paths), device=config.device, resize_hw=config.resize_hw)
    logger = StreamLogger(sequence, resize_hw=config.resize_hw)
    tracker = MultiViewTracker(scaled_cams, config.tracker)
    estimator = LandmarkEstimator(config.mammanet_weights, device=config.device)
    fitting = FittingStage(scaled_cams, config.fitter)
    return StreamingPipeline(sequence, reader, logger, tracker=tracker, landmarks=estimator, fitting=fitting)


def main(config: BenchmarkConfig) -> int:
    config.tracker.device = config.device
    config.fitter.device = config.device
    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)

    # Warmup pass: fresh pipeline, short slice; absorbs weight load + autotune.
    warmup: StreamingPipeline = build_pipeline(config, sequence)
    warmup.run(max_frames=config.warmup_frames)
    print(f"warmup done ({config.warmup_frames} frames)")

    timed: StreamingPipeline = build_pipeline(config, sequence)
    stats = timed.run()
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
