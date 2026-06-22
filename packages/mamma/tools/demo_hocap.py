"""Streaming pipeline on the HOCap sample (ingestion + visual check).

HOCap is a tabletop hand-object capture — upper-body framing is out of
MAMMA's full-body domain, so this demo validates ingestion (calibration,
sync, tracking, landmarks, fit running end-to-end), not fit quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro
from simplecv.rerun_log_utils import RerunTyroConfig

from mamma.datasets.hocap import load_hocap_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import PipelineStats, build_streaming_pipeline
from mamma.fitting.window_fitter import FitterConfig
from mamma.tracking.tracker import TrackerConfig


@dataclass
class DemoConfig:
    rr_config: RerunTyroConfig
    """Rerun viewer/save/connect behavior."""
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(expected_subjects=1))
    """Tracker settings (HOCap is single-subject)."""
    fitter: FitterConfig = field(default_factory=FitterConfig)
    """Sliding-window SMPL-X fitter settings."""
    data_dir: Path = Path("data/hocap/sample")
    """HOCap sample root."""
    subject_id: str = "8"
    """HOCap subject."""
    sequence_name: str = "20231024_180733"
    """HOCap capture."""
    max_cameras: int = 4
    """Exo cameras to use (realtime envelope)."""
    resize_hw: tuple[int, int] = (480, 640)
    """Engine resolution (HOCap RealSense is natively 640x480)."""
    max_frames: int | None = 240
    """Frame cap (the sample is ~1085 frames)."""
    device: str = "cuda"
    """Compute device."""


def main(config: DemoConfig) -> None:
    sequence: MultiViewSequence = load_hocap_sequence(
        config.data_dir, subject_id=config.subject_id, sequence_name=config.sequence_name, max_cameras=config.max_cameras
    )
    print(f"{sequence.name}: {len(sequence.cameras)} cameras, {sequence.frame_count} frames @ {sequence.fps} fps")

    pipeline = build_streaming_pipeline(
        sequence,
        resize_hw=config.resize_hw,
        device=config.device,
        tracker_config=config.tracker,
        fitter_config=config.fitter,
        mammanet_weights=Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors"),
    )
    stats: PipelineStats = pipeline.run(max_frames=config.max_frames)
    print(f"processed {stats.ticks} ticks x {len(sequence.cameras)} cams in {stats.elapsed_s:.2f}s ({stats.ticks_per_s:.1f} ticks/s)")
    print(stats.profiler.report())


if __name__ == "__main__":
    main(tyro.cli(DemoConfig))
