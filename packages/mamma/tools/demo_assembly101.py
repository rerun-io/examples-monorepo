"""Streaming pipeline on the Assembly101 720p sample (ingestion + visual check).

Assembly101 frames a seated subject assembling toys — partial-body framing is
out of MAMMA's full-body domain, so this validates ingestion end-to-end, not
fit quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro
from simplecv.rerun_log_utils import RerunTyroConfig

from mamma.datasets.assembly101 import load_assembly101_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import PipelineStats, build_streaming_pipeline
from mamma.fitting.window_fitter import FitterConfig
from mamma.tracking.tracker import TrackerConfig


@dataclass
class DemoConfig:
    rr_config: RerunTyroConfig
    """Rerun viewer/save/connect behavior."""
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(expected_subjects=1))
    """Tracker settings (single subject at the bench)."""
    fitter: FitterConfig = field(default_factory=FitterConfig)
    """Sliding-window SMPL-X fitter settings."""
    data_dir: Path = Path("data/assembly101-720-sample")
    """Assembly101 sample root."""
    sequence_name: str = "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724"
    """Sequence name."""
    max_cameras: int = 4
    """Exo cameras to use (realtime envelope; the sample has 8)."""
    resize_hw: tuple[int, int] = (720, 1280)
    """Engine resolution (videos are natively 720p)."""
    max_frames: int | None = 240
    """Frame cap (the sequence is ~16k frames @ 60 fps)."""
    device: str = "cuda"
    """Compute device."""


def main(config: DemoConfig) -> None:
    sequence: MultiViewSequence = load_assembly101_sequence(
        config.data_dir, sequence_name=config.sequence_name, max_cameras=config.max_cameras
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
