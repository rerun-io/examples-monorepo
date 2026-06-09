"""Streaming pipeline demo on a MAMMA-format capture (crossing_arms by default).

M1 scope: NVDEC 4K decode -> GPU resize to 720p -> NVENC H.264 -> per-camera
Rerun VideoStreams + posed pinholes. Model stages land in M2+.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from jaxtyping import UInt8
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.video_io import TorchCodecMultiVideoReader

from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.viz.stream_logger import StreamLogger


@dataclass
class DemoConfig:
    rr_config: RerunTyroConfig
    """Rerun viewer/save/connect behavior."""
    data_dir: Path = Path("data/inputs/indoors/crossing_arms")
    """MAMMA-format capture directory (meta/ + videos_light/)."""
    resize_hw: tuple[int, int] = (720, 1280)
    """(height, width) the pipeline works at; 4K sources are NVDEC-decoded and GPU-resized."""
    chunk_size: int = 32
    """Frames decoded per camera per chunk."""
    max_frames: int | None = None
    """Optional frame cap for quick runs."""
    device: str = "cuda"
    """Decode/compute device."""


def main(config: DemoConfig) -> None:
    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)
    print(f"sequence {sequence.name}: {len(sequence.cameras)} cameras, {sequence.frame_count} frames @ {sequence.fps} fps")

    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(
        list(sequence.video_paths),
        device=config.device,
        resize_hw=config.resize_hw,
    )
    logger: StreamLogger = StreamLogger(sequence, resize_hw=config.resize_hw)
    logger.setup()

    start: float = time.perf_counter()
    frame_idx: int = 0
    for chunk in reader.iter_chunks(chunk_size=config.chunk_size, max_frames=config.max_frames):
        chunk_len: int = min(int(video.shape[0]) for video in chunk)
        for local_idx in range(chunk_len):
            frames: list[UInt8[torch.Tensor, "3 h w"]] = [video[local_idx] for video in chunk]
            logger.log_tick_video(frame_idx, frames)
            frame_idx += 1
    logger.flush()

    elapsed: float = time.perf_counter() - start
    clip_seconds: float = frame_idx / sequence.fps
    print(f"processed {frame_idx} frames x {len(sequence.cameras)} cams in {elapsed:.2f}s (clip {clip_seconds:.2f}s, {frame_idx / elapsed:.1f} ticks/s)")
    for cam_name, stats in logger.encoder_stats.items():
        print(f"  {cam_name}: encoder={stats['encoder']} avg={stats['avg_ms_per_frame']}ms/frame bytes={stats['total_bytes']}")


if __name__ == "__main__":
    main(tyro.cli(DemoConfig))
