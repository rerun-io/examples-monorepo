from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rerun as rr
import torch
from jaxtyping import Int
from numpy import ndarray

from simplecv._torchcodec_view_utils import TIMELINE, build_blueprint, log_torchcodec_decoded_chunks
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from simplecv.video_io import TorchCodecMultiVideoReader


@dataclass
class TorchCodecMultiviewConfig:
    """TorchCodec multiview Rerun viewer config."""

    rr_config: RerunTyroConfig
    video_dir: Path
    """Directory containing one or more ``.mp4`` videos."""
    max_videos: int = 0
    """Maximum videos to log for the side-by-side check. Use ``0`` for all videos."""
    max_frames: int = 32 * 10
    """Maximum decoded frames per video to log as ``rr.Image``."""
    chunk_size: int = 32
    """TorchCodec decode chunk size."""
    device: str = "cuda"
    """TorchCodec decode device."""


def discover_video_paths(video_dir: Path, max_videos: int) -> list[Path]:
    """Find sorted ``.mp4`` videos under a directory."""
    if max_videos < 0:
        raise ValueError("max_videos must be non-negative")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    video_paths: list[Path] = sorted(video_path for video_path in video_dir.rglob("*.mp4") if video_path.is_file())
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 videos found under {video_dir}")
    if max_videos == 0:
        return video_paths
    return video_paths[:max_videos]


def video_entity_names(video_dir: Path, video_paths: list[Path]) -> list[str]:
    """Derive stable Rerun entity names from video paths."""
    entity_names: list[str] = []
    for video_path in video_paths:
        relative_path: Path = video_path.relative_to(video_dir)
        entity_name: str = relative_path.with_suffix("").as_posix()
        entity_names.append(entity_name)
    return entity_names


def main(config: TorchCodecMultiviewConfig) -> None:
    """Log native video streams and TorchCodec decoded image frames to Rerun."""
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but torch.cuda.is_available() is false. Use --device cpu for a smoke run."
        )

    video_paths: list[Path] = discover_video_paths(config.video_dir, max_videos=config.max_videos)
    video_names: list[str] = video_entity_names(config.video_dir, video_paths)

    rr.send_blueprint(build_blueprint(video_names))

    frame_timestamps_by_video: list[Int[ndarray, "num_frames"]] = []
    for video_name, video_path in zip(video_names, video_paths, strict=True):
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
            video_path,
            video_log_path=Path(f"/videos/{video_name}/video"),
            timeline=TIMELINE,
        )
        frame_timestamps_by_video.append(frame_timestamps_ns)

    video_sources: list[Path | bytes] = list(video_paths)
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(
        video_sources,
        device=config.device,
        seek_mode="approximate",
    )
    total_frames: int = min(config.max_frames, len(reader))
    log_torchcodec_decoded_chunks(
        chunked_videos=reader.iter_chunks(chunk_size=config.chunk_size, max_frames=total_frames),
        video_names=video_names,
        frame_timestamps_by_video=frame_timestamps_by_video,
        total_frames=total_frames,
        chunk_size=config.chunk_size,
    )
