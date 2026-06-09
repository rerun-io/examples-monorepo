from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rerun as rr
import torch
from jaxtyping import Int
from numpy import ndarray

from simplecv._torchcodec_view_utils import TIMELINE, build_blueprint, log_torchcodec_decoded_chunks
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from simplecv.video_io import TorchCodecVideoReader, validate_video_file_path


@dataclass
class TorchCodecSingleviewConfig:
    """TorchCodec singleview Rerun viewer config."""

    rr_config: RerunTyroConfig
    video_path: Path
    """Path to an ``.mp4`` video."""
    max_frames: int = 32 * 10
    """Maximum decoded frames to log as ``rr.Image``."""
    chunk_size: int = 32
    """TorchCodec decode chunk size."""
    device: str = "cuda"
    """TorchCodec decode device."""


def main(config: TorchCodecSingleviewConfig) -> None:
    """Log native video streams and TorchCodec decoded image frames to Rerun."""
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but torch.cuda.is_available() is false. Use --device cpu for a smoke run."
        )

    video_path: Path = validate_video_file_path(config.video_path)
    video_name: str = video_path.stem
    rr.send_blueprint(build_blueprint([video_name]))

    frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        video_path,
        video_log_path=Path(f"/videos/{video_name}/video"),
        timeline=TIMELINE,
    )

    video_source: Path = config.video_path
    torch_reader: TorchCodecVideoReader = TorchCodecVideoReader(
        video_source,
        device=config.device,
        seek_mode="approximate",
    )
    total_frames: int = min(config.max_frames, len(torch_reader))
    log_torchcodec_decoded_chunks(
        chunked_videos=([video] for video in torch_reader.iter_chunks(chunk_size=config.chunk_size, max_frames=total_frames)),
        video_names=[video_name],
        frame_timestamps_by_video=[frame_timestamps_ns],
        total_frames=total_frames,
        chunk_size=config.chunk_size,
    )
