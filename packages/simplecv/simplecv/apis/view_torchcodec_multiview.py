from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Int, UInt8
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm

from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from simplecv.video_io import TorchCodecMultiVideoReader

DEFAULT_SAVE_PATH: Path = Path("/tmp/torchcodec_multiview.rrd")
TIMELINE: str = "video_time"
PREVIEW_SECONDS: float = 1.0


@dataclass
class TorchCodecMultiviewConfig:
    """TorchCodec multiview Rerun viewer config."""

    rr_config: RerunTyroConfig
    video_dir: Path
    """Directory containing one or more ``.mp4`` videos."""
    max_videos: int = 0
    """Maximum videos to log for the side-by-side check. Use ``0`` for all videos."""
    max_frames: int = 32
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


def build_blueprint(video_names: list[str]) -> rrb.Blueprint:
    """Build a compact video-stream vs decoded-image comparison blueprint."""
    rows: list[rrb.Horizontal] = []
    for video_name in video_names:
        video_root: str = f"/videos/{video_name}"
        row: rrb.Horizontal = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"{video_root}/video", name=f"{video_name} video"),
            rrb.Spatial2DView(origin=f"{video_root}/decoded", name=f"{video_name} decoded"),
            column_shares=[1, 1],
        )
        rows.append(row)

    time_panel: rrb.TimePanel = rrb.TimePanel(
        timeline=TIMELINE,
        play_state="playing",
        loop_mode="selection",
    )
    return rrb.Blueprint(rrb.Tabs(contents=rows), time_panel, collapse_panels=True)


def rgb_chw_to_rgb_hwc_numpy(rgb_chw: UInt8[Tensor, "3 h w"]) -> UInt8[ndarray, "h w 3"]:
    """Move one RGB CHW torch tensor to a contiguous RGB HWC numpy image."""
    rgb_hwc: UInt8[Tensor, "h w 3"] = rearrange(rgb_chw, "c h w -> h w c")
    rgb_np: UInt8[ndarray, "h w 3"] = rgb_hwc.cpu().contiguous().numpy()
    return rgb_np


def log_torchcodec_decoded_images(
    *,
    reader: TorchCodecMultiVideoReader,
    video_names: list[str],
    frame_timestamps_by_video: list[Int[ndarray, "num_frames"]],
    max_frames: int,
    chunk_size: int,
) -> None:
    """Decode multiview chunks with TorchCodec and log RGB images next to video streams."""
    total_frames: int = min(max_frames, len(reader))
    for chunk_start, videos in enumerate(
        tqdm(
            reader.iter_chunks(chunk_size=chunk_size, max_frames=total_frames),
            total=(total_frames + chunk_size - 1) // chunk_size,
            desc="TorchCodec images",
            unit="chunk",
        )
    ):
        start_frame_idx: int = chunk_start * chunk_size
        for video_idx, video in enumerate(
            tqdm(
                videos,
                desc="Videos in chunk",
                leave=False,
            )
        ):
            video_name: str = video_names[video_idx]
            frame_timestamps_ns: Int[ndarray, "num_frames"] = frame_timestamps_by_video[video_idx]
            for local_frame_idx in range(video.shape[0]):
                frame_idx: int = start_frame_idx + local_frame_idx
                if frame_idx >= len(frame_timestamps_ns):
                    continue
                rgb_chw: UInt8[Tensor, "3 h w"] = video[local_frame_idx]
                rgb_hwc: UInt8[ndarray, "h w 3"] = rgb_chw_to_rgb_hwc_numpy(rgb_chw)
                rr.set_time(TIMELINE, duration=float(frame_timestamps_ns[frame_idx]) * 1e-9)
                rr.set_time("frame", sequence=frame_idx)
                rr.log(
                    f"/videos/{video_name}/decoded",
                    rr.Image(rgb_hwc, color_model=rr.ColorModel.RGB).compress(jpeg_quality=80),
                )


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
    log_torchcodec_decoded_images(
        reader=reader,
        video_names=video_names,
        frame_timestamps_by_video=frame_timestamps_by_video,
        max_frames=config.max_frames,
        chunk_size=config.chunk_size,
    )
