from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Int, UInt8
from numpy import ndarray
from tqdm import tqdm

from simplecv.data.exo.hocap_exo import ExoCameraIDs
from simplecv.rerun_log_utils import log_video
from simplecv.video_io import TorchCodecMultiVideoReader

DEFAULT_HOCAP_ROOT: Path = Path("data/hocap/sample")
DEFAULT_SAVE_PATH: Path = Path("/tmp/hocap_torchcodec_multiview.rrd")
TIMELINE: str = "video_time"
PREVIEW_SECONDS: float = 1.0
HOCAP_EXO_CAMERA_NAMES: tuple[str, ...] = tuple(get_args(ExoCameraIDs))


@dataclass
class HocapTorchCodecViewConfig:
    """HOCAP TorchCodec multiview Rerun viewer config."""

    root_directory: Path = DEFAULT_HOCAP_ROOT
    """HOCAP sample/full root directory containing ``subject_<id>/<sequence>``."""
    subject_id: str = "8"
    """HOCAP subject id."""
    sequence_name: str = "20231024_180733"
    """HOCAP sequence name."""
    save_path: Path = DEFAULT_SAVE_PATH
    """Path to save the generated Rerun recording."""
    max_videos: int = len(HOCAP_EXO_CAMERA_NAMES)
    """Maximum number of exo camera videos to log for the side-by-side check."""
    max_frames: int = 32
    """Maximum decoded frames per video to log as ``rr.Image``."""
    chunk_size: int = 32
    """TorchCodec decode chunk size."""
    device: str = "cuda"
    """TorchCodec decode device."""


def find_hocap_video_paths(root_directory: Path, subject_id: str, sequence_name: str, max_videos: int) -> list[Path]:
    """Find sorted HOCAP exo camera videos under one sequence directory."""
    if max_videos <= 0:
        raise ValueError("max_videos must be positive")
    sequence_dir: Path = root_directory / f"subject_{subject_id}" / sequence_name
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"HOCAP sequence directory does not exist: {sequence_dir}")

    video_paths: list[Path] = []
    for camera_name in HOCAP_EXO_CAMERA_NAMES:
        video_path: Path = sequence_dir / camera_name / "output.mp4"
        if video_path.is_file():
            video_paths.append(video_path)
    if not video_paths:
        raise FileNotFoundError(f"No HOCAP output.mp4 videos found under {sequence_dir}")
    return video_paths[:max_videos]


def build_blueprint(camera_names: list[str]) -> rrb.Blueprint:
    """Build a compact video-stream vs decoded-image comparison blueprint."""
    rows: list[rrb.Horizontal] = []
    for camera_name in camera_names:
        camera_root: str = f"/hocap/{camera_name}"
        row: rrb.Horizontal = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"{camera_root}/video", name=f"{camera_name} video"),
            rrb.Spatial2DView(origin=f"{camera_root}/decoded", name=f"{camera_name} decoded"),
            column_shares=[1, 1],
        )
        rows.append(row)
    preview_time_selection: rrb.components.AbsoluteTimeRange = rrb.components.AbsoluteTimeRange(
        min=rr.datatypes.TimeInt(seconds=0.0),
        max=rr.datatypes.TimeInt(seconds=PREVIEW_SECONDS),
    )
    time_panel: rrb.TimePanel = rrb.TimePanel(
        timeline=TIMELINE,
        play_state="playing",
        loop_mode="selection",
        time_selection=preview_time_selection,
    )
    return rrb.Blueprint(rrb.Vertical(contents=rows), time_panel, collapse_panels=True)


def rgb_chw_to_rgb_hwc_numpy(rgb_chw: UInt8[torch.Tensor, "3 h w"]) -> UInt8[ndarray, "h w 3"]:
    """Move one RGB CHW torch tensor to a contiguous RGB HWC numpy image."""
    rgb_hwc: UInt8[torch.Tensor, "h w 3"] = rearrange(rgb_chw, "c h w -> h w c")
    rgb_np: UInt8[ndarray, "h w 3"] = rgb_hwc.cpu().contiguous().numpy()
    return rgb_np


def log_torchcodec_decoded_images(
    *,
    reader: TorchCodecMultiVideoReader,
    camera_names: list[str],
    frame_timestamps_by_video: list[Int[ndarray, "num_frames"]],
    max_frames: int,
    chunk_size: int,
    recording: rr.RecordingStream,
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
        for camera_idx, video in enumerate(videos):
            camera_name: str = camera_names[camera_idx]
            frame_timestamps_ns: Int[ndarray, "num_frames"] = frame_timestamps_by_video[camera_idx]
            for local_frame_idx in range(int(video.shape[0])):
                frame_idx: int = start_frame_idx + local_frame_idx
                if frame_idx >= len(frame_timestamps_ns):
                    continue
                rgb_chw: UInt8[torch.Tensor, "3 h w"] = video[local_frame_idx]
                rgb_hwc: UInt8[ndarray, "h w 3"] = rgb_chw_to_rgb_hwc_numpy(rgb_chw)
                rr.set_time(TIMELINE, duration=float(frame_timestamps_ns[frame_idx]) * 1e-9, recording=recording)
                rr.set_time("frame", sequence=frame_idx, recording=recording)
                rr.log(
                    f"/hocap/{camera_name}/decoded",
                    rr.Image(rgb_hwc, color_model=rr.ColorModel.RGB).compress(jpeg_quality=80),
                    recording=recording,
                )


def main(config: HocapTorchCodecViewConfig) -> None:
    """Log HOCAP native video streams and TorchCodec decoded image frames to Rerun."""
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false. Use --device cpu for a smoke run.")

    video_paths: list[Path] = find_hocap_video_paths(
        config.root_directory,
        subject_id=config.subject_id,
        sequence_name=config.sequence_name,
        max_videos=config.max_videos,
    )
    camera_names: list[str] = [path.parent.name for path in video_paths]

    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    recording: rr.RecordingStream = rr.RecordingStream(
        application_id="hocap_torchcodec_multiview",
        recording_id=f"subject_{config.subject_id}_{config.sequence_name}",
    )
    recording.save(str(config.save_path))
    rr.send_blueprint(build_blueprint(camera_names), recording=recording)

    frame_timestamps_by_video: list[Int[ndarray, "num_frames"]] = []
    for camera_name, video_path in zip(camera_names, video_paths, strict=True):
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
            video_path,
            video_log_path=Path(f"/hocap/{camera_name}/video"),
            timeline=TIMELINE,
            recording=recording,
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
        camera_names=camera_names,
        frame_timestamps_by_video=frame_timestamps_by_video,
        max_frames=config.max_frames,
        chunk_size=config.chunk_size,
        recording=recording,
    )
    recording.flush(timeout_sec=30.0)
    print(f"Saved {config.save_path}")
