from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Int64, UInt8
from tqdm import tqdm

from simplecv.video_io import MultiVideoReader, TorchCodecMultiVideoReader

RGB_LOW_VIDEO_GLOB: str = "*_rgb_low.mp4"
DEFAULT_VIDEO_GLOB: str = "*.mp4"


@dataclass
class BenchmarkConfig:
    """TorchCodec CUDA multiview benchmark config."""

    video_dir: Path
    """Directory containing the synchronized mp4 videos."""
    device: str = "cuda"
    """Torch device passed to TorchCodec."""
    max_frames: int = 120
    """Frames per video to decode. Use -1 for all frames."""
    chunk_size: int = 32
    """Frames per video per TorchCodec chunk."""
    workers: int | None = None
    """Number of camera decode workers. Defaults to one worker per video."""
    ffmpeg_threads: int = 0
    """FFmpeg thread count passed to each TorchCodec decoder."""
    seek_mode: Literal["exact", "approximate"] = "approximate"
    """TorchCodec seek mode."""
    check_fallback: bool = False
    """Print TorchCodec CPU fallback status before benchmarking."""
    skip_torchcodec: bool = False
    """Skip the TorchCodec CUDA benchmark."""
    skip_existing: bool = False
    """Skip the existing MultiVideoReader benchmark."""
    dry_run: bool = False
    """Print discovered video metadata without decoding frames."""


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Decode benchmark timing."""

    label: str
    """Reader label."""
    elapsed_seconds: float
    """Wall-clock seconds for the timed decode path."""
    camera_frames: int
    """Total decoded camera frames across all videos."""
    detail: str
    """Short detail string for the output table."""


def find_video_paths(video_dir: Path) -> list[Path]:
    """Return sorted RGB-low mp4 videos when present, otherwise sorted mp4 videos."""
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    if not video_dir.is_dir():
        raise ValueError(f"Video path must be a directory: {video_dir}")

    video_paths: list[Path] = sorted(path for path in video_dir.glob(RGB_LOW_VIDEO_GLOB) if path.is_file())
    if not video_paths:
        video_paths = sorted(path for path in video_dir.glob(DEFAULT_VIDEO_GLOB) if path.is_file())
    if not video_paths:
        raise FileNotFoundError(f"No mp4 videos found under {video_dir}")
    return video_paths


def format_bytes(num_bytes: int) -> str:
    """Format a byte count."""
    value: float = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"


def required_int(value: int | None, name: str) -> int:
    """Return a required TorchCodec metadata integer."""
    if value is None:
        raise ValueError(f"TorchCodec metadata field {name} is missing")
    return int(value)


def torchcodec_frame_count(video_paths: list[Path]) -> tuple[int, int, int]:
    """Read min frame count, height, and width with TorchCodec metadata."""
    from torchcodec.decoders import VideoDecoder

    frame_counts: list[int] = []
    heights: list[int] = []
    widths: list[int] = []
    for video_path in video_paths:
        decoder = VideoDecoder(video_path, device="cpu", dimension_order="NCHW", num_ffmpeg_threads=0)
        frame_counts.append(required_int(decoder.metadata.num_frames, "num_frames"))
        heights.append(required_int(decoder.metadata.height, "height"))
        widths.append(required_int(decoder.metadata.width, "width"))
    return min(frame_counts), heights[0], widths[0]


def torchcodec_fallback_status(video_paths: list[Path], device: str) -> list[str]:
    """Return TorchCodec CPU fallback status strings for each video."""
    from torchcodec.decoders import VideoDecoder

    statuses: list[str] = []
    for video_path in video_paths:
        decoder = VideoDecoder(video_path, device=device, dimension_order="NCHW", num_ffmpeg_threads=0)
        statuses.append(str(getattr(decoder, "cpu_fallback", "unavailable")))
    return statuses


def maybe_synchronize(device: str) -> None:
    """Synchronize CUDA work when the selected device is CUDA."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_torchcodec_chunked(
    video_paths: list[Path],
    device: str,
    max_frames: int | None,
    num_workers: int | None,
    chunk_size: int,
    num_ffmpeg_threads: int,
    seek_mode: Literal["exact", "approximate"],
) -> TimingResult:
    """Decode the multiview sequence in GPU-resident chunks."""
    maybe_synchronize(device)
    start: float = time.perf_counter()
    video_sources: list[Path | bytes] = list(video_paths)
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(
        video_sources,
        device=device,
        num_workers=num_workers,
        num_ffmpeg_threads=num_ffmpeg_threads,
        seek_mode=seek_mode,
    )
    frame_count: int = len(reader) if max_frames is None else min(max_frames, len(reader))
    total_chunks: int = (frame_count + chunk_size - 1) // chunk_size
    camera_frames: int = 0
    checksum_tensor: Int64[torch.Tensor, ""] | None = None
    for chunk in tqdm(
        reader.iter_chunks(chunk_size=chunk_size, max_frames=max_frames),
        total=total_chunks,
        desc="TorchCodec chunks",
        unit="chunk",
        mininterval=5.0,
    ):
        videos: list[UInt8[torch.Tensor, "b 3 h w"]] = chunk
        for video in videos:
            camera_frames += int(video.shape[0])
            partial_checksum: Int64[torch.Tensor, ""] = video[:, 0, 0, 0].sum(dtype=torch.int64)
            checksum_tensor = partial_checksum if checksum_tensor is None else checksum_tensor + partial_checksum
    maybe_synchronize(device)
    elapsed_seconds: float = time.perf_counter() - start
    checksum: int = int(checksum_tensor.item()) if checksum_tensor is not None else 0
    return TimingResult(
        label=f"TorchCodec {device}",
        elapsed_seconds=elapsed_seconds,
        camera_frames=camera_frames,
        detail=f"chunk_size={chunk_size}, workers={num_workers}, seek={seek_mode}, checksum={checksum}",
    )


def benchmark_existing(video_paths: list[Path], max_frames: int | None) -> TimingResult:
    """Decode with the existing OpenCV-backed MultiVideoReader."""
    start: float = time.perf_counter()
    reader: MultiVideoReader = MultiVideoReader(video_paths)
    frame_count: int = len(reader) if max_frames is None else min(max_frames, len(reader))
    camera_frames: int = 0
    checksum: int = 0
    for frame_idx, bgr_list in enumerate(
        tqdm(reader, total=frame_count, desc="MultiVideoReader", unit="frame", mininterval=5.0)
    ):
        if bgr_list is None:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        for bgr in bgr_list:
            camera_frames += 1
            checksum += int(bgr[0, 0, 0])
    elapsed_seconds: float = time.perf_counter() - start
    return TimingResult(
        label="MultiVideoReader",
        elapsed_seconds=elapsed_seconds,
        camera_frames=camera_frames,
        detail=f"streamed BGR numpy frames, checksum={checksum}",
    )


def print_timing(result: TimingResult) -> None:
    """Print one timing row."""
    fps: float = result.camera_frames / result.elapsed_seconds
    print(f"{result.label:20s} {result.elapsed_seconds:9.3f}s  {fps:9.1f} camera-fps  {result.detail}", flush=True)


def main(config: BenchmarkConfig) -> None:
    """Run the TorchCodec CUDA multiview benchmark."""
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    video_paths: list[Path] = find_video_paths(config.video_dir)
    full_frame_count, height, width = torchcodec_frame_count(video_paths)
    max_frames: int | None = None if config.max_frames < 0 else config.max_frames
    decode_frame_count: int = full_frame_count if max_frames is None else min(max_frames, full_frame_count)
    required_bytes: int = len(video_paths) * decode_frame_count * 3 * height * width

    print(f"Video dir: {config.video_dir}", flush=True)
    print(f"Videos: {len(video_paths)} x {width}x{height}, {full_frame_count} frames/video", flush=True)
    print(f"Full sequence tensor footprint: {format_bytes(required_bytes)}", flush=True)
    chunk_frame_count: int = min(config.chunk_size, decode_frame_count)
    chunk_bytes: int = len(video_paths) * chunk_frame_count * 3 * height * width
    print(f"Chunked TorchCodec tensor bytes: {format_bytes(chunk_bytes)} per chunk", flush=True)
    if config.device.startswith("cuda") and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"CUDA memory: {format_bytes(free_bytes)} free / {format_bytes(total_bytes)} total", flush=True)
    if config.dry_run:
        return
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false. Use --device cpu for a CPU smoke run.")
    if config.check_fallback:
        fallback_statuses: list[str] = torchcodec_fallback_status(video_paths, config.device)
        unique_statuses: str = " | ".join(sorted(set(fallback_statuses)))
        print(f"TorchCodec fallback status: {unique_statuses}", flush=True)

    if not config.skip_torchcodec:
        torchcodec_result: TimingResult = benchmark_torchcodec_chunked(
            video_paths,
            config.device,
            max_frames,
            config.workers,
            config.chunk_size,
            config.ffmpeg_threads,
            config.seek_mode,
        )
        print(flush=True)
        print_timing(torchcodec_result)

    if not config.skip_existing:
        existing_result: TimingResult = benchmark_existing(video_paths, max_frames)
        print_timing(existing_result)
