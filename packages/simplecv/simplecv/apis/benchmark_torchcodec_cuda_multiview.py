from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Int64, UInt8
from tqdm import tqdm

from simplecv._benchmark_utils import (
    CONSOLE,
    TimingResult,
    TorchCodecVideoMetadata,
    maybe_synchronize,
    print_timing_results,
    read_torchcodec_fallback_statuses,
    read_torchcodec_video_metadata,
    validate_chunk_size,
)
from simplecv.image_types import BGRList, ImageBGR
from simplecv.print_utils import format_bytes
from simplecv.video_io import (
    MultiVideoReader,
    TorchCodecMultiVideoReader,
    bgr_hwc_frames_to_rgb_nchw_tensor,
)

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
    """Skip the OpenCV-backed MultiVideoReader benchmarks."""
    dry_run: bool = False
    """Print discovered video metadata without decoding frames."""


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
    validate_chunk_size(chunk_size)
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
    total_camera_frames: int = frame_count * len(video_paths)
    camera_frames: int = 0
    checksum_tensor: Int64[torch.Tensor, ""] | None = None
    with tqdm(
        total=total_camera_frames,
        desc=f"TorchCodec chunks ({chunk_size} frames/chunk x {len(video_paths)} cameras)",
        unit=" camera-frames",
        mininterval=5.0,
    ) as progress_bar:
        for chunk in reader.iter_chunks(chunk_size=chunk_size, max_frames=max_frames):
            videos: list[UInt8[torch.Tensor, "b 3 h w"]] = chunk
            chunk_camera_frames: int = 0
            for video in videos:
                decoded_frames: int = video.shape[0]
                camera_frames += decoded_frames
                chunk_camera_frames += decoded_frames
                partial_checksum: Int64[torch.Tensor, ""] = video[:, 0, 0, 0].sum(dtype=torch.int64)
                checksum_tensor = partial_checksum if checksum_tensor is None else checksum_tensor + partial_checksum
            progress_bar.update(chunk_camera_frames)
    maybe_synchronize(device)
    elapsed_seconds: float = time.perf_counter() - start
    checksum: int = int(checksum_tensor.item()) if checksum_tensor is not None else 0
    return TimingResult(
        label=f"TorchCodec {device}",
        elapsed_seconds=elapsed_seconds,
        frames=camera_frames,
        detail=f"chunk_size={chunk_size}, workers={num_workers}, seek={seek_mode}, checksum={checksum}",
    )


def benchmark_existing(video_paths: list[Path], max_frames: int | None) -> TimingResult:
    """Decode with the existing OpenCV-backed MultiVideoReader."""
    start: float = time.perf_counter()
    reader: MultiVideoReader = MultiVideoReader(video_paths)
    frame_count: int = len(reader) if max_frames is None else min(max_frames, len(reader))
    total_camera_frames: int = frame_count * len(video_paths)
    camera_frames: int = 0
    checksum: int = 0
    with tqdm(
        total=total_camera_frames,
        desc=f"MultiVideoReader ({len(video_paths)} cameras)",
        unit=" camera-frames",
        mininterval=5.0,
    ) as progress_bar:
        for bgr_list in islice(reader, frame_count):
            if bgr_list is None:
                break
            bgr_frames: BGRList = bgr_list
            decoded_camera_frames: int = len(bgr_frames)
            for bgr in bgr_frames:
                checksum += int(bgr[0, 0, 0])
            camera_frames += decoded_camera_frames
            progress_bar.update(decoded_camera_frames)
    elapsed_seconds: float = time.perf_counter() - start
    return TimingResult(
        label="MultiVideoReader",
        elapsed_seconds=elapsed_seconds,
        frames=camera_frames,
        detail=f"streamed BGR numpy frames, checksum={checksum}",
    )


def benchmark_existing_tensor(
    video_paths: list[Path],
    max_frames: int | None,
    device: str,
    chunk_size: int,
) -> TimingResult:
    """Decode with MultiVideoReader and convert chunks to RGB ``NCHW`` uint8 tensors."""
    validate_chunk_size(chunk_size)
    maybe_synchronize(device)
    start: float = time.perf_counter()
    reader: MultiVideoReader = MultiVideoReader(video_paths)
    frame_count: int = len(reader) if max_frames is None else min(max_frames, len(reader))
    total_camera_frames: int = frame_count * len(video_paths)
    camera_frames: int = 0
    checksum_tensor: Int64[torch.Tensor, ""] | None = None
    bgr_chunks_by_camera: list[list[ImageBGR]] = [[] for _ in video_paths]
    with tqdm(
        total=total_camera_frames,
        desc=f"MultiVideoReader -> {device} ({chunk_size} frames/chunk x {len(video_paths)} cameras)",
        unit=" camera-frames",
        mininterval=5.0,
    ) as progress_bar:
        for bgr_list in islice(reader, frame_count):
            if bgr_list is None:
                break
            bgr_frames: BGRList = bgr_list
            for camera_idx, bgr in enumerate(bgr_frames):
                bgr_chunks_by_camera[camera_idx].append(bgr)
            if all(len(bgr_chunk) == chunk_size for bgr_chunk in bgr_chunks_by_camera):
                chunk_camera_frames: int = 0
                for bgr_chunk in bgr_chunks_by_camera:
                    video: UInt8[torch.Tensor, "b 3 h w"] = bgr_hwc_frames_to_rgb_nchw_tensor(bgr_chunk, device=device)
                    decoded_frames: int = video.shape[0]
                    camera_frames += decoded_frames
                    chunk_camera_frames += decoded_frames
                    partial_checksum: Int64[torch.Tensor, ""] = video[:, 0, 0, 0].sum(dtype=torch.int64)
                    checksum_tensor = partial_checksum if checksum_tensor is None else checksum_tensor + partial_checksum
                bgr_chunks_by_camera = [[] for _ in video_paths]
                progress_bar.update(chunk_camera_frames)
        if any(bgr_chunks_by_camera):
            final_camera_frames: int = 0
            for bgr_chunk in bgr_chunks_by_camera:
                if not bgr_chunk:
                    continue
                final_video: UInt8[torch.Tensor, "b 3 h w"] = bgr_hwc_frames_to_rgb_nchw_tensor(bgr_chunk, device=device)
                final_frame_count: int = final_video.shape[0]
                camera_frames += final_frame_count
                final_camera_frames += final_frame_count
                final_checksum: Int64[torch.Tensor, ""] = final_video[:, 0, 0, 0].sum(dtype=torch.int64)
                checksum_tensor = final_checksum if checksum_tensor is None else checksum_tensor + final_checksum
            progress_bar.update(final_camera_frames)
    maybe_synchronize(device)
    elapsed_seconds: float = time.perf_counter() - start
    checksum: int = int(checksum_tensor.item()) if checksum_tensor is not None else 0
    return TimingResult(
        label=f"MultiVideoReader -> {device}",
        elapsed_seconds=elapsed_seconds,
        frames=camera_frames,
        detail=f"RGB NCHW uint8 tensors, chunk_size={chunk_size}, checksum={checksum}",
    )


def main(config: BenchmarkConfig) -> None:
    """Run the TorchCodec CUDA multiview benchmark."""
    validate_chunk_size(config.chunk_size)

    video_paths: list[Path] = find_video_paths(config.video_dir)
    metadata: TorchCodecVideoMetadata = read_torchcodec_video_metadata(video_paths)
    full_frame_count: int = metadata.frame_count
    height: int = metadata.height
    width: int = metadata.width
    max_frames: int | None = None if config.max_frames < 0 else config.max_frames
    decode_frame_count: int = full_frame_count if max_frames is None else min(max_frames, full_frame_count)
    required_bytes: int = len(video_paths) * decode_frame_count * 3 * height * width

    CONSOLE.print(f"Video dir: {config.video_dir}", markup=False)
    CONSOLE.print(f"Videos: {len(video_paths)} x {width}x{height}, {full_frame_count} frames/video", markup=False)
    CONSOLE.print(f"Full sequence tensor footprint: {format_bytes(required_bytes)}")
    chunk_frame_count: int = min(config.chunk_size, decode_frame_count)
    chunk_bytes: int = len(video_paths) * chunk_frame_count * 3 * height * width
    CONSOLE.print(f"Chunked TorchCodec tensor bytes: {format_bytes(chunk_bytes)} per chunk")
    if config.device.startswith("cuda") and torch.cuda.is_available():
        cuda_memory_result: tuple[int, int] = torch.cuda.mem_get_info()
        free_bytes: int = cuda_memory_result[0]
        total_bytes: int = cuda_memory_result[1]
        CONSOLE.print(f"CUDA memory: {format_bytes(free_bytes)} free / {format_bytes(total_bytes)} total")
    if config.dry_run:
        return
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but torch.cuda.is_available() is false. Use --device cpu for a CPU smoke run."
        )
    if config.check_fallback:
        fallback_statuses: list[str] = read_torchcodec_fallback_statuses(video_paths, config.device)
        unique_statuses: str = " | ".join(sorted(set(fallback_statuses)))
        CONSOLE.print(f"TorchCodec fallback status: {unique_statuses}", markup=False)

    timing_results: list[TimingResult] = []
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
        timing_results.append(torchcodec_result)

    if not config.skip_existing:
        existing_tensor_result: TimingResult = benchmark_existing_tensor(
            video_paths,
            max_frames,
            config.device,
            config.chunk_size,
        )
        timing_results.append(existing_tensor_result)

        existing_result: TimingResult = benchmark_existing(video_paths, max_frames)
        timing_results.append(existing_result)

    print_timing_results(timing_results, throughput_header="Camera FPS")
