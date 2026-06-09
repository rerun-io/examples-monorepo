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
from simplecv.image_types import ImageBGR
from simplecv.print_utils import format_bytes
from simplecv.video_io import (
    TorchCodecVideoReader,
    VideoReader,
    bgr_hwc_frames_to_rgb_nchw_tensor,
    validate_video_file_path,
)


@dataclass
class BenchmarkConfig:
    """TorchCodec CUDA singleview benchmark config."""

    video_path: Path
    """Path to the mp4 video."""
    device: str = "cuda"
    """Torch device passed to TorchCodec."""
    max_frames: int = 120
    """Frames to decode. Use -1 for all frames."""
    chunk_size: int = 32
    """Frames per TorchCodec chunk."""
    ffmpeg_threads: int = 0
    """FFmpeg thread count passed to TorchCodec."""
    seek_mode: Literal["exact", "approximate"] = "approximate"
    """TorchCodec seek mode."""
    check_fallback: bool = False
    """Print TorchCodec CPU fallback status before benchmarking."""
    skip_torchcodec: bool = False
    """Skip the TorchCodec CUDA benchmark."""
    skip_existing: bool = False
    """Skip the OpenCV-backed VideoReader benchmarks."""
    dry_run: bool = False
    """Print discovered video metadata without decoding frames."""


def benchmark_torchcodec_chunked(
    video_path: Path,
    device: str,
    max_frames: int | None,
    chunk_size: int,
    num_ffmpeg_threads: int,
    seek_mode: Literal["exact", "approximate"],
) -> TimingResult:
    """Decode one video in GPU-resident chunks."""
    validate_chunk_size(chunk_size)
    maybe_synchronize(device)
    start_time: float = time.perf_counter()
    reader: TorchCodecVideoReader = TorchCodecVideoReader(
        video_path,
        device=device,
        num_ffmpeg_threads=num_ffmpeg_threads,
        seek_mode=seek_mode,
    )
    frame_count: int = len(reader) if max_frames is None else min(max_frames, len(reader))
    frames: int = 0
    checksum_tensor: Int64[torch.Tensor, ""] | None = None
    with tqdm(
        total=frame_count,
        desc=f"TorchCodec chunks ({chunk_size} frames/chunk)",
        unit=" frames",
        mininterval=5.0,
    ) as progress_bar:
        for video in reader.iter_chunks(chunk_size=chunk_size, max_frames=max_frames):
            decoded_frames: int = video.shape[0]
            frames += decoded_frames
            partial_checksum: Int64[torch.Tensor, ""] = video[:, 0, 0, 0].sum(dtype=torch.int64)
            checksum_tensor = partial_checksum if checksum_tensor is None else checksum_tensor + partial_checksum
            progress_bar.update(decoded_frames)
    maybe_synchronize(device)
    elapsed_seconds: float = time.perf_counter() - start_time
    checksum: int = int(checksum_tensor.item()) if checksum_tensor is not None else 0
    return TimingResult(
        label=f"TorchCodec {device}",
        elapsed_seconds=elapsed_seconds,
        frames=frames,
        detail=f"chunk_size={chunk_size}, seek={seek_mode}, checksum={checksum}",
    )


def benchmark_existing(video_path: Path, max_frames: int | None) -> TimingResult:
    """Decode with the existing OpenCV-backed VideoReader."""
    start_time: float = time.perf_counter()
    reader: VideoReader = VideoReader(video_path)
    frame_count: int = len(reader) if max_frames is None else min(max_frames, len(reader))
    frames: int = 0
    checksum: int = 0
    for bgr_frame in tqdm(islice(reader, frame_count), total=frame_count, desc="VideoReader", unit=" frames", mininterval=5.0):
        bgr: ImageBGR = bgr_frame
        frames += 1
        checksum += int(bgr[0, 0, 0])
    elapsed_seconds: float = time.perf_counter() - start_time
    return TimingResult(
        label="VideoReader",
        elapsed_seconds=elapsed_seconds,
        frames=frames,
        detail=f"streamed BGR numpy frames, checksum={checksum}",
    )


def benchmark_existing_tensor(
    video_path: Path,
    max_frames: int | None,
    device: str,
    chunk_size: int,
) -> TimingResult:
    """Decode with VideoReader and convert chunks to RGB ``NCHW`` uint8 tensors."""
    validate_chunk_size(chunk_size)
    maybe_synchronize(device)
    start_time: float = time.perf_counter()
    reader: VideoReader = VideoReader(video_path)
    frame_count: int = len(reader) if max_frames is None else min(max_frames, len(reader))
    frames: int = 0
    checksum_tensor: Int64[torch.Tensor, ""] | None = None
    bgr_chunk: list[ImageBGR] = []
    with tqdm(
        total=frame_count,
        desc=f"VideoReader -> {device} ({chunk_size} frames/chunk)",
        unit=" frames",
        mininterval=5.0,
    ) as progress_bar:
        for bgr_frame in islice(reader, frame_count):
            bgr: ImageBGR = bgr_frame
            bgr_chunk.append(bgr)
            if len(bgr_chunk) == chunk_size:
                video: UInt8[torch.Tensor, "b 3 h w"] = bgr_hwc_frames_to_rgb_nchw_tensor(bgr_chunk, device=device)
                decoded_frames: int = video.shape[0]
                frames += decoded_frames
                partial_checksum: Int64[torch.Tensor, ""] = video[:, 0, 0, 0].sum(dtype=torch.int64)
                checksum_tensor = partial_checksum if checksum_tensor is None else checksum_tensor + partial_checksum
                bgr_chunk = []
                progress_bar.update(decoded_frames)
        if bgr_chunk:
            final_video: UInt8[torch.Tensor, "b 3 h w"] = bgr_hwc_frames_to_rgb_nchw_tensor(bgr_chunk, device=device)
            final_frame_count: int = final_video.shape[0]
            frames += final_frame_count
            final_checksum: Int64[torch.Tensor, ""] = final_video[:, 0, 0, 0].sum(dtype=torch.int64)
            checksum_tensor = final_checksum if checksum_tensor is None else checksum_tensor + final_checksum
            progress_bar.update(final_frame_count)
    maybe_synchronize(device)
    elapsed_seconds: float = time.perf_counter() - start_time
    checksum: int = int(checksum_tensor.item()) if checksum_tensor is not None else 0
    return TimingResult(
        label=f"VideoReader -> {device}",
        elapsed_seconds=elapsed_seconds,
        frames=frames,
        detail=f"RGB NCHW uint8 tensor, chunk_size={chunk_size}, checksum={checksum}",
    )


def main(config: BenchmarkConfig) -> None:
    """Run the TorchCodec CUDA singleview benchmark."""
    validate_chunk_size(config.chunk_size)

    video_path: Path = validate_video_file_path(config.video_path)
    metadata: TorchCodecVideoMetadata = read_torchcodec_video_metadata([video_path])
    full_frame_count: int = metadata.frame_count
    height: int = metadata.height
    width: int = metadata.width
    max_frames: int | None = None if config.max_frames < 0 else config.max_frames
    decode_frame_count: int = full_frame_count if max_frames is None else min(max_frames, full_frame_count)
    required_bytes: int = decode_frame_count * 3 * height * width

    CONSOLE.print(f"Video: {video_path}", markup=False)
    CONSOLE.print(f"Resolution: {width}x{height}, {full_frame_count} frames", markup=False)
    CONSOLE.print(f"Full tensor footprint: {format_bytes(required_bytes)}")
    chunk_frame_count: int = min(config.chunk_size, decode_frame_count)
    chunk_bytes: int = chunk_frame_count * 3 * height * width
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
        fallback_statuses: list[str] = read_torchcodec_fallback_statuses([video_path], config.device)
        fallback_status: str = fallback_statuses[0]
        CONSOLE.print(f"TorchCodec fallback status: {fallback_status}", markup=False)

    timing_results: list[TimingResult] = []
    if not config.skip_torchcodec:
        torchcodec_result: TimingResult = benchmark_torchcodec_chunked(
            video_path,
            config.device,
            max_frames,
            config.chunk_size,
            config.ffmpeg_threads,
            config.seek_mode,
        )
        timing_results.append(torchcodec_result)

    if not config.skip_existing:
        existing_tensor_result: TimingResult = benchmark_existing_tensor(
            video_path,
            max_frames,
            config.device,
            config.chunk_size,
        )
        timing_results.append(existing_tensor_result)

        existing_result: TimingResult = benchmark_existing(video_path, max_frames)
        timing_results.append(existing_result)

    print_timing_results(timing_results)
