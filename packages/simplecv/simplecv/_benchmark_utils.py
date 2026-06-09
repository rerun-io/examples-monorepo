from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import torch
from rich import box
from rich.console import Console
from rich.table import Table

from simplecv.video_io import required_torchcodec_int

CONSOLE_WIDTH: int = 140
CONSOLE: Console = Console(width=CONSOLE_WIDTH)


@dataclass(frozen=True, slots=True)
class TorchCodecVideoMetadata:
    """TorchCodec video metadata needed by decode benchmarks."""

    frame_count: int
    """Number of frames in the benchmarked sequence."""
    height: int
    """Frame height in pixels."""
    width: int
    """Frame width in pixels."""


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Decode benchmark timing."""

    label: str
    """Reader label."""
    elapsed_seconds: float
    """Wall-clock seconds for the timed decode path."""
    frames: int
    """Total decoded frames, or camera-frames for multiview benchmarks."""
    detail: str
    """Short detail string for the output table."""


def validate_chunk_size(chunk_size: int) -> None:
    """Validate a chunk size before constructing benchmark readers."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")


def maybe_synchronize(device: str) -> None:
    """Synchronize CUDA work when the selected device is CUDA."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def read_torchcodec_video_metadata(video_paths: Sequence[Path]) -> TorchCodecVideoMetadata:
    """Read min frame count, height, and width with TorchCodec metadata."""
    if not video_paths:
        raise ValueError("video_paths must contain at least one video")

    from torchcodec.decoders import VideoDecoder

    frame_counts: list[int] = []
    heights: list[int] = []
    widths: list[int] = []
    for video_path in video_paths:
        decoder: VideoDecoder = VideoDecoder(video_path, device="cpu", dimension_order="NCHW", num_ffmpeg_threads=0)
        frame_counts.append(required_torchcodec_int(decoder.metadata.num_frames, "num_frames"))
        heights.append(required_torchcodec_int(decoder.metadata.height, "height"))
        widths.append(required_torchcodec_int(decoder.metadata.width, "width"))
    metadata: TorchCodecVideoMetadata = TorchCodecVideoMetadata(
        frame_count=min(frame_counts),
        height=heights[0],
        width=widths[0],
    )
    return metadata


def read_torchcodec_fallback_statuses(video_paths: Sequence[Path], device: str) -> list[str]:
    """Return TorchCodec CPU fallback status strings for each video."""
    from torchcodec.decoders import VideoDecoder

    statuses: list[str] = []
    for video_path in video_paths:
        decoder: VideoDecoder = VideoDecoder(video_path, device=device, dimension_order="NCHW", num_ffmpeg_threads=0)
        statuses.append(str(getattr(decoder, "cpu_fallback", "unavailable")))
    return statuses


def build_timing_table(results: list[TimingResult], throughput_header: str = "FPS") -> Table:
    """Build the Rich timing results table."""
    if not results:
        raise ValueError("results must contain at least one timing result")

    baseline_result: TimingResult = results[0]
    baseline_fps: float = baseline_result.frames / baseline_result.elapsed_seconds
    table: Table = Table(
        title=f"Results (baseline: {baseline_result.label})",
        box=box.SIMPLE_HEAVY,
        border_style="bright_black",
        header_style="bold white",
        title_style="bold cyan",
        row_styles=["", "dim"],
        show_lines=False,
    )
    table.add_column("Reader", no_wrap=True, style="cyan")
    table.add_column("Time", justify="right", no_wrap=True, style="magenta")
    table.add_column(throughput_header, justify="right", no_wrap=True, style="green")
    table.add_column("Relative", justify="right", no_wrap=True, style="yellow")
    table.add_column("Details", style="white")
    for result in results:
        fps: float = result.frames / result.elapsed_seconds
        relative: float = fps / baseline_fps
        table.add_row(
            result.label,
            f"{result.elapsed_seconds:,.3f}s",
            f"{fps:,.1f}",
            f"{relative:.2f}x",
            result.detail,
        )
    return table


def format_timing_table(results: list[TimingResult], throughput_header: str = "FPS") -> str:
    """Format timing results as a readable comparison table."""
    console_buffer: StringIO = StringIO()
    console: Console = Console(file=console_buffer, width=CONSOLE_WIDTH, force_terminal=False, color_system=None)
    if not results:
        console.print("Results")
        console.print("No benchmark paths ran.")
        return console_buffer.getvalue().rstrip()

    console.print(build_timing_table(results, throughput_header=throughput_header))
    return console_buffer.getvalue().rstrip()


def print_timing_results(results: list[TimingResult], throughput_header: str = "FPS") -> None:
    """Print benchmark timing results with consistent empty-result handling."""
    CONSOLE.print()
    if results:
        CONSOLE.print(build_timing_table(results, throughput_header=throughput_header))
    else:
        CONSOLE.print(format_timing_table(results, throughput_header=throughput_header))
