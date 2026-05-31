from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from simplecv.video_io import MultiVideoReader, TorchCodecCudaMultiVideoReader, TorchCodecVideoChunk

DEFAULT_ROOT: Path = Path("/mnt/8tb/data/assembly101-original")
DEFAULT_SEQUENCE: str = "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724"
DEFAULT_EPFL_ROOT: Path = Path("/mnt/8tb/data/epfl-smart-kitchen-av1")
DEFAULT_EPFL_SPLIT: str = "train"
DEFAULT_EPFL_PARTICIPANT: str = "YH2002"
DEFAULT_EPFL_SESSION: str = "2023_12_04_10_15_23"
DEFAULT_CHUNK_SIZE: int = 32
DEFAULT_SEEK_MODE: Literal["exact", "approximate"] = "approximate"
EPFL_EXO_CAMERA_NAMES: tuple[str, ...] = (
    "output0",
    "Aoutput0",
    "Aoutput1",
    "Aoutput2",
    "Aoutput3",
    "Boutput0",
    "Boutput1",
    "Boutput2",
    "Boutput3",
)


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


def format_bytes(num_bytes: int) -> str:
    """Format a byte count."""
    value: float = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"


def assembly101_exo_rgb_paths(root: Path, sequence: str) -> list[Path]:
    """Return the 8 Assembly101 720p AV1 exo RGB videos."""
    video_dir: Path = root / "videos" / "av1-720-new" / sequence
    video_paths: list[Path] = sorted(video_dir.glob("C*_rgb_low.mp4"))
    if len(video_paths) != 8:
        raise ValueError(f"Expected 8 exo RGB videos in {video_dir}, found {len(video_paths)}")
    return video_paths


def epfl_exo_rgb_paths(root: Path, split: str, participant: str, session: str) -> list[Path]:
    """Return the 9 EPFL-Smart-Kitchen exo RGB AV1 videos."""
    video_dir: Path = root / "Public_release_videos" / split / participant / session / "videos"
    video_paths: list[Path] = [video_dir / f"{camera_name}.mp4" for camera_name in EPFL_EXO_CAMERA_NAMES]
    missing_paths: list[Path] = [video_path for video_path in video_paths if not video_path.exists()]
    if missing_paths:
        missing_text: str = ", ".join(path.name for path in missing_paths)
        raise FileNotFoundError(f"Missing EPFL exo RGB videos in {video_dir}: {missing_text}")
    return video_paths


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
    """Decode the full multiview sequence in GPU-resident chunks."""
    maybe_synchronize(device)
    start: float = time.perf_counter()
    reader: TorchCodecCudaMultiVideoReader = TorchCodecCudaMultiVideoReader(
        video_paths,
        device=device,
        num_workers=num_workers,
        num_ffmpeg_threads=num_ffmpeg_threads,
        seek_mode=seek_mode,
    )
    frame_count: int = len(reader) if max_frames is None else min(max_frames, len(reader))
    total_chunks: int = (frame_count + chunk_size - 1) // chunk_size
    camera_frames: int = 0
    checksum_tensor: torch.Tensor | None = None
    for chunk in tqdm(
        reader.iter_chunks(chunk_size=chunk_size, max_frames=max_frames),
        total=total_chunks,
        desc="TorchCodec chunks",
        unit="chunk",
        mininterval=5.0,
    ):
        chunk_data: TorchCodecVideoChunk = chunk
        for video in chunk_data.videos:
            camera_frames += int(video.shape[0])
            partial_checksum: torch.Tensor = video[:, 0, 0, 0].sum(dtype=torch.int64)
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


def positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    parsed_value: int = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed_value


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


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("assembly101", "epfl"), default="assembly101")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--sequence", type=str, default=DEFAULT_SEQUENCE)
    parser.add_argument("--epfl-split", type=str, default=DEFAULT_EPFL_SPLIT)
    parser.add_argument("--epfl-participant", type=str, default=DEFAULT_EPFL_PARTICIPANT)
    parser.add_argument("--epfl-session", type=str, default=DEFAULT_EPFL_SESSION)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-frames", type=int, default=120, help="Frames per video to decode; use -1 for all frames.")
    parser.add_argument("--chunk-size", type=positive_int, default=DEFAULT_CHUNK_SIZE, help="Frames per video per TorchCodec chunk.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ffmpeg-threads", type=int, default=0)
    parser.add_argument("--seek-mode", choices=("exact", "approximate"), default=DEFAULT_SEEK_MODE)
    parser.add_argument("--check-fallback", action="store_true")
    parser.add_argument("--skip-torchcodec", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser: argparse.ArgumentParser = build_parser()
    args = parser.parse_args()

    root: Path = args.root if args.root is not None else (DEFAULT_EPFL_ROOT if args.dataset == "epfl" else DEFAULT_ROOT)
    if args.dataset == "epfl":
        dataset_label: str = f"EPFL-Smart-Kitchen {args.epfl_split}/{args.epfl_participant}/{args.epfl_session}"
        video_paths: list[Path] = epfl_exo_rgb_paths(root, args.epfl_split, args.epfl_participant, args.epfl_session)
    else:
        dataset_label = f"Assembly101 sequence: {args.sequence}"
        video_paths = assembly101_exo_rgb_paths(root, args.sequence)
    full_frame_count, height, width = torchcodec_frame_count(video_paths)
    max_frames: int | None = None if args.max_frames < 0 else args.max_frames
    decode_frame_count: int = full_frame_count if max_frames is None else min(max_frames, full_frame_count)
    required_bytes: int = len(video_paths) * decode_frame_count * 3 * height * width

    print(dataset_label, flush=True)
    print(f"Videos: {len(video_paths)} x {width}x{height}, {full_frame_count} frames/video", flush=True)
    print(f"Full sequence tensor footprint: {format_bytes(required_bytes)}", flush=True)
    chunk_frame_count: int = min(args.chunk_size, decode_frame_count)
    chunk_bytes: int = len(video_paths) * chunk_frame_count * 3 * height * width
    print(f"Chunked TorchCodec tensor bytes: {format_bytes(chunk_bytes)} per chunk", flush=True)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"CUDA memory: {format_bytes(free_bytes)} free / {format_bytes(total_bytes)} total", flush=True)
    if args.dry_run:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false. Use --device cpu for a CPU smoke run.")
    if args.check_fallback:
        fallback_statuses: list[str] = torchcodec_fallback_status(video_paths, args.device)
        unique_statuses: str = " | ".join(sorted(set(fallback_statuses)))
        print(f"TorchCodec fallback status: {unique_statuses}", flush=True)

    if not args.skip_torchcodec:
        torchcodec_result: TimingResult = benchmark_torchcodec_chunked(
            video_paths,
            args.device,
            max_frames,
            args.workers,
            args.chunk_size,
            args.ffmpeg_threads,
            args.seek_mode,
        )
        print(flush=True)
        print_timing(torchcodec_result)

    if not args.skip_existing:
        existing_result: TimingResult = benchmark_existing(video_paths, max_frames)
        print_timing(existing_result)


if __name__ == "__main__":
    main()
