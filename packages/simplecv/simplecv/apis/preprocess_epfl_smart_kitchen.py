"""Build an AV1-transcoded EPFL-Smart-Kitchen mirror for SimpleCV."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

EPFL_SOURCE_ROOT: Path = Path("/mnt/8tb/data/epfl-smart-kitchen")
EPFL_AV1_MIRROR_ROOT: Path = Path("/mnt/8tb/data/epfl-smart-kitchen-av1")
VIDEO_DIR_NAMES: tuple[str, str] = ("videos", "videos_depth")
GENERATED_VIDEO_SUFFIXES: tuple[str, str] = (".rerun_h264.mp4", ".rerun_av1.mp4")
GENERATED_FILENAMES: set[str] = {"av1_encode_report.md", "av1_encode_report.json"}
GENERATED_DIR_SUFFIXES: tuple[str, ...] = ("-av1-test",)
EXCLUDED_ARCHIVE_SUFFIXES: tuple[str, ...] = (".zip",)


@dataclass(frozen=True)
class MirrorCopyJob:
    """A byte-for-byte metadata or annotation copy in the AV1 mirror."""

    source_path: Path
    """Source file in the original EPFL tree."""
    target_path: Path
    """Target file in the AV1 mirror tree."""
    relative_path: Path
    """Path relative to the dataset root."""


@dataclass(frozen=True)
class MirrorVideoJob:
    """A video transcode from source MP4 to AV1 MP4."""

    source_path: Path
    """Source MP4 in the original EPFL tree."""
    target_path: Path
    """Target AV1 MP4 in the mirror tree."""
    relative_path: Path
    """Path relative to the dataset root."""
    group: Literal["videos", "videos_depth"]
    """EPFL video directory type."""


@dataclass(frozen=True)
class EpflMirrorPlan:
    """Full file operation plan for an EPFL AV1 mirror."""

    source_root: Path
    """Original EPFL dataset root."""
    target_root: Path
    """AV1 mirror dataset root."""
    video_jobs: tuple[MirrorVideoJob, ...]
    """Video files to transcode to AV1."""
    copy_jobs: tuple[MirrorCopyJob, ...]
    """Non-video files to copy byte-for-byte."""


@dataclass(frozen=True)
class VideoProbe:
    """Selected ffprobe metadata for one video stream."""

    path: str
    """Video path."""
    codec_name: str
    """Video codec name reported by ffprobe."""
    width: int
    """Encoded width in pixels."""
    height: int
    """Encoded height in pixels."""
    pix_fmt: str
    """Pixel format."""
    duration_sec: float | None
    """Duration in seconds when available."""
    frame_count: int | None
    """Frame count when available."""
    size_bytes: int
    """File size in bytes."""


@dataclass(frozen=True)
class EncodeResult:
    """Manifest row for one AV1 mirror video."""

    relative_path: str
    """Path relative to the mirror root."""
    group: str
    """EPFL video directory type."""
    status: Literal["dry-run", "encoded", "skipped"]
    """Operation result."""
    source_codec: str
    """Source codec name."""
    target_codec: str
    """Target codec name."""
    width: int
    """Target width in pixels."""
    height: int
    """Target height in pixels."""
    source_frame_count: int | None
    """Source frame count when available."""
    target_frame_count: int | None
    """Target frame count when available."""
    source_duration_sec: float | None
    """Source duration when available."""
    target_duration_sec: float | None
    """Target duration when available."""
    source_size_bytes: int
    """Source file size in bytes."""
    target_size_bytes: int
    """Target file size in bytes."""
    elapsed_sec: float
    """Wall-clock encode or validation time."""
    message: str = ""
    """Additional validation notes."""


@dataclass
class PreprocessConfig:
    """Configuration for building the EPFL-Smart-Kitchen AV1 mirror."""

    source_root: Path = EPFL_SOURCE_ROOT
    """Original EPFL dataset root."""
    target_root: Path = EPFL_AV1_MIRROR_ROOT
    """Output root for the AV1-transcoded mirror."""
    sequence_key: str = ""
    """Optional single sequence filter, for example ``train/YH2002/2023_12_04_10_15_23``."""
    include_depth: bool = True
    """When ``True``, transcode ``videos_depth/*.mp4`` in addition to RGB/HoloLens videos."""
    num_workers: int = 1
    """Number of concurrent ffmpeg transcodes. Keep low for NVENC session limits."""
    force: bool = False
    """When ``True``, re-encode existing target MP4s."""
    dry_run: bool = False
    """When ``True``, print the plan without copying or encoding."""
    max_videos: int | None = None
    """Optional cap on video transcodes for smoke tests."""
    ffmpeg_path: str = "ffmpeg"
    """ffmpeg executable."""
    ffprobe_path: str = "ffprobe"
    """ffprobe executable."""
    preset: str = "p5"
    """NVENC AV1 preset."""
    tune: str = "hq"
    """NVENC tune."""
    cq: int = 38
    """NVENC constant quality value."""
    duration_tolerance_sec: float = 0.25
    """Allowed absolute duration difference between source and AV1 target."""
    count_frames: bool = False
    """When ``True``, ask ffprobe to decode/count frames for slower exact validation."""


def _is_generated_path(relative_path: Path) -> bool:
    """Return whether ``relative_path`` is generated output that must not enter the mirror."""
    if relative_path.name in GENERATED_FILENAMES:
        return True
    if relative_path.suffix.lower() in EXCLUDED_ARCHIVE_SUFFIXES:
        return True
    if any(part.endswith(GENERATED_DIR_SUFFIXES) for part in relative_path.parts):
        return True
    return relative_path.name.endswith(GENERATED_VIDEO_SUFFIXES)


def _is_video_path(relative_path: Path, *, include_depth: bool) -> bool:
    """Return whether ``relative_path`` is an EPFL video payload file."""
    if relative_path.suffix.lower() != ".mp4":
        return False
    parent_name: str = relative_path.parent.name
    if parent_name == "videos":
        return True
    return bool(include_depth and parent_name == "videos_depth")


def _matches_sequence_key(relative_path: Path, sequence_key: str) -> bool:
    """Return whether ``relative_path`` belongs to an optional EPFL sequence key."""
    if not sequence_key:
        return True
    sequence_relative_path: Path = Path(sequence_key)
    prefixes: tuple[Path, Path] = (
        Path("Public_release_videos") / sequence_relative_path,
        Path("Public_release_pose") / sequence_relative_path,
    )
    return any(relative_path == prefix or relative_path.is_relative_to(prefix) for prefix in prefixes)


def build_mirror_plan(
    *,
    source_root: Path,
    target_root: Path,
    sequence_key: str = "",
    include_depth: bool = True,
    max_videos: int | None = None,
) -> EpflMirrorPlan:
    """Build the copy/transcode plan for an EPFL AV1 mirror.

    Args:
        source_root: Original EPFL dataset root.
        target_root: AV1 mirror target root.
        sequence_key: Optional ``split/participant/session`` filter.
        include_depth: Include ``videos_depth/*.mp4`` when true.
        max_videos: Optional cap for smoke-test encodes.

    Returns:
        Deterministic mirror operation plan.
    """
    if not source_root.exists():
        raise FileNotFoundError(f"EPFL source root does not exist: {source_root}")
    if source_root.resolve() == target_root.resolve():
        raise ValueError("EPFL AV1 mirror target_root must differ from source_root")

    video_jobs: list[MirrorVideoJob] = []
    copy_jobs: list[MirrorCopyJob] = []
    source_paths: list[Path] = sorted(path for path in source_root.rglob("*") if path.is_file())
    for source_path in source_paths:
        relative_path: Path = source_path.relative_to(source_root)
        if _is_generated_path(relative_path):
            continue
        if not _matches_sequence_key(relative_path, sequence_key):
            continue
        target_path: Path = target_root / relative_path
        if _is_video_path(relative_path, include_depth=include_depth):
            group: Literal["videos", "videos_depth"] = "videos_depth" if relative_path.parent.name == "videos_depth" else "videos"
            video_jobs.append(
                MirrorVideoJob(
                    source_path=source_path,
                    target_path=target_path,
                    relative_path=relative_path,
                    group=group,
                )
            )
        else:
            copy_jobs.append(
                MirrorCopyJob(
                    source_path=source_path,
                    target_path=target_path,
                    relative_path=relative_path,
                )
            )
    if max_videos is not None:
        video_jobs = video_jobs[:max_videos]
    return EpflMirrorPlan(
        source_root=source_root,
        target_root=target_root,
        video_jobs=tuple(video_jobs),
        copy_jobs=tuple(copy_jobs),
    )


def dataset_card_text() -> str:
    """Return the Hugging Face dataset card for the EPFL AV1 mirror."""
    return """---
license: cc-by-nc-4.0
tags:
- video
- computer-vision
- epfl-smart-kitchen
- simplecv
- av1
pretty_name: EPFL-Smart-Kitchen AV1 SimpleCV Mirror
---

# EPFL-Smart-Kitchen AV1 SimpleCV Mirror

This is an AV1-transcoded SimpleCV-compatible mirror of the EPFL-Smart-Kitchen-30 dataset.

Original data:

- Collected videos/data: https://zenodo.org/records/15535461
- Poses/annotations: https://zenodo.org/records/15551913
- GitHub: https://github.com/amathislab/EPFL-Smart-Kitchen

## Contents

- RGB and HoloLens videos transcoded to AV1 MP4
- Depth videos transcoded to AV1 MP4
- Metadata, timestamps, IMUs, poses, and annotations preserved from the local release
- Manifests for sequence inventory and encode validation

## Important Notes

- Videos were transcoded from the original MP4 release using NVIDIA NVENC AV1.
- This mirror is intended for non-commercial research use.
- Use the official Zenodo records for the canonical release.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
See `LICENSE`.
"""


def license_text() -> str:
    """Return a short license pointer for the mirrored dataset."""
    return """Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

See https://creativecommons.org/licenses/by-nc/4.0/

The original EPFL-Smart-Kitchen Zenodo records describe the public release as CC BY-NC 4.0.
"""


def _run_json(command: list[str]) -> dict[str, Any]:
    """Run a command that emits JSON and return the parsed object."""
    process: subprocess.CompletedProcess[str] = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    data: dict[str, Any] = json.loads(process.stdout)
    return data


def _optional_int(value: object) -> int | None:
    """Parse an optional ffprobe integer field."""
    if value is None or value == "N/A":
        return None
    return int(str(value))


def _optional_float(value: object) -> float | None:
    """Parse an optional ffprobe float field."""
    if value is None or value == "N/A":
        return None
    return float(str(value))


def probe_video(video_path: Path, *, ffprobe_path: str = "ffprobe", count_frames: bool = False) -> VideoProbe:
    """Probe one video's codec, dimensions, frame count, duration, and size."""
    stream_entries: str = "stream=codec_name,width,height,pix_fmt,nb_frames,duration"
    command: list[str] = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
        stream_entries = "stream=codec_name,width,height,pix_fmt,nb_frames,nb_read_frames,duration"
    command.extend(
        [
            "-show_entries",
            stream_entries,
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(video_path),
        ]
    )
    data: dict[str, Any] = _run_json(
        command
    )
    streams: list[dict[str, Any]] = data.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")
    stream: dict[str, Any] = streams[0]
    format_info: dict[str, Any] = data.get("format", {})
    frame_count: int | None = _optional_int(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _optional_int(stream.get("nb_frames"))
    duration_sec: float | None = _optional_float(stream.get("duration"))
    if duration_sec is None:
        duration_sec = _optional_float(format_info.get("duration"))
    return VideoProbe(
        path=str(video_path),
        codec_name=str(stream["codec_name"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
        pix_fmt=str(stream["pix_fmt"]),
        duration_sec=duration_sec,
        frame_count=frame_count,
        size_bytes=int(format_info.get("size") or video_path.stat().st_size),
    )


def _validate_target_probe(
    *,
    source_probe: VideoProbe,
    target_probe: VideoProbe,
    duration_tolerance_sec: float,
) -> str:
    """Validate target AV1 stream compatibility and return non-fatal notes."""
    messages: list[str] = []
    if target_probe.codec_name != "av1":
        raise ValueError(f"Expected AV1 target codec, got {target_probe.codec_name}: {target_probe.path}")
    if target_probe.pix_fmt != "yuv420p":
        raise ValueError(f"Expected yuv420p target pixel format, got {target_probe.pix_fmt}: {target_probe.path}")
    if (target_probe.width, target_probe.height) != (source_probe.width, source_probe.height):
        raise ValueError(
            "Target dimensions do not match source: "
            f"{source_probe.width}x{source_probe.height} -> {target_probe.width}x{target_probe.height}"
        )
    if (
        source_probe.frame_count is not None
        and target_probe.frame_count is not None
        and source_probe.frame_count != target_probe.frame_count
    ):
        warnings.warn(
            f"Target frame count does not match source: {source_probe.frame_count} -> {target_probe.frame_count}",
            stacklevel=2,
        )
        messages.append("frame_count_mismatch")
    if source_probe.duration_sec is not None and target_probe.duration_sec is not None:
        duration_delta_sec: float = abs(source_probe.duration_sec - target_probe.duration_sec)
        if duration_delta_sec > duration_tolerance_sec:
            raise ValueError(
                f"Target duration differs from source by {duration_delta_sec:.3f}s, "
                f"tolerance is {duration_tolerance_sec:.3f}s"
            )
    if target_probe.size_bytes >= source_probe.size_bytes:
        messages.append("target_not_smaller_than_source")
    return ";".join(messages)


def _encode_command(job: MirrorVideoJob, tmp_path: Path, config: PreprocessConfig) -> list[str]:
    """Build the ffmpeg AV1 NVENC command for one mirror video."""
    return [
        config.ffmpeg_path,
        "-hide_banner",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(job.source_path),
        "-map",
        "0:v:0",
        "-c:v",
        "av1_nvenc",
        "-preset",
        config.preset,
        "-tune",
        config.tune,
        "-rc",
        "vbr",
        "-cq",
        str(config.cq),
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]


def encode_video(job: MirrorVideoJob, config: PreprocessConfig) -> EncodeResult:
    """Encode or validate one AV1 mirror video."""
    start_time: float = time.perf_counter()
    source_probe: VideoProbe = probe_video(
        job.source_path,
        ffprobe_path=config.ffprobe_path,
        count_frames=config.count_frames,
    )
    if job.target_path.exists() and not config.force:
        target_probe: VideoProbe = probe_video(
            job.target_path,
            ffprobe_path=config.ffprobe_path,
            count_frames=config.count_frames,
        )
        message: str = _validate_target_probe(
            source_probe=source_probe,
            target_probe=target_probe,
            duration_tolerance_sec=config.duration_tolerance_sec,
        )
        return _encode_result(
            job=job,
            status="skipped",
            source_probe=source_probe,
            target_probe=target_probe,
            elapsed_sec=time.perf_counter() - start_time,
            message=message,
        )

    job.target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path = job.target_path.with_name(f"{job.target_path.name}.tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()
    command: list[str] = _encode_command(job, tmp_path, config)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg executable not found: {config.ffmpeg_path}") from exc
    except subprocess.CalledProcessError as exc:
        stderr: str = exc.stderr or ""
        if "Unknown encoder 'av1_nvenc'" in stderr or "av1_nvenc" in stderr:
            raise RuntimeError(
                "ffmpeg failed to use the av1_nvenc encoder. Confirm this ffmpeg build has NVIDIA AV1 NVENC "
                "support and that the NVIDIA driver exposes an AV1-capable GPU."
            ) from exc
        raise
    tmp_path.replace(job.target_path)

    target_probe = probe_video(job.target_path, ffprobe_path=config.ffprobe_path, count_frames=config.count_frames)
    message = _validate_target_probe(
        source_probe=source_probe,
        target_probe=target_probe,
        duration_tolerance_sec=config.duration_tolerance_sec,
    )
    return _encode_result(
        job=job,
        status="encoded",
        source_probe=source_probe,
        target_probe=target_probe,
        elapsed_sec=time.perf_counter() - start_time,
        message=message,
    )


def _encode_result(
    *,
    job: MirrorVideoJob,
    status: Literal["encoded", "skipped"],
    source_probe: VideoProbe,
    target_probe: VideoProbe,
    elapsed_sec: float,
    message: str,
) -> EncodeResult:
    """Create an encode manifest row from probes."""
    return EncodeResult(
        relative_path=job.relative_path.as_posix(),
        group=job.group,
        status=status,
        source_codec=source_probe.codec_name,
        target_codec=target_probe.codec_name,
        width=target_probe.width,
        height=target_probe.height,
        source_frame_count=source_probe.frame_count,
        target_frame_count=target_probe.frame_count,
        source_duration_sec=source_probe.duration_sec,
        target_duration_sec=target_probe.duration_sec,
        source_size_bytes=source_probe.size_bytes,
        target_size_bytes=target_probe.size_bytes,
        elapsed_sec=elapsed_sec,
        message=message,
    )


def copy_metadata_files(copy_jobs: tuple[MirrorCopyJob, ...], *, force: bool) -> int:
    """Copy non-video files and return the number of files written."""
    copied_count: int = 0
    for job in tqdm(copy_jobs, desc="Copy metadata", unit="file"):
        should_copy: bool = force or not job.target_path.exists()
        if not should_copy:
            source_stat = job.source_path.stat()
            target_stat = job.target_path.stat()
            should_copy = source_stat.st_size != target_stat.st_size or source_stat.st_mtime > target_stat.st_mtime
        if should_copy:
            job.target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(job.source_path, job.target_path)
            copied_count += 1
    return copied_count


def _sequence_key_from_relative_path(relative_path: Path) -> tuple[str, str, str] | None:
    """Extract ``split/participant/session`` from a mirrored EPFL relative path."""
    parts: tuple[str, ...] = relative_path.parts
    if len(parts) < 4 or parts[0] not in {"Public_release_pose", "Public_release_videos"}:
        return None
    return parts[1], parts[2], parts[3]


def write_sequence_manifest(plan: EpflMirrorPlan, manifest_dir: Path) -> Path:
    """Write a per-sequence inventory manifest."""
    rows_by_key: dict[tuple[str, str, str], dict[str, int | str | bool]] = {}
    for job in plan.video_jobs:
        key: tuple[str, str, str] | None = _sequence_key_from_relative_path(job.relative_path)
        if key is None:
            continue
        row: dict[str, int | str | bool] = rows_by_key.setdefault(
            key,
            {
                "split": key[0],
                "participant_id": key[1],
                "session_name": key[2],
                "has_pose": False,
                "num_rgb_videos": 0,
                "num_depth_videos": 0,
            },
        )
        if job.group == "videos_depth":
            row["num_depth_videos"] = int(row["num_depth_videos"]) + 1
        else:
            row["num_rgb_videos"] = int(row["num_rgb_videos"]) + 1
    for job in plan.copy_jobs:
        key = _sequence_key_from_relative_path(job.relative_path)
        if key is None or job.relative_path.parts[0] != "Public_release_pose":
            continue
        row = rows_by_key.setdefault(
            key,
            {
                "split": key[0],
                "participant_id": key[1],
                "session_name": key[2],
                "has_pose": False,
                "num_rgb_videos": 0,
                "num_depth_videos": 0,
            },
        )
        row["has_pose"] = True

    manifest_dir.mkdir(parents=True, exist_ok=True)
    path: Path = manifest_dir / "sequences.csv"
    fieldnames: list[str] = ["split", "participant_id", "session_name", "has_pose", "num_rgb_videos", "num_depth_videos"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows_by_key):
            writer.writerow(rows_by_key[key])
    return path


def write_upload_payload(plan: EpflMirrorPlan, manifest_dir: Path) -> Path:
    """Write the relative file list expected to be uploaded to Hugging Face."""
    payload_paths: set[str] = {
        "README.md",
        "LICENSE",
        "manifests/sequences.csv",
        "manifests/encode_report.csv",
        "manifests/encode_report.json",
        "manifests/upload_payload.txt",
    }
    payload_paths.update(job.relative_path.as_posix() for job in plan.video_jobs)
    payload_paths.update(job.relative_path.as_posix() for job in plan.copy_jobs)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path: Path = manifest_dir / "upload_payload.txt"
    path.write_text("\n".join(sorted(payload_paths)) + "\n")
    return path


def write_encode_reports(results: list[EncodeResult], manifest_dir: Path) -> tuple[Path, Path]:
    """Write CSV and JSON encode reports."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    csv_path: Path = manifest_dir / "encode_report.csv"
    json_path: Path = manifest_dir / "encode_report.json"
    rows: list[dict[str, Any]] = [asdict(result) for result in results]
    fieldnames: list[str] = list(rows[0].keys()) if rows else list(EncodeResult.__dataclass_fields__.keys())
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def write_dataset_files(target_root: Path) -> None:
    """Write top-level Hugging Face dataset files."""
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "README.md").write_text(dataset_card_text())
    (target_root / "LICENSE").write_text(license_text())


def run_plan(plan: EpflMirrorPlan, config: PreprocessConfig) -> list[EncodeResult]:
    """Execute copy and encode jobs for an EPFL AV1 mirror plan."""
    copied_count: int = copy_metadata_files(plan.copy_jobs, force=config.force)
    print(f"Copied metadata/annotation files: {copied_count}")

    results: list[EncodeResult] = []
    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [executor.submit(encode_video, job, config) for job in plan.video_jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Encode AV1", unit="video"):
            results.append(future.result())
    results.sort(key=lambda result: result.relative_path)
    return results


def main(config: PreprocessConfig) -> None:
    """Build the EPFL AV1 mirror according to ``config``."""
    plan: EpflMirrorPlan = build_mirror_plan(
        source_root=config.source_root,
        target_root=config.target_root,
        sequence_key=config.sequence_key,
        include_depth=config.include_depth,
        max_videos=config.max_videos,
    )
    print(f"Source: {plan.source_root}")
    print(f"Target: {plan.target_root}")
    print(f"Copy files: {len(plan.copy_jobs)}")
    print(f"Video transcodes: {len(plan.video_jobs)}")
    if config.dry_run:
        return

    effective_config: PreprocessConfig = replace(config)
    results: list[EncodeResult] = run_plan(plan, effective_config)
    manifest_dir: Path = config.target_root / "manifests"
    write_dataset_files(config.target_root)
    write_sequence_manifest(plan, manifest_dir)
    write_encode_reports(results, manifest_dir)
    write_upload_payload(plan, manifest_dir)
    encoded_count: int = sum(1 for result in results if result.status == "encoded")
    skipped_count: int = sum(1 for result in results if result.status == "skipped")
    source_bytes: int = sum(result.source_size_bytes for result in results)
    target_bytes: int = sum(result.target_size_bytes for result in results)
    saved_fraction: float = 1.0 - (target_bytes / source_bytes) if source_bytes > 0 else 0.0
    print(f"Encoded: {encoded_count}; skipped valid existing: {skipped_count}")
    print(f"Video bytes: {source_bytes} -> {target_bytes} ({saved_fraction:.1%} saved)")
