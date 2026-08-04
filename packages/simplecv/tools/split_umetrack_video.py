"""CLI for splitting stacked UmeTrack recordings into per-camera MP4 files.

This script expects each recording to contain a horizontally stacked H.264 video
(`recording_XX.mp4`) and a JSON metadata file (`recording_XX.json`). It produces
an output directory containing four cropped videos (`top_left.mp4`, etc.) plus a
copy of the JSON.

Usage (dry run by default, processes first candidate only):
    python tools/split_umetrack_video.py \
        --input-root /path/to/umetrack-data/raw_data \
        --output-root /mnt/nas/datasets/umetrack-split

Execute full batch:
    python tools/split_umetrack_video.py \
        --input-root ... \
        --output-root ... \
        --execute
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import tyro
from tqdm import tqdm

LOGGER = logging.getLogger("split_umetrack_video")

# Horizontal ordering defined by CAMERA_PANEL_ORDER in view_umetrack_data.py
CAMERA_STREAMS: Sequence[tuple[str, str]] = (
    ("TL", "top_left"),
    ("BL", "bottom_left"),
    ("BR", "bottom_right"),
    ("TR", "top_right"),
)


@dataclass(frozen=True)
class Recording:
    video_path: Path
    json_path: Path
    relative_parent: Path


@dataclass
class CLIArgs:
    input_root: Path
    output_root: Path
    pattern: str = "recording_*.mp4"
    limit: int | None = None
    execute: bool = False
    force: bool = False
    jobs: int = 1
    ffmpeg_bin: str = os.environ.get("FFMPEG_BIN", "ffmpeg")
    ffprobe_bin: str = os.environ.get("FFPROBE_BIN", "ffprobe")
    dry_run_recording: str | None = None
    codec: str = "av1_nvenc"
    crf: float | None = None
    preset: str = "p7"
    pix_fmt: str = "yuv420p"
    cq: float = 35.0
    bitrate: str = "0"
    rate_control: str = "vbr"
    no_progress: bool = False


def discover_recordings(input_root: Path, pattern: str) -> list[Recording]:
    recordings: list[Recording] = []
    for video_path in sorted(input_root.glob(f"**/{pattern}")):
        rel_parent = video_path.parent.relative_to(input_root)
        json_path = video_path.with_suffix(".json")
        if not json_path.exists():
            LOGGER.warning("Skipping %s (missing JSON sibling %s)", video_path, json_path.name)
            continue
        recordings.append(Recording(video_path=video_path, json_path=json_path, relative_parent=rel_parent))
    recordings.sort(key=lambda item: (str(item.relative_parent), item.video_path.name))
    return recordings


def ffprobe_video(ffprobe_bin: str, video_path: Path) -> dict[str, str | int | float]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{proc.stderr}")
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    nb_frames = int(stream.get("nb_frames", 0)) if stream.get("nb_frames") else 0
    r_frame_rate = stream.get("r_frame_rate", "0/1")
    return {
        "width": width,
        "height": height,
        "nb_frames": nb_frames,
        "r_frame_rate": r_frame_rate,
    }


def build_ffmpeg_command_for_panel(
    ffmpeg_bin: str,
    video_path: Path,
    segment_width: int,
    frame_height: int,
    panel_index: int,
    output_path: Path,
    codec: str,
    crf: float | None,
    preset: str | None,
    pix_fmt: str | None,
    cq: float | None,
    bitrate: str | None,
    rate_control: str | None,
) -> list[str]:
    x_offset = panel_index * segment_width
    crop_filter = f"crop={segment_width}:{frame_height}:{x_offset}:0"
    command: list[str] = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(video_path),
        "-vf",
        crop_filter,
        "-an",
    ]
    command.extend(["-c:v", codec])
    if crf is not None:
        command.extend(["-crf", str(crf)])
    if preset is not None:
        command.extend(["-preset", str(preset)])
    if pix_fmt is not None:
        command.extend(["-pix_fmt", pix_fmt])
    if cq is not None:
        command.extend(["-cq", str(cq)])
    if bitrate is not None:
        command.extend(["-b:v", bitrate])
    if rate_control is not None:
        command.extend(["-rc", rate_control])
    command.append(str(output_path))
    return command


def run_ffmpeg(command: Sequence[str]) -> None:
    proc = subprocess.run(command, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {' '.join(command)}")


def ensure_outputs(
    output_root: Path,
    recording: Recording,
) -> list[Path]:
    target_dir = output_root / recording.relative_parent / recording.video_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [target_dir / f"{name}.mp4" for _, name in CAMERA_STREAMS]
    return output_paths


def should_skip(output_paths: Sequence[Path], force: bool) -> bool:
    if force:
        return False
    return all(path.exists() for path in output_paths)


def copy_json(recording: Recording, output_root: Path, force: bool) -> Path:
    target_dir = output_root / recording.relative_parent / recording.video_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    target_json = target_dir / recording.json_path.name
    if target_json.exists() and not force:
        return target_json
    shutil.copy2(recording.json_path, target_json)
    return target_json


def process_recording(
    ffmpeg_bin: str,
    ffprobe_bin: str,
    recording: Recording,
    output_root: Path,
    force: bool,
    codec: str,
    crf: float | None,
    preset: str | None,
    pix_fmt: str | None,
    cq: float | None,
    bitrate: str | None,
    rate_control: str | None,
) -> dict:
    if codec == "copy":
        raise ValueError("Stream copy is incompatible with cropping. Choose an encoder (e.g. libx264, libsvtav1).")
    probe = ffprobe_video(ffprobe_bin=ffprobe_bin, video_path=recording.video_path)
    width = int(probe["width"])
    height = int(probe["height"])
    segments = len(CAMERA_STREAMS)
    if width % segments != 0:
        raise ValueError(f"{recording.video_path} width {width} not divisible by {segments}")
    segment_width = width // segments

    output_paths = ensure_outputs(output_root=output_root, recording=recording)
    if should_skip(output_paths, force=force):
        LOGGER.info("Skipping %s (outputs already exist)", recording.video_path)
        copy_json(recording, output_root=output_root, force=force)
        return {"status": "skipped"}

    # copy JSON ahead of video splitting
    copy_json(recording, output_root=output_root, force=force)

    for idx, (_panel_id, _panel_name) in enumerate(CAMERA_STREAMS):
        command = build_ffmpeg_command_for_panel(
            ffmpeg_bin=ffmpeg_bin,
            video_path=recording.video_path,
            segment_width=segment_width,
            frame_height=height,
            panel_index=idx,
            output_path=output_paths[idx],
            codec=codec,
            crf=crf,
            preset=preset,
            pix_fmt=pix_fmt,
            cq=cq,
            bitrate=bitrate,
            rate_control=rate_control,
        )
        run_ffmpeg(command)
    return {
        "status": "processed",
        "codec": codec,
        "crf": str(crf) if crf is not None else "",
        "cq": str(cq) if cq is not None else "",
        "frames": str(probe.get("nb_frames", "")),
        "frame_rate": probe.get("r_frame_rate", "0/1"),
    }


def dry_run_summary(recording: Recording, probe: dict[str, str | int | float], output_paths: Sequence[Path]) -> str:
    details = [
        f"Recording: {recording.video_path}",
        f"JSON:      {recording.json_path}",
        f"Rel dir:   {recording.relative_parent}",
        f"Width:     {probe['width']} px",
        f"Height:    {probe['height']} px",
        f"Frame rate:{probe['r_frame_rate']}",
        f"Frames:    {probe['nb_frames']}",
        "Planned outputs:",
    ]
    details.extend(f"  - {path}" for path in output_paths)
    return "\n".join(details)


def run_dry_run(
    ffprobe_bin: str,
    recording: Recording,
    output_root: Path,
) -> None:
    probe = ffprobe_video(ffprobe_bin=ffprobe_bin, video_path=recording.video_path)
    outputs = ensure_outputs(output_root=output_root, recording=recording)
    summary = dry_run_summary(recording=recording, probe=probe, output_paths=outputs)
    print(summary)
    print("\nNo files were written. Re-run with --execute to process all recordings.")


def execute_batch(
    ffmpeg_bin: str,
    ffprobe_bin: str,
    recordings: Sequence[Recording],
    output_root: Path,
    force: bool,
    jobs: int,
    codec: str,
    crf: float | None,
    preset: str | None,
    pix_fmt: str | None,
    cq: float | None,
    bitrate: str | None,
    rate_control: str | None,
    show_progress: bool,
) -> None:
    if jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if jobs == 1:
        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(total=len(recordings), unit="rec", desc="Splitting")
        for recording in recordings:
            LOGGER.info("Processing %s", recording.video_path)
            start_time = perf_counter()
            result = process_recording(
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                recording=recording,
                output_root=output_root,
                force=force,
                codec=codec,
                crf=crf,
                preset=preset,
                pix_fmt=pix_fmt,
                cq=cq,
                bitrate=bitrate,
                rate_control=rate_control,
            )
            LOGGER.info(
                "%s -> %s (%s, cq=%s) [%.2fs]",
                recording.video_path.stem,
                result["status"],
                result.get("codec"),
                result.get("cq"),
                perf_counter() - start_time,
            )
            if progress is not None:
                progress.update()
        if progress is not None:
            progress.close()
        return

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(total=len(recordings), unit="rec", desc="Splitting")
        futures = {
            executor.submit(
                process_recording,
                ffmpeg_bin,
                ffprobe_bin,
                recording,
                output_root,
                force,
                codec,
                crf,
                preset,
                pix_fmt,
                cq,
                bitrate,
                rate_control,
            ): recording
            for recording in recordings
        }
        for future in as_completed(futures):
            recording = futures[future]
            try:
                start_time = perf_counter()
                result = future.result()
                LOGGER.info(
                    "%s -> %s (%s, cq=%s) [%.2fs]",
                    recording.video_path.stem,
                    result["status"],
                    result.get("codec"),
                    result.get("cq"),
                    perf_counter() - start_time,
                )
            except Exception as exc:
                LOGGER.error("Failed to process %s: %s", recording.video_path, exc)
                raise
            finally:
                if progress is not None:
                    progress.update()
        if progress is not None:
            progress.close()


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def main() -> int:
    configure_logging()
    args = tyro.cli(CLIArgs)
    input_root: Path = args.input_root.resolve()
    output_root: Path = args.output_root.resolve()

    if not input_root.exists():
        LOGGER.error("Input root %s does not exist.", input_root)
        return 1

    recordings = discover_recordings(input_root=input_root, pattern=args.pattern)
    if not recordings:
        LOGGER.error("No recordings matched pattern '%s' under %s", args.pattern, input_root)
        return 1

    if args.limit is not None and args.limit > 0:
        recordings = recordings[: args.limit]

    if not args.execute:
        if args.dry_run_recording:
            matches = [
                rec for rec in recordings if str(rec.relative_parent / rec.video_path.name) == args.dry_run_recording
            ]
            if matches:
                dry_candidate = matches[0]
            else:
                LOGGER.warning("dry-run recording %s not found; defaulting to first match.", args.dry_run_recording)
                dry_candidate = recordings[0]
        else:
            dry_candidate = recordings[0]
        run_dry_run(ffprobe_bin=args.ffprobe_bin, recording=dry_candidate, output_root=output_root)
        return 0

    execute_batch(
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
        recordings=recordings,
        output_root=output_root,
        force=args.force,
        jobs=args.jobs,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        pix_fmt=args.pix_fmt,
        cq=args.cq,
        bitrate=args.bitrate,
        rate_control=args.rate_control,
        show_progress=not args.no_progress,
    )
    LOGGER.info("Completed splitting for %d recording(s).", len(recordings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
