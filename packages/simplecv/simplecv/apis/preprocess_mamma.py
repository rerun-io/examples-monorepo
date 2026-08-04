"""Build the AV1 yuv420 mirror for downloaded MAMMA sequences.

MAMMA ships yuv444 video (H.264 CRF5 originals, H.265 CRF16/24 variants,
H.265 Rext ``videos_light`` for iPhones). NVDEC and the rerun viewer have no
4:4:4 fast path (see docs/video_decode_format_tradeoffs.md), so every camera
clip is re-encoded once to AV1 Main yuv420p into a ``videos_av1/`` dir next to
its source dir. The MAMMA exoego adapter prefers ``videos_av1`` automatically.

Existing valid targets are probed and skipped, mirroring the EPFL mirror tool.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from simplecv.apis.preprocess_epfl_smart_kitchen import VideoProbe, probe_video
from simplecv.configs.dataset_paths import MAMMA_SOURCE_ROOT
from simplecv.data.exoego.mamma import VIDEO_DIR_PREFERENCE, discover_sequence_names


@dataclass
class PreprocessConfig:
    """Configuration for the MAMMA AV1 yuv420 re-encode."""

    root_directory: Path = MAMMA_SOURCE_ROOT
    """Root containing downloaded MAMMA sequences (see download_mamma)."""
    target_dir_name: str = "videos_av1"
    """Per-sequence output dir for the AV1 yuv420 mirror."""
    encoder: Literal["av1_nvenc", "libsvtav1"] = "av1_nvenc"
    """AV1 encoder: NVENC hardware (default) or SVT-AV1 CPU fallback."""
    preset: str | None = None
    """Encoder preset (`p1`-`p7` for av1_nvenc, `0`-`13` for libsvtav1). None picks the encoder default (`p5` / `8`)."""
    cq: int = 30
    """Constant-quality target (NVENC `-cq` / SVT-AV1 `-crf`); repo storage default is 30."""
    force: bool = False
    """Re-encode even when a valid target already exists."""
    ffmpeg_path: str = "ffmpeg"
    """ffmpeg executable."""
    ffprobe_path: str = "ffprobe"
    """ffprobe executable."""
    duration_tolerance_sec: float = 0.25
    """Maximum source/target duration difference before validation fails."""


def _resolve_preset(config: PreprocessConfig) -> str:
    """Resolve (and validate) the preset for the chosen encoder.

    Returns the encoder default when unset (``p5`` for av1_nvenc, ``8`` for
    libsvtav1); raises when an explicit preset does not match the encoder's form
    (av1_nvenc wants ``pN``, libsvtav1 wants an integer) instead of silently
    substituting one.
    """
    if config.encoder == "av1_nvenc":
        if config.preset is None:
            return "p5"
        if not (config.preset.startswith("p") and config.preset[1:].isdigit()):
            raise ValueError(f"av1_nvenc preset must look like 'p1'..'p7', got {config.preset!r}")
        return config.preset
    if config.preset is None:
        return "8"
    if not config.preset.isdigit():
        raise ValueError(f"libsvtav1 preset must be an integer 0..13, got {config.preset!r}")
    return config.preset


def _encode_command(source_path: Path, tmp_path: Path, config: PreprocessConfig) -> list[str]:
    """Build the ffmpeg AV1 yuv420 command for one video."""
    preset: str = _resolve_preset(config)
    if config.encoder == "av1_nvenc":
        encoder_args: list[str] = ["-c:v", "av1_nvenc", "-preset", preset, "-tune", "hq", "-rc", "vbr", "-cq", str(config.cq), "-b:v", "0"]
    else:
        encoder_args = ["-c:v", "libsvtav1", "-preset", preset, "-crf", str(config.cq)]
    return [
        config.ffmpeg_path,
        "-hide_banner",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        # Force the mp4 muxer: the tmp file has a non-.mp4 suffix so a crashed
        # encode never leaves a partial file that the adapter's *.mp4 glob picks up.
        "-f",
        "mp4",
        str(tmp_path),
    ]


def _validate_target(source_probe: VideoProbe, target_probe: VideoProbe, duration_tolerance_sec: float) -> None:
    """Raise unless the target is a same-size, same-duration AV1 yuv420p stream."""
    if target_probe.codec_name != "av1":
        raise ValueError(f"Expected AV1 target codec, got {target_probe.codec_name}: {target_probe.path}")
    if target_probe.pix_fmt != "yuv420p":
        raise ValueError(f"Expected yuv420p target pixel format, got {target_probe.pix_fmt}: {target_probe.path}")
    if (target_probe.width, target_probe.height) != (source_probe.width, source_probe.height):
        raise ValueError(
            f"Target dimensions do not match source: {source_probe.width}x{source_probe.height} -> {target_probe.width}x{target_probe.height}"
        )
    if source_probe.duration_sec is not None and target_probe.duration_sec is not None:
        duration_delta_sec: float = abs(source_probe.duration_sec - target_probe.duration_sec)
        if duration_delta_sec > duration_tolerance_sec:
            raise ValueError(f"Target duration differs from source by {duration_delta_sec:.3f}s (tolerance {duration_tolerance_sec:.3f}s)")


def encode_video(source_path: Path, target_path: Path, config: PreprocessConfig) -> str:
    """Encode one video (or validate an existing target); returns 'encoded' or 'skipped'."""
    source_probe: VideoProbe = probe_video(source_path, ffprobe_path=config.ffprobe_path)
    if target_path.exists() and not config.force:
        target_probe: VideoProbe = probe_video(target_path, ffprobe_path=config.ffprobe_path)
        _validate_target(source_probe, target_probe, config.duration_tolerance_sec)
        return "skipped"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Non-.mp4 suffix so a crashed/killed encode leaves nothing the adapter's
    # *.mp4 glob would treat as a real camera video.
    tmp_path: Path = target_path.with_name(f"{target_path.name}.part")
    tmp_path.unlink(missing_ok=True)
    command: list[str] = _encode_command(source_path, tmp_path, config)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg executable not found: {config.ffmpeg_path}") from exc
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        stderr: str = exc.stderr or ""
        if "av1_nvenc" in stderr:
            raise RuntimeError(
                "ffmpeg failed with av1_nvenc. Confirm this build has NVIDIA AV1 NVENC support "
                "and an AV1-capable GPU, or rerun with --encoder libsvtav1."
            ) from exc
        raise
    tmp_path.replace(target_path)

    target_probe = probe_video(target_path, ffprobe_path=config.ffprobe_path)
    _validate_target(source_probe, target_probe, config.duration_tolerance_sec)
    return "encoded"


def source_video_dir_for_sequence(sequence_dir: Path, target_dir_name: str) -> Path | None:
    """First existing source video dir by preference, excluding the AV1 target dir."""
    for name in VIDEO_DIR_PREFERENCE:
        if name == target_dir_name:
            continue
        video_dir: Path = sequence_dir / name
        if video_dir.is_dir() and any(video_dir.glob("*.mp4")):
            return video_dir
    return None


def main(config: PreprocessConfig) -> None:
    """Re-encode every downloaded MAMMA sequence's videos to AV1 yuv420."""
    sequence_names: list[str] = discover_sequence_names(config.root_directory)
    if not sequence_names:
        raise FileNotFoundError(f"No MAMMA sequences under {config.root_directory} (run simplecv-download-mamma first)")

    num_encoded: int = 0
    num_skipped: int = 0
    for sequence_name in sequence_names:
        sequence_dir: Path = config.root_directory / sequence_name
        source_dir: Path | None = source_video_dir_for_sequence(sequence_dir, config.target_dir_name)
        if source_dir is None:
            print(f"[warn] {sequence_name}: no source video dir, skipping")
            continue
        target_dir: Path = sequence_dir / config.target_dir_name
        for source_path in sorted(source_dir.glob("*.mp4")):
            start_time: float = time.perf_counter()
            status: str = encode_video(source_path, target_dir / source_path.name, config)
            elapsed_sec: float = time.perf_counter() - start_time
            print(f"[{status}] {sequence_name}/{source_dir.name}/{source_path.name} -> {config.target_dir_name}/ ({elapsed_sec:.1f}s)")
            num_encoded += int(status == "encoded")
            num_skipped += int(status == "skipped")

    print(f"\nDone. {num_encoded} encoded, {num_skipped} already valid, root: {config.root_directory}")
