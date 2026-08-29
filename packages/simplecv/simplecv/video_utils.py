import atexit
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Int
from numpy import ndarray


def frame_at(frame_timestamps_ns: Int[ndarray, "n"], time_ns: float) -> int:
    """Return the latest frame at or before a viewer time.

    Args:
        frame_timestamps_ns: Int[np.ndarray, "n"] monotonic frame timestamps in nanoseconds.
        time_ns: Viewer time in nanoseconds.

    Returns:
        Zero-based frame index, clamped to the available timeline.
    """
    frame_idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right")) - 1
    return max(0, min(frame_idx, int(frame_timestamps_ns.shape[0]) - 1))


def create_temp_video_from_img_dir(
    image_directory: Path,
    fps: int = 30,
    quality: Literal["low", "medium", "high", "max", "optimal"] = "optimal",
    delete_on_exit: bool = True,
    image_extension: Literal["jpg", "png"] = "jpg",  # jpg or png
    save_file: bool = False,
) -> Path:
    """
    Create a temporary H.264 video file using NVIDIA GPU acceleration.

    Args:
        image_directory: Path to directory with images
        fps: Frames per second
        quality: Quality preset
        delete_on_exit: Whether to delete the file when program exits
        image_extension: Image file extension (jpg or png)

    Returns:
        Path to the temporary video file
    """
    # Map quality settings to NVENC presets and CQ values
    quality_settings = {
        "low": ("p6", "30"),  # preset, cq value
        "medium": ("p4", "23"),
        "high": ("p2", "18"),
        "max": ("p1", "12"),
    }

    preset, cq = quality_settings[quality]

    # Create a temporary file with .mp4 extension
    if not save_file:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_path = Path(temp_file.name)

        # If requested, register for deletion when program exits
        if delete_on_exit:
            atexit.register(lambda p: p.unlink(missing_ok=True), output_path)
    else:
        output_path: Path = image_directory / "output.mp4"

    # Build ffmpeg command base
    cmd_base: list[str] = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        f"{str(image_directory)}/*.{image_extension}",
    ]

    cmd_encoder_specific: list[str] = []

    if quality == "optimal":
        # AV1 NVENC settings for "optimal" quality
        cmd_encoder_specific.extend(
            [
                "-c:v",
                "av1_nvenc",
                "-preset",
                "p5",  # Balanced preset for AV1 NVENC (can be tuned: p1-p7, higher is slower)
                "-cq",
                "30",  # Constant Quality level (CRF equivalent)
                "-g",
                "2",  # Keyframe interval
                "-pix_fmt",
                "yuv420p",  # Standard pixel format
                # You might need to adjust or add other AV1 specific flags depending on your driver/ffmpeg version
                # e.g., -rc constqp -qp 30 for some rate control setups
            ]
        )
    else:
        # H.264 NVENC settings for other quality levels
        quality_settings = {
            "low": ("p6", "30"),  # preset, cq value
            "medium": ("p4", "23"),
            "high": ("p2", "18"),
            "max": ("p1", "12"),
        }
        preset, cq_h264 = quality_settings[quality]
        cmd_encoder_specific.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                preset,
                "-rc:v",
                "vbr_hq",  # High quality variable bitrate mode
                "-cq",
                cq_h264,  # Quality level
                "-b:v",
                "0",  # Let CQ control bitrate
                "-profile:v",
                "high",  # High profile for better compression
                "-g",
                "30",  # Keyframe interval for H.264
                "-bf",
                "3",  # Maximum 3 B-frames between reference frames
                "-pix_fmt",
                "yuv420p",  # Standard pixel format for compatibility
            ]
        )

    # Combine base command, encoder specific commands, and output path
    cmd: list[str] = cmd_base + cmd_encoder_specific + [str(output_path)]

    # Execute FFmpeg
    start_time = timer()
    process = subprocess.run(cmd, capture_output=True)
    end_time = timer()

    print(f"FFmpeg encoding completed in {end_time - start_time:.2f} seconds.")

    if process.returncode != 0:
        error_msg = process.stderr.decode()
        raise RuntimeError(f"FFmpeg encoding failed: {error_msg}")

    return output_path


Resolution: TypeAlias = Literal["1080p", "720p", "480p", "360p"]

RESOLUTION_MAP: dict[Resolution, tuple[int, int]] = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360),
}

@lru_cache(maxsize=1)
def _available_ffmpeg_video_encoders() -> set[str]:
    """
    Probe ffmpeg for available video encoders once and cache the result.

    Returns:
        A set containing encoder names (e.g. {"libsvtav1", "libx264"}).
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return set()

    if proc.returncode != 0 or not proc.stdout:
        return set()

    encoders: set[str] = set()
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith(("Encoders:", "------", "."))
            or " =" in line
        ):
            continue
        parts = line.split()
        if len(parts) >= 2:
            encoders.add(parts[1])

    return encoders


def _encoder_args_for(name: str, _available: set[str]) -> list[str]:
    """
    Map an encoder name to a list of ffmpeg CLI arguments.

    Args:
        name: encoder identifier (e.g. "libsvtav1")
        _available: set of encoders discovered in the local ffmpeg build

    Raises:
        ValueError: if no mapping exists for name.
    """
    if name == "av1_nvenc":
        return [
            "-c:v",
            "av1_nvenc",
            "-preset",
            os.getenv("SIMPLECV_NVENC_PRESET", "p5"),
            "-cq",
            os.getenv("SIMPLECV_NVENC_CQ", "30"),
            "-g",
            os.getenv("SIMPLECV_GOP", "2"),
            "-bf",
            os.getenv("SIMPLECV_NVENC_BFRAMES", "0"),
            "-pix_fmt",
            "yuv420p",
        ]

    if name == "av1_qsv":
        return [
            "-c:v",
            "av1_qsv",
            "-global_quality",
            os.getenv("SIMPLECV_QSV_QUALITY", "28"),
            "-look_ahead",
            os.getenv("SIMPLECV_QSV_LOOKAHEAD", "1"),
            "-g",
            os.getenv("SIMPLECV_GOP", "2"),
            "-pix_fmt",
            "yuv420p",
        ]

    if name == "av1_vaapi":
        return [
            "-hwaccel",
            "vaapi",
            "-vaapi_device",
            os.getenv("SIMPLECV_VAAPI_DEVICE", "/dev/dri/renderD128"),
            "-c:v",
            "av1_vaapi",
            "-global_quality",
            os.getenv("SIMPLECV_VAAPI_QUALITY", "28"),
            "-g",
            os.getenv("SIMPLECV_GOP", "2"),
            "-pix_fmt",
            "yuv420p",
        ]

    if name == "libsvtav1":
        preset = os.getenv("SIMPLECV_LIBSVT_PRESET", "12")
        svt_params = os.getenv("SIMPLECV_LIBSVT_PARAMS")
        args = [
            "-c:v",
            "libsvtav1",
            "-preset",
            preset,
            "-crf",
            os.getenv("SIMPLECV_LIBSVT_CRF", "30"),
            "-g",
            os.getenv("SIMPLECV_GOP", "2"),
            "-pix_fmt",
            "yuv420p",
        ]
        if svt_params:
            args += ["-svtav1-params", svt_params]
        return args

    if name == "libaom-av1":
        return [
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            os.getenv("SIMPLECV_LIBAOM_SPEED", "8"),
            "-crf",
            os.getenv("SIMPLECV_LIBAOM_CRF", "30"),
            "-b:v",
            os.getenv("SIMPLECV_LIBAOM_BITRATE", "0"),
            "-g",
            os.getenv("SIMPLECV_GOP", "2"),
            "-pix_fmt",
            "yuv420p",
        ]

    if name == "hevc_videotoolbox":
        args = [
            "-c:v",
            "hevc_videotoolbox",
            "-g",
            os.getenv("SIMPLECV_GOP", "2"),
            "-pix_fmt",
            "yuv420p",
        ]
        # Allow overriding profile/quality via env
        quality = os.getenv("SIMPLECV_VIDEOTOOLBOX_QUALITY")
        if quality:
            args += ["-global_quality", quality]
        vt_profile = os.getenv("SIMPLECV_VIDEOTOOLBOX_PROFILE")
        if vt_profile:
            args += ["-profile:v", vt_profile]
        # Tag ensures compatibility across Apple decoders.
        args += ["-tag:v", "hvc1"]
        return args

    if name == "h264_videotoolbox":
        args = [
            "-c:v",
            "h264_videotoolbox",
            "-g",
            os.getenv("SIMPLECV_GOP", "2"),
            "-pix_fmt",
            "yuv420p",
        ]
        quality = os.getenv("SIMPLECV_VIDEOTOOLBOX_QUALITY")
        if quality:
            args += ["-global_quality", quality]
        vt_profile = os.getenv("SIMPLECV_VIDEOTOOLBOX_PROFILE")
        if vt_profile:
            args += ["-profile:v", vt_profile]
        return args

    if name == "libx265":
        return [
            "-c:v",
            "libx265",
            "-preset",
            os.getenv("SIMPLECV_X265_PRESET", "medium"),
            "-crf",
            os.getenv("SIMPLECV_X265_CRF", "28"),
            "-g",
            os.getenv("SIMPLECV_GOP", "2"),
            "-pix_fmt",
            "yuv420p",
        ]

    if name == "libx264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            os.getenv("SIMPLECV_X264_PRESET", "medium"),
            "-crf",
            os.getenv("SIMPLECV_X264_CRF", "23"),
            "-g",
            os.getenv("SIMPLECV_GOP", "2"),
            "-pix_fmt",
            "yuv420p",
        ]

    raise ValueError(f"No encoder mapping defined for '{name}'")


@lru_cache(maxsize=1)
def _select_optimal_video_encoder_args() -> tuple[str, list[str]]:
    """
    Choose ffmpeg video encoder arguments suited for the current platform.

    Returns:
        Tuple containing the encoder name and ffmpeg CLI arguments (without
        output path or audio settings).
    """
    encoders = _available_ffmpeg_video_encoders()

    # Respect explicit user override when available.
    env_encoder = os.getenv("SIMPLECV_VIDEO_ENCODER")
    if env_encoder:
        if env_encoder not in encoders:
            raise RuntimeError(
                f"Requested encoder '{env_encoder}' not available. "
                f"Available encoders: {', '.join(sorted(encoders)) or 'none'}"
            )
        return env_encoder, _encoder_args_for(env_encoder, encoders)

    # Preferred order: NVIDIA NVENC → Intel QSV → VAAPI → libsvtav1 → libaom → Apple VT → x265 → x264
    priority = [
        "av1_nvenc",
        "av1_qsv",
        "av1_vaapi",
        "libsvtav1",
        "libaom-av1",
        "hevc_videotoolbox",
        "h264_videotoolbox",
        "libx265",
        "libx264",
    ]

    for encoder in priority:
        if encoder in encoders:
            return encoder, _encoder_args_for(encoder, encoders)

    # No known encoder: return libx264 defaults to guarantee success.
    fallback = "libx264"
    return fallback, _encoder_args_for(fallback, encoders)


def reencode_video_optimal(
    input_video_path: Path,
    *,
    resize: Resolution | None = None,
    delete_on_exit: bool = True,
    save_file: bool = False,
    output_directory: Path | None = None,
    verbose: bool = False,
) -> Path:
    """
    Re-encode an existing video to AV1 (GPU when possible) and optionally
    downsample to 1080p / 720p / 360p, keeping everything else unchanged.
    Falls back to CPU encoders when NVIDIA NVENC is unavailable.

    Args:
        input_video_path: Path to the source video that should be re-encoded.
        resize: Optional predefined resolution (e.g. ``"720p"``) used to downscale the video before logging.
        delete_on_exit: Whether temporary files should be removed when the process exits.
        save_file: If ``True``, write the re-encoded video next to the input file.
        output_directory: Optional output directory when ``save_file`` is ``True``.
        verbose: Emit FFmpeg timing information when ``True``.

    Returns:
        Path to the temporary video file.
    """
    if not input_video_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    # ── output destination (unchanged) ──────────────────────────────────────
    if not save_file:
        with tempfile.NamedTemporaryFile(suffix="_optimal.mp4", delete=False) as tmp:
            output_path = Path(tmp.name)
        if delete_on_exit:
            atexit.register(lambda p: p.unlink(missing_ok=True), output_path)
    else:
        out_dir = output_directory or input_video_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{input_video_path.stem}_optimal.mp4"

    # ── ffmpeg command (base) ───────────────────────────────────────────────
    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video_path),
    ]

    # NEW: in-line scale filter when a preset is chosen
    if resize:
        w, h = RESOLUTION_MAP[resize]
        cmd += ["-vf", f"scale={w}:{h}"]

    # ── select encoder arguments based on platform capabilities ────────────
    encoder_name, video_encoder_args = _select_optimal_video_encoder_args()
    cmd += video_encoder_args
    cmd += [
        "-c:a",
        "copy",  # Copy audio stream without re-encoding
        str(output_path),
    ]

    # ── run ffmpeg & handle errors ─────────────────────────────────────────
    t0 = timer()
    proc = subprocess.run(cmd, capture_output=True)
    dt = timer() - t0
    if verbose:
        print(f"FFmpeg re-encoding using {encoder_name} completed in {dt:.2f} s")

    if proc.returncode:
        if not save_file and output_path.exists():
            output_path.unlink()
        raise RuntimeError(proc.stderr.decode().strip())

    return output_path
