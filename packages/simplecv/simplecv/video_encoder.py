"""Hardware-accelerated video encoding via PyAV.

Ported from ``rerun-io/examples-monorepo/packages/pyvrs-viewer`` and extended
with :class:`MP4Writer` for container output.

Two classes:

* :class:`VideoEncoder` — outputs raw codec packets (for Rerun ``VideoStream``).
* :class:`MP4Writer` — wraps a PyAV MP4 container for file output.

Both auto-select NVENC hardware encoding when available, falling back to CPU.
"""

from __future__ import annotations

import time
from enum import Enum
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from jaxtyping import UInt8
from numpy import ndarray

# ──────────────────────── Codec selection ─────────────────────────────────── #


class VideoCodecChoice(Enum):
    """Supported video codec families."""

    H265 = "h265"
    AV1 = "av1"


_ENCODER_CANDIDATES: dict[VideoCodecChoice, list[str]] = {
    VideoCodecChoice.H265: ["hevc_nvenc", "libx265"],
    VideoCodecChoice.AV1: ["av1_nvenc", "libsvtav1"],
}

_GOP_SIZE: int = 30
"""Keyframe interval — 1 second at 30 fps."""

_CRF: int = 30
"""Default constant-rate-factor / quality target."""


def _encoder_options(name: str) -> dict[str, str]:
    """Return codec-specific encoder options for *name*."""
    if name == "libsvtav1":
        return {"preset": "8", "crf": str(_CRF)}
    if name == "libx265":
        return {"preset": "fast", "x265-params": f"crf={_CRF}"}
    # NVENC hardware encoders: defaults give best size/quality tradeoff.
    return {}


# ─────────────── ffmpeg subprocess transcode ──────────────────────────────── #
# Used when the source codec (e.g. monochrome H.265 Rext) cannot be handled
# by PyAV's NVENC path or needs pixel format conversion. ~10x faster than
# the equivalent PyAV decode→reformat→encode loop because ffmpeg runs the
# full pipeline in C without Python frame iteration overhead.


def pick_ffmpeg_encoder() -> str:
    """Return the best available ffmpeg AV1/H.265 encoder.

    Prefers NVENC GPU (``av1_nvenc`` > ``hevc_nvenc``) then CPU fallback
    (``libsvtav1``).

    Raises ``RuntimeError`` if ffmpeg is not installed.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg to preprocess Aria Gen2 VRS files.")

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode:
        raise RuntimeError(f"ffmpeg -encoders failed: {result.stderr[:200]}")

    encoders: str = result.stdout

    for candidate in ["av1_nvenc", "hevc_nvenc"]:
        if candidate in encoders:
            return candidate
    return "libsvtav1"


def ffmpeg_transcode(
    input_path: Path,
    output_path: Path,
    input_format: str = "hevc",
    fps: int = 30,
    encoder: str | None = None,
    cq: int | None = None,
) -> None:
    """Transcode a raw video bitstream to yuv420p MP4 via ffmpeg subprocess.

    Uses NVENC GPU encoding by default with constant-quality (CQ) mode.
    GOP=30, no B-frames.

    Args:
        input_path: Raw bitstream file (e.g. Annex-B ``.h265``).
        output_path: Destination ``.mp4`` path.
        input_format: ffmpeg input format (``hevc``, ``h264``, etc.).
        fps: Output frame rate.
        encoder: Explicit encoder name, or ``None`` to auto-detect via
            :func:`pick_ffmpeg_encoder`.
        cq: NVENC constant-quality value (0–51, higher = smaller file).
            ``None`` uses the NVENC default (~28). For CPU fallback the
            module-level ``_CRF`` is used instead.
    """
    import subprocess

    if encoder is None:
        encoder = pick_ffmpeg_encoder()

    encoder_args: list[str] = [
        "-c:v", encoder, "-pix_fmt", "yuv420p",
        "-g", str(_GOP_SIZE), "-bf", "0",
    ]
    if "nvenc" in encoder:
        if cq is not None:
            encoder_args += ["-cq", str(cq)]
    elif encoder == "libsvtav1":
        encoder_args += ["-crf", str(_CRF), "-preset", "8"]

    cmd: list[str] = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        # -r before -i sets the input framerate so ffmpeg doesn't assume
        # a default (25fps) and resample/duplicate frames.
        "-r", str(fps),
        "-f", input_format, "-i", str(input_path),
        *encoder_args,
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr[-300:]}")


# ──────────────────── VideoEncoder (raw packets) ──────────────────────────── #


class VideoEncoder:
    """Encode frames to raw H.265/AV1 codec packets (for Rerun VideoStream).

    Ported from ``pyvrs-viewer/video_encoder.py``.  Lazy-initialises the PyAV
    ``CodecContext`` on the first frame so the caller doesn't need to know
    dimensions up front.

    Usage::

        encoder = VideoEncoder(codec=VideoCodecChoice.AV1, fps=30.0)
        for y, u, v in yuv_frames:
            packets = encoder.encode_yuv_planes(y, u, v)
            for pts, data in packets:
                rr.log(path, rr.VideoStream.from_fields(sample=data))
        for pts, data in encoder.flush():
            rr.log(path, rr.VideoStream.from_fields(sample=data))
    """

    def __init__(self, codec: VideoCodecChoice = VideoCodecChoice.AV1, fps: float = 30.0) -> None:
        self._codec: VideoCodecChoice = codec
        self._fps: float = fps
        self._ctx: av.VideoCodecContext | None = None
        self._encoder_name: str = ""
        self._frame_number: int = 0
        self._neutral_uv: UInt8[ndarray, "h2 w2"] | None = None
        self._total_encode_sec: float = 0.0
        self._total_bytes: int = 0

    @property
    def encoder_name(self) -> str:
        """Selected encoder name (e.g. ``'av1_nvenc'``)."""
        return self._encoder_name

    @property
    def next_pts(self) -> int:
        """PTS that will be assigned to the *next* submitted frame."""
        return self._frame_number

    @property
    def stats(self) -> dict[str, object]:
        """Encoding performance metrics."""
        fps: float = self._frame_number / max(self._total_encode_sec, 1e-9)
        avg_ms: float = (self._total_encode_sec / max(self._frame_number, 1)) * 1000
        return {
            "encoder": self._encoder_name,
            "codec": self._codec.value,
            "frames": self._frame_number,
            "total_encode_sec": round(self._total_encode_sec, 3),
            "total_bytes": self._total_bytes,
            "fps": round(fps, 1),
            "avg_ms_per_frame": round(avg_ms, 2),
        }

    # ── Lazy init ─────────────────────────────────────────────────────────

    def _init_encoder(self, width: int, height: int) -> None:
        # Most codecs require even dimensions.
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1

        candidates: list[str] = _ENCODER_CANDIDATES[self._codec]
        errors: list[str] = []

        for name in candidates:
            try:
                ctx: av.VideoCodecContext = av.CodecContext.create(name, "w")
                ctx.width = width
                ctx.height = height
                ctx.pix_fmt = "yuv420p"
                ctx.time_base = Fraction(1, int(self._fps))
                ctx.max_b_frames = 0
                ctx.gop_size = _GOP_SIZE
                ctx.options = _encoder_options(name)
                ctx.open()

                self._ctx = ctx
                self._encoder_name = name
                return
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        msg: str = "No working encoder found. Tried:\n" + "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(msg)

    # ── Encoding methods ──────────────────────────────────────────────────

    def encode_yuv_planes(
        self,
        y_plane: UInt8[ndarray, "h w"],
        u_plane: UInt8[ndarray, "h2 w2"] | None = None,
        v_plane: UInt8[ndarray, "h2 w2"] | None = None,
    ) -> list[tuple[int, bytes]]:
        """Encode YUV420 planes directly — fastest path (no color conversion)."""
        height: int = y_plane.shape[0]
        width: int = y_plane.shape[1]

        if self._ctx is None:
            self._init_encoder(width, height)

        frame: av.VideoFrame = av.VideoFrame(width=width, height=height, format="yuv420p")
        frame.planes[0].update(y_plane)
        if u_plane is not None and v_plane is not None:
            frame.planes[1].update(u_plane)
            frame.planes[2].update(v_plane)
        else:
            if self._neutral_uv is None or self._neutral_uv.shape != (height // 2, width // 2):
                self._neutral_uv = np.full((height // 2, width // 2), 128, dtype=np.uint8)
            frame.planes[1].update(self._neutral_uv)
            frame.planes[2].update(self._neutral_uv)

        frame.pts = self._frame_number
        self._frame_number += 1

        assert self._ctx is not None
        t0: float = time.perf_counter()
        packets: list[tuple[int, bytes]] = [(pkt.pts, bytes(pkt)) for pkt in self._ctx.encode(frame)]
        self._total_encode_sec += time.perf_counter() - t0

        for _pts, p in packets:
            self._total_bytes += len(p)
        return packets

    def encode_frame(
        self,
        rgb_or_gray: UInt8[ndarray, "h w"] | UInt8[ndarray, "h w 3"],
    ) -> list[tuple[int, bytes]]:
        """Encode an RGB or grayscale frame (with internal YUV conversion).

        Do NOT pass BGR images — use ``encode_yuv_planes()`` with pre-decoded
        YUV data instead for correct colors and better performance.
        """
        if self._ctx is None:
            height: int = rgb_or_gray.shape[0]
            width: int = rgb_or_gray.shape[1]
            self._init_encoder(width, height)

        channels: int = 1 if rgb_or_gray.ndim == 2 else rgb_or_gray.shape[2]
        if channels == 1:
            frame: av.VideoFrame = av.VideoFrame.from_ndarray(
                rgb_or_gray if rgb_or_gray.ndim == 2 else rgb_or_gray[:, :, 0], format="gray"
            )
        elif channels == 3:
            frame = av.VideoFrame.from_ndarray(rgb_or_gray, format="rgb24")
        elif channels == 4:
            frame = av.VideoFrame.from_ndarray(rgb_or_gray[:, :, :3], format="rgb24")
        else:
            frame = av.VideoFrame.from_ndarray(rgb_or_gray[:, :, 0], format="gray")

        frame = frame.reformat(format="yuv420p")
        frame.pts = self._frame_number
        self._frame_number += 1

        assert self._ctx is not None
        t0 = time.perf_counter()
        packets: list[tuple[int, bytes]] = [(pkt.pts, bytes(pkt)) for pkt in self._ctx.encode(frame)]
        self._total_encode_sec += time.perf_counter() - t0

        for _pts, p in packets:
            self._total_bytes += len(p)
        return packets

    def flush(self) -> list[tuple[int, bytes]]:
        """Flush buffered frames from the encoder."""
        if self._ctx is None:
            return []
        t0: float = time.perf_counter()
        packets: list[tuple[int, bytes]] = [(pkt.pts, bytes(pkt)) for pkt in self._ctx.encode(None)]
        self._total_encode_sec += time.perf_counter() - t0
        for _pts, p in packets:
            self._total_bytes += len(p)
        return packets


# ──────────────────── MP4Writer (container output) ────────────────────────── #


class MP4Writer:
    """Write encoded video frames to an MP4 container file.

    Shares codec selection with :class:`VideoEncoder` but uses a PyAV
    container (``av.open(path, "w")``) instead of a raw ``CodecContext``.

    Usage::

        writer = MP4Writer(Path("out.mp4"), codec=VideoCodecChoice.AV1, fps=30.0)
        for y, u, v in yuv_frames:
            writer.write_yuv_planes(y, u, v)
        writer.close()
    """

    def __init__(self, output_path: Path, codec: VideoCodecChoice = VideoCodecChoice.AV1, fps: float = 30.0) -> None:
        self._output_path: Path = output_path
        self._codec: VideoCodecChoice = codec
        self._fps: float = fps
        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._encoder_name: str = ""
        self._frame_number: int = 0
        self._neutral_uv: UInt8[ndarray, "h2 w2"] | None = None
        self._total_encode_sec: float = 0.0
        self._total_bytes: int = 0

    @property
    def encoder_name(self) -> str:
        """Selected encoder name."""
        return self._encoder_name

    @property
    def stats(self) -> dict[str, object]:
        """Encoding performance metrics."""
        fps: float = self._frame_number / max(self._total_encode_sec, 1e-9)
        avg_ms: float = (self._total_encode_sec / max(self._frame_number, 1)) * 1000
        return {
            "encoder": self._encoder_name,
            "codec": self._codec.value,
            "frames": self._frame_number,
            "total_encode_sec": round(self._total_encode_sec, 3),
            "total_bytes": self._total_bytes,
            "fps": round(fps, 1),
            "avg_ms_per_frame": round(avg_ms, 2),
        }

    def _init_container(self, width: int, height: int) -> None:
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        candidates: list[str] = _ENCODER_CANDIDATES[self._codec]
        errors: list[str] = []

        for name in candidates:
            container: av.container.OutputContainer | None = None
            try:
                container = av.open(str(self._output_path), mode="w")
                stream: av.video.stream.VideoStream = container.add_stream(name, rate=round(self._fps))
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                stream.gop_size = _GOP_SIZE
                stream.max_b_frames = 0
                # NVENC: container.add_stream() defaults to unlimited bitrate
                # unlike CodecContext.create() which defaults to 2 Mbps.
                # Match the CodecContext default for comparable file sizes.
                if "nvenc" in name:
                    stream.bit_rate = 2_000_000
                stream.options = _encoder_options(name)

                self._container = container
                self._stream = stream
                self._encoder_name = name
                return
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                # Close the partially opened container on failure
                if container is not None:
                    import contextlib

                    with contextlib.suppress(Exception):
                        container.close()

        msg = "No working encoder found for MP4. Tried:\n" + "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(msg)

    def write_yuv_planes(
        self,
        y_plane: UInt8[ndarray, "h w"],
        u_plane: UInt8[ndarray, "h2 w2"] | None = None,
        v_plane: UInt8[ndarray, "h2 w2"] | None = None,
    ) -> None:
        """Encode YUV420 planes and write to MP4 — fastest path."""
        height: int = y_plane.shape[0]
        width: int = y_plane.shape[1]

        if self._container is None:
            self._init_container(width, height)

        frame: av.VideoFrame = av.VideoFrame(width=width, height=height, format="yuv420p")
        frame.planes[0].update(y_plane)
        if u_plane is not None and v_plane is not None:
            frame.planes[1].update(u_plane)
            frame.planes[2].update(v_plane)
        else:
            if self._neutral_uv is None or self._neutral_uv.shape != (height // 2, width // 2):
                self._neutral_uv = np.full((height // 2, width // 2), 128, dtype=np.uint8)
            frame.planes[1].update(self._neutral_uv)
            frame.planes[2].update(self._neutral_uv)

        frame.pts = self._frame_number
        self._frame_number += 1

        assert self._stream is not None and self._container is not None
        t0: float = time.perf_counter()
        for packet in self._stream.encode(frame):
            self._total_bytes += packet.size
            self._container.mux(packet)
        self._total_encode_sec += time.perf_counter() - t0

    def write_frame(
        self,
        rgb_or_gray: UInt8[ndarray, "h w"] | UInt8[ndarray, "h w 3"],
    ) -> None:
        """Encode an RGB or grayscale frame and write to MP4.

        Do NOT pass BGR images — use ``write_yuv_planes()`` instead.
        """
        if self._container is None:
            self._init_container(rgb_or_gray.shape[1], rgb_or_gray.shape[0])

        channels: int = 1 if rgb_or_gray.ndim == 2 else rgb_or_gray.shape[2]
        if channels == 1:
            frame: av.VideoFrame = av.VideoFrame.from_ndarray(
                rgb_or_gray if rgb_or_gray.ndim == 2 else rgb_or_gray[:, :, 0], format="gray"
            )
        elif channels == 3:
            frame = av.VideoFrame.from_ndarray(rgb_or_gray, format="rgb24")
        else:
            frame = av.VideoFrame.from_ndarray(rgb_or_gray[:, :, :3], format="rgb24")

        frame = frame.reformat(format="yuv420p")
        frame.pts = self._frame_number
        self._frame_number += 1

        assert self._stream is not None and self._container is not None
        t0: float = time.perf_counter()
        for packet in self._stream.encode(frame):
            self._total_bytes += packet.size
            self._container.mux(packet)
        self._total_encode_sec += time.perf_counter() - t0

    def close(self) -> None:
        """Flush encoder and close the MP4 container."""
        if self._stream is not None and self._container is not None:
            t0: float = time.perf_counter()
            for packet in self._stream.encode():
                self._total_bytes += packet.size
                self._container.mux(packet)
            self._total_encode_sec += time.perf_counter() - t0
            self._container.close()
            self._container = None
            self._stream = None
