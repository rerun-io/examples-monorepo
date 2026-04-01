"""Video encoder using PyAV with hardware acceleration fallback to CPU.

Supports H265 and AV1 codecs. Encodes numpy image frames to raw codec
packets suitable for rr.VideoStream.
"""

import logging
import time
from enum import Enum
from fractions import Fraction

import av
from jaxtyping import UInt8
from numpy import ndarray

logger: logging.Logger = logging.getLogger(__name__)


class VideoCodecChoice(Enum):
    """Codec choices for video encoding."""

    H265 = "h265"
    """H.265/HEVC — good compression, wide hardware support."""
    AV1 = "av1"
    """AV1 — best compression, newer hardware required for decode."""


# Encoder preference order per codec: hardware (NVENC) → software (CPU)
_ENCODER_CANDIDATES: dict[VideoCodecChoice, list[str]] = {
    VideoCodecChoice.H265: ["hevc_nvenc", "libx265"],
    VideoCodecChoice.AV1: ["av1_nvenc", "libsvtav1"],
}


class VideoEncoder:
    """Per-stream video encoder. Tries hardware (NVENC) first, falls back to CPU.

    Lazily initialized on the first call to encode_frame() when dimensions are known.
    Outputs raw codec packets suitable for rr.VideoStream.
    """

    def __init__(self, codec: VideoCodecChoice = VideoCodecChoice.H265, fps: float = 30.0) -> None:
        self._codec: VideoCodecChoice = codec
        self._fps: float = fps
        self._ctx: av.VideoCodecContext | None = None
        self._encoder_name: str = ""
        self._frame_number: int = 0
        self._total_encode_sec: float = 0.0
        self._total_bytes: int = 0

    @property
    def encoder_name(self) -> str:
        return self._encoder_name

    @property
    def stats(self) -> dict[str, object]:
        """Return encoding statistics."""
        return {
            "encoder": self._encoder_name,
            "codec": self._codec.value,
            "frames": self._frame_number,
            "total_encode_sec": round(self._total_encode_sec, 3),
            "total_bytes": self._total_bytes,
            "fps": round(self._frame_number / self._total_encode_sec, 1) if self._total_encode_sec > 0 else 0,
            "avg_ms_per_frame": round(self._total_encode_sec / self._frame_number * 1000, 2) if self._frame_number > 0 else 0,
        }

    def _init_encoder(self, width: int, height: int) -> None:
        """Try each encoder candidate until one works."""
        # Ensure even dimensions (required by most codecs)
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1

        candidates: list[str] = _ENCODER_CANDIDATES[self._codec]
        for name in candidates:
            try:
                ctx: av.VideoCodecContext = av.CodecContext.create(name, "w")
                ctx.width = width
                ctx.height = height
                ctx.pix_fmt = "yuv420p"
                ctx.time_base = Fraction(1, int(self._fps))
                ctx.max_b_frames = 0  # Required by Rerun VideoStream
                if name == "libx265":
                    ctx.options = {"preset": "fast"}
                elif name == "libsvtav1":
                    ctx.options = {"preset": "8"}  # Fast preset for SVT-AV1 (0=slow, 13=fastest)
                ctx.open()
                self._ctx = ctx
                self._encoder_name = name
                logger.info("VideoEncoder: using %s (%dx%d @ %gfps)", name, width, height, self._fps)
                return
            except Exception:
                logger.debug("VideoEncoder: %s not available, trying next", name)
                continue

        msg: str = f"No {self._codec.value} encoder available. Tried: " + ", ".join(candidates)
        raise RuntimeError(msg)

    def encode_frame(self, image: UInt8[ndarray, "h w"] | UInt8[ndarray, "h w 3"]) -> list[bytes]:
        """Encode a numpy image frame to codec packets.

        Args:
            image: Grayscale (h, w) or RGB (h, w, 3) uint8 numpy array.

        Returns:
            List of encoded packet byte strings.
        """
        if self._ctx is None:
            height: int = image.shape[0]
            width: int = image.shape[1]
            self._init_encoder(width, height)

        # Convert numpy array to av.VideoFrame
        if image.ndim == 2:
            frame: av.VideoFrame = av.VideoFrame.from_ndarray(image, format="gray")
        elif image.shape[2] == 3:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        elif image.shape[2] == 4:
            frame = av.VideoFrame.from_ndarray(image[:, :, :3], format="rgb24")
        else:
            frame = av.VideoFrame.from_ndarray(image[:, :, 0], format="gray")

        # Convert to yuv420p (required by video encoders)
        frame = frame.reformat(format="yuv420p")
        frame.pts = self._frame_number
        self._frame_number += 1

        assert self._ctx is not None
        t0: float = time.perf_counter()
        packets: list[bytes] = [bytes(pkt) for pkt in self._ctx.encode(frame)]
        self._total_encode_sec += time.perf_counter() - t0

        for p in packets:
            self._total_bytes += len(p)
        return packets

    def flush(self) -> list[bytes]:
        """Flush any remaining buffered frames from the encoder.

        Returns:
            List of encoded packet byte strings.
        """
        if self._ctx is None:
            return []
        t0: float = time.perf_counter()
        packets: list[bytes] = [bytes(pkt) for pkt in self._ctx.encode(None)]
        self._total_encode_sec += time.perf_counter() - t0

        for p in packets:
            self._total_bytes += len(p)
        return packets
