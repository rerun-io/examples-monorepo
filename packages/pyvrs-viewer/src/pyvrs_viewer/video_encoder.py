"""H265 video encoder using PyAV with NVENC hardware acceleration fallback to libx265.

Encodes numpy image frames to H265 Annex B packets suitable for rr.VideoStream.
"""

import logging
from fractions import Fraction

import av
from jaxtyping import UInt8
from numpy import ndarray

logger: logging.Logger = logging.getLogger(__name__)

# Encoder preference order: NVENC (hardware) → libx265 (CPU)
_ENCODER_CANDIDATES: list[str] = ["hevc_nvenc", "libx265"]


class VideoEncoder:
    """Per-stream H265 encoder. Tries NVENC first, falls back to libx265.

    Lazily initialized on the first call to encode_frame() when dimensions are known.
    Outputs raw H265 Annex B packets suitable for rr.VideoStream.
    """

    def __init__(self, fps: float = 30.0) -> None:
        self._fps: float = fps
        self._ctx: av.VideoCodecContext | None = None
        self._encoder_name: str = ""
        self._frame_number: int = 0

    @property
    def encoder_name(self) -> str:
        return self._encoder_name

    def _init_encoder(self, width: int, height: int) -> None:
        """Try each encoder candidate until one works."""
        # Ensure even dimensions (required by most codecs)
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1

        for name in _ENCODER_CANDIDATES:
            try:
                ctx: av.VideoCodecContext = av.CodecContext.create(name, "w")
                ctx.width = width
                ctx.height = height
                ctx.pix_fmt = "yuv420p"
                ctx.time_base = Fraction(1, int(self._fps))
                ctx.max_b_frames = 0  # Required by Rerun VideoStream
                if name == "libx265":
                    ctx.options = {"preset": "fast"}
                ctx.open()
                self._ctx = ctx
                self._encoder_name = name
                logger.info("VideoEncoder: using %s (%dx%d @ %gfps)", name, width, height, self._fps)
                return
            except Exception:
                logger.debug("VideoEncoder: %s not available, trying next", name)
                continue

        msg: str = "No H265 encoder available. Tried: " + ", ".join(_ENCODER_CANDIDATES)
        raise RuntimeError(msg)

    def encode_frame(self, image: UInt8[ndarray, "h w"] | UInt8[ndarray, "h w 3"]) -> list[bytes]:
        """Encode a numpy image frame to H265 packets.

        Args:
            image: Grayscale (h, w) or RGB (h, w, 3) uint8 numpy array.

        Returns:
            List of encoded H265 packet byte strings.
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

        # Convert to yuv420p (required by H265 encoders)
        frame = frame.reformat(format="yuv420p")
        frame.pts = self._frame_number
        self._frame_number += 1

        assert self._ctx is not None
        packets: list[bytes] = [bytes(pkt) for pkt in self._ctx.encode(frame)]
        return packets

    def flush(self) -> list[bytes]:
        """Flush any remaining buffered frames from the encoder.

        Returns:
            List of encoded H265 packet byte strings.
        """
        if self._ctx is None:
            return []
        packets: list[bytes] = [bytes(pkt) for pkt in self._ctx.encode(None)]
        return packets
