"""Image stream handler: logs camera frames to Rerun.

Supports three image formats with priority for memory efficiency:
  1. video (H264/H265) → rr.VideoStream (raw codec bytes, no decode)
  2. jpg → rr.EncodedImage (raw JPEG bytes, no decode)
  3. raw → rr.Image (decoded pixel array, fallback)
"""

import logging

import numpy as np
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray

logger: logging.Logger = logging.getLogger(__name__)


class FramePlayer:
    """Handles image streams from a VRS file and logs them to Rerun.

    Mirrors the C++ FramePlayer from rerun-io/cpp-example-vrs.
    """

    def __init__(self, stream_id: str, stream_name: str) -> None:
        self._stream_id: str = stream_id
        self._entity_path: str = stream_name
        self._enabled: bool = True
        self._frame_number: int = 0
        self._codec_logged: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def on_configuration_record(self, metadata: dict[str, object]) -> None:
        """Log static configuration metadata as a TextDocument."""
        if not self._enabled:
            return
        config_str: str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        rr.log(f"{self._entity_path}/configuration", rr.TextDocument(config_str), static=True)

    def on_data_record(
        self,
        timestamp_sec: float,
        image_spec: dict[str, object],
        image_block: UInt8[ndarray, "n"],
        metadata: dict[str, object],
    ) -> None:
        """Log a single image frame to Rerun.

        Args:
            timestamp_sec: Record timestamp in seconds.
            image_spec: Image spec dict with keys like image_format, codec_name, width, height.
            image_block: Raw image data as 1D uint8 numpy array (encoded for jpg/video, decoded for raw).
            metadata: Per-frame metadata dict.
        """
        if not self._enabled:
            return

        rr.set_time("timestamp", duration=timestamp_sec)
        rr.set_time("frame_number", sequence=self._frame_number)
        self._frame_number += 1

        # Log per-frame metadata
        if metadata:
            meta_str: str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
            rr.log(f"{self._entity_path}/data", rr.TextDocument(meta_str))

        image_format: str = str(image_spec.get("image_format", "raw"))

        if image_format == "video":
            self._log_video_frame(image_spec, image_block)
        elif image_format == "jpg":
            self._log_jpeg_frame(image_block)
        elif image_format == "png":
            self._log_png_frame(image_block)
        else:
            self._log_raw_frame(image_spec, image_block)

    def _log_video_frame(self, image_spec: dict[str, object], raw_bytes: UInt8[ndarray, "n"]) -> None:
        """Log H264/H265 video codec frame via rr.VideoStream."""
        codec_name: str = str(image_spec.get("codec_name", "h264")).lower()

        if not self._codec_logged:
            # Log codec as static on first frame
            codec = rr.VideoCodec.H265 if ("h265" in codec_name or "hevc" in codec_name) else rr.VideoCodec.H264
            rr.log(self._entity_path, rr.VideoStream(codec=codec), static=True)
            self._codec_logged = True
            logger.info("Stream %s: using VideoStream with codec %s", self._stream_id, codec_name)

        frame_bytes: bytes = raw_bytes.tobytes()
        rr.log(self._entity_path, rr.VideoStream(sample=frame_bytes))

    def _log_jpeg_frame(self, raw_bytes: UInt8[ndarray, "n"]) -> None:
        """Log raw JPEG bytes via rr.EncodedImage (no decode needed)."""
        jpeg_bytes: bytes = raw_bytes.tobytes()
        rr.log(self._entity_path, rr.EncodedImage(contents=jpeg_bytes, media_type="image/jpeg"))

    def _log_png_frame(self, raw_bytes: UInt8[ndarray, "n"]) -> None:
        """Log raw PNG bytes via rr.EncodedImage (no decode needed)."""
        png_bytes: bytes = raw_bytes.tobytes()
        rr.log(self._entity_path, rr.EncodedImage(contents=png_bytes, media_type="image/png"))

    def _log_raw_frame(self, image_spec: dict[str, object], pixel_array: UInt8[ndarray, "n"]) -> None:
        """Log decoded raw pixel array via rr.Image."""
        width: int = int(image_spec.get("width", 0))
        height: int = int(image_spec.get("height", 0))
        pixel_format: str = str(image_spec.get("pixel_format", ""))

        if width == 0 or height == 0:
            logger.warning("Stream %s: raw image with unknown dimensions, skipping", self._stream_id)
            return

        # Determine channels from pixel format
        channels: int = _channels_from_pixel_format(pixel_format)

        try:
            if channels == 1:
                image: UInt8[np.ndarray, "h w"] = pixel_array.reshape(height, width)
            else:
                image = pixel_array.reshape(height, width, channels)
            rr.log(self._entity_path, rr.Image(image))
        except ValueError:
            logger.warning(
                "Stream %s: failed to reshape raw image (%d bytes) to %dx%dx%d, skipping",
                self._stream_id,
                pixel_array.size,
                height,
                width,
                channels,
            )
            self._enabled = False


def _channels_from_pixel_format(pixel_format: str) -> int:
    """Infer channel count from VRS pixel format string."""
    fmt: str = pixel_format.lower()
    if "grey" in fmt or "depth" in fmt or "scalar" in fmt:
        return 1
    if "rgba" in fmt or "bgra" in fmt:
        return 4
    if "rgb" in fmt or "bgr" in fmt:
        return 3
    # Default to 1 channel for unknown formats
    return 1
