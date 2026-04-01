"""Image stream handler: logs camera frames to Rerun.

When encode_video=True (default), JPEG/RAW images are re-encoded to H265 or AV1
via VideoStream for dramatically smaller RRD files. When encode_video=False,
images are logged as-is (EncodedImage for JPEG, Image for RAW).

Video codec streams (H264/H265 already in VRS) always pass through directly.
"""

import logging

import cv2
import numpy as np
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray

from pyvrs_viewer.video_encoder import VideoCodecChoice, VideoEncoder

logger: logging.Logger = logging.getLogger(__name__)


class FramePlayer:
    """Handles image streams from a VRS file and logs them to Rerun.

    Mirrors the C++ FramePlayer from rerun-io/cpp-example-vrs, with optional
    H265 video encoding for smaller RRD output.
    """

    def __init__(self, stream_id: str, stream_name: str, *, encode_video: bool = True, video_codec: VideoCodecChoice = VideoCodecChoice.H265) -> None:
        self._stream_id: str = stream_id
        self._entity_path: str = stream_name
        self._enabled: bool = True
        self._frame_number: int = 0
        self._codec_logged: bool = False
        self._encode_video: bool = encode_video
        self._video_codec: VideoCodecChoice = video_codec
        self._encoder: VideoEncoder | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def entity_path(self) -> str:
        return self._entity_path

    @property
    def encoder_stats(self) -> dict[str, object] | None:
        """Return encoding statistics, or None if no encoder was used."""
        return self._encoder.stats if self._encoder is not None else None

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
            # Already encoded — pass through directly regardless of encode_video flag
            self._log_video_passthrough(image_spec, image_block)
        elif self._encode_video:
            # Re-encode to H265 for smaller RRD
            self._log_encoded_frame(image_format, image_spec, image_block)
        elif image_format == "jpg":
            self._log_jpeg_frame(image_block)
        elif image_format == "png":
            self._log_png_frame(image_block)
        else:
            self._log_raw_frame(image_spec, image_block)

    def flush(self) -> None:
        """Flush any remaining frames from the video encoder."""
        if self._encoder is None:
            return
        flush_packets: list[bytes] = self._encoder.flush()
        for packet_bytes in flush_packets:
            rr.log(self._entity_path, rr.VideoStream.from_fields(sample=packet_bytes))

    # ── Video encoding path (encode_video=True) ─────────────────────────

    def _log_encoded_frame(
        self,
        image_format: str,
        image_spec: dict[str, object],
        image_block: UInt8[ndarray, "n"],
    ) -> None:
        """Decode image, re-encode to video codec, log as VideoStream."""
        # Decode to numpy
        if image_format in ("jpg", "png"):
            decoded: UInt8[np.ndarray, "h w"] | UInt8[np.ndarray, "h w 3"] | None = cv2.imdecode(image_block, cv2.IMREAD_UNCHANGED)
            if decoded is None:
                logger.warning("Stream %s: failed to decode %s image, skipping frame", self._stream_id, image_format)
                return
            # cv2 returns BGR for color images — convert to RGB
            if decoded.ndim == 3 and decoded.shape[2] == 3:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        else:
            # RAW format — reshape using spec dimensions
            decoded = self._decode_raw_image(image_spec, image_block)
            if decoded is None:
                return

        # Lazily create encoder on first frame
        if self._encoder is None:
            self._encoder = VideoEncoder(codec=self._video_codec)
            # Log codec as static — map our codec choice to Rerun's VideoCodec enum
            rr_codec = rr.VideoCodec.AV1 if self._video_codec == VideoCodecChoice.AV1 else rr.VideoCodec.H265
            rr.log(self._entity_path, rr.VideoStream(codec=rr_codec), static=True)
            self._codec_logged = True

        # Encode and log packets
        packets: list[bytes] = self._encoder.encode_frame(decoded)
        for packet_bytes in packets:
            rr.log(self._entity_path, rr.VideoStream.from_fields(sample=packet_bytes))

    def _decode_raw_image(
        self, image_spec: dict[str, object], pixel_array: UInt8[ndarray, "n"]
    ) -> UInt8[np.ndarray, "h w"] | UInt8[np.ndarray, "h w 3"] | None:
        """Reshape a raw pixel array into an image using spec dimensions."""
        width: int = int(image_spec.get("width", 0))
        height: int = int(image_spec.get("height", 0))
        pixel_format: str = str(image_spec.get("pixel_format", ""))

        if width == 0 or height == 0:
            logger.warning("Stream %s: raw image with unknown dimensions, skipping", self._stream_id)
            return None

        channels: int = _channels_from_pixel_format(pixel_format)
        try:
            if channels == 1:
                return pixel_array.reshape(height, width)
            return pixel_array.reshape(height, width, channels)
        except ValueError:
            logger.warning("Stream %s: failed to reshape raw image, disabling", self._stream_id)
            self._enabled = False
            return None

    # ── Video passthrough (already H264/H265 in VRS) ────────────────────

    def _log_video_passthrough(self, image_spec: dict[str, object], raw_bytes: UInt8[ndarray, "n"]) -> None:
        """Log H264/H265 video codec frame via rr.VideoStream (passthrough)."""
        codec_name: str = str(image_spec.get("codec_name", "h264")).lower()

        if not self._codec_logged:
            codec = rr.VideoCodec.H265 if ("h265" in codec_name or "hevc" in codec_name) else rr.VideoCodec.H264
            rr.log(self._entity_path, rr.VideoStream(codec=codec), static=True)
            self._codec_logged = True
            logger.info("Stream %s: passthrough VideoStream with codec %s", self._stream_id, codec_name)

        frame_bytes: bytes = raw_bytes.tobytes()
        rr.log(self._entity_path, rr.VideoStream.from_fields(sample=frame_bytes))

    # ── Image-only paths (encode_video=False) ───────────────────────────

    def _log_jpeg_frame(self, raw_bytes: UInt8[ndarray, "n"]) -> None:
        """Log raw JPEG bytes via rr.EncodedImage (no decode needed)."""
        rr.log(self._entity_path, rr.EncodedImage(contents=raw_bytes.tobytes(), media_type="image/jpeg"))

    def _log_png_frame(self, raw_bytes: UInt8[ndarray, "n"]) -> None:
        """Log raw PNG bytes via rr.EncodedImage (no decode needed)."""
        rr.log(self._entity_path, rr.EncodedImage(contents=raw_bytes.tobytes(), media_type="image/png"))

    def _log_raw_frame(self, image_spec: dict[str, object], pixel_array: UInt8[ndarray, "n"]) -> None:
        """Log decoded raw pixel array via rr.Image."""
        decoded: UInt8[np.ndarray, "h w"] | UInt8[np.ndarray, "h w 3"] | None = self._decode_raw_image(image_spec, pixel_array)
        if decoded is not None:
            rr.log(self._entity_path, rr.Image(decoded))


def _channels_from_pixel_format(pixel_format: str) -> int:
    """Infer channel count from VRS pixel format string."""
    fmt: str = pixel_format.lower()
    if "grey" in fmt or "depth" in fmt or "scalar" in fmt:
        return 1
    if "rgba" in fmt or "bgra" in fmt:
        return 4
    if "rgb" in fmt or "bgr" in fmt:
        return 3
    return 1
