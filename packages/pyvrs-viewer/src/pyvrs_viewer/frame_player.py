"""Image stream handler: logs camera frames to Rerun.

When encode_video=True (default), JPEG images are decoded directly to YUV planes
via turbojpeg (no RGB intermediate) and encoded to H265/AV1 via VideoStream.
When encode_video=False, raw JPEG bytes are passed to EncodedImage (no decode).

Video codec streams (H264/H265 already in VRS) always pass through directly.
"""

import logging

import numpy as np
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray
from turbojpeg import TurboJPEG

from pyvrs_viewer.video_encoder import VideoCodecChoice, VideoEncoder

logger: logging.Logger = logging.getLogger(__name__)

# Thread-local TurboJPEG instance (safe for concurrent use from thread pools)
_tj: TurboJPEG = TurboJPEG()


class FramePlayer:
    """Handles image streams from a VRS file and logs them to Rerun.

    Mirrors the C++ FramePlayer from rerun-io/cpp-example-vrs, with optional
    video encoding for smaller RRD output.
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
        # Map encoder PTS → (source_timestamp, frame_number) for correct timestamp on delayed packets
        self._pts_to_time: dict[int, tuple[float, int]] = {}

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
        """Log a single image frame to Rerun."""
        if not self._enabled:
            return

        rr.set_time("timestamp", duration=timestamp_sec)
        rr.set_time("frame_number", sequence=self._frame_number)
        self._frame_number += 1

        if metadata:
            meta_str: str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
            rr.log(f"{self._entity_path}/data", rr.TextDocument(meta_str))

        image_format: str = str(image_spec.get("image_format", "raw"))

        if image_format == "video":
            self._log_video_passthrough(image_spec, image_block)
        elif self._encode_video:
            self._log_encoded_frame(timestamp_sec, image_format, image_spec, image_block)
        elif image_format in ("jpg", "png"):
            media_type: str = "image/jpeg" if image_format == "jpg" else "image/png"
            rr.log(self._entity_path, rr.EncodedImage(contents=image_block.tobytes(), media_type=media_type))
        else:
            self._log_raw_frame(image_spec, image_block)

    def flush(self) -> None:
        """Flush any remaining frames from the video encoder."""
        if self._encoder is None:
            return
        for pts, packet_bytes in self._encoder.flush():
            self._log_packet_with_correct_time(pts, packet_bytes)

    # ── Video encoding path (encode_video=True) ─────────────────────────

    def _ensure_encoder(self) -> None:
        """Lazily create encoder and log codec as static on first frame."""
        if self._encoder is None:
            self._encoder = VideoEncoder(codec=self._video_codec)
            rr_codec = rr.VideoCodec.AV1 if self._video_codec == VideoCodecChoice.AV1 else rr.VideoCodec.H265
            rr.log(self._entity_path, rr.VideoStream(codec=rr_codec), static=True)
            self._codec_logged = True

    def _log_packet_with_correct_time(self, pts: int, packet_bytes: bytes) -> None:
        """Log an encoded packet using the source frame's timestamp (not the current frame's).

        Encoders buffer frames — the packet emitted when submitting frame N may
        actually be for frame N-2. Use the PTS to look up the correct source timestamp.
        """
        ts_info: tuple[float, int] | None = self._pts_to_time.get(pts)
        if ts_info is not None:
            rr.set_time("timestamp", duration=ts_info[0])
            rr.set_time("frame_number", sequence=ts_info[1])
        rr.log(self._entity_path, rr.VideoStream.from_fields(sample=packet_bytes))

    def encode_and_log_yuv(
        self,
        timestamp_sec: float,
        yuv_planes: list[UInt8[np.ndarray, "..."]],
        metadata: dict[str, object],
    ) -> None:
        """Encode pre-decoded YUV planes and log to Rerun (used by parallel pipeline)."""
        if not self._enabled:
            return

        # Log metadata with the source frame's timestamp
        rr.set_time("timestamp", duration=timestamp_sec)
        rr.set_time("frame_number", sequence=self._frame_number)

        if metadata:
            meta_str: str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
            rr.log(f"{self._entity_path}/data", rr.TextDocument(meta_str))

        # Record PTS→timestamp mapping BEFORE encoding (encoder assigns PTS = _frame_number)
        self._ensure_encoder()
        assert self._encoder is not None
        encoder_pts: int = self._encoder._frame_number  # Next PTS the encoder will assign
        self._pts_to_time[encoder_pts] = (timestamp_sec, self._frame_number)
        self._frame_number += 1

        if len(yuv_planes) == 1:
            packets: list[tuple[int, bytes]] = self._encoder.encode_yuv_planes(yuv_planes[0])
        elif len(yuv_planes) >= 3:
            packets = self._encoder.encode_yuv_planes(yuv_planes[0], yuv_planes[1], yuv_planes[2])
        else:
            return

        # Log packets with their actual source timestamps (may differ due to encoder buffering)
        for pts, packet_bytes in packets:
            self._log_packet_with_correct_time(pts, packet_bytes)

    def _log_encoded_frame(
        self,
        timestamp_sec: float,
        image_format: str,
        image_spec: dict[str, object],
        image_block: UInt8[ndarray, "n"],
    ) -> None:
        """Decode image to YUV via turbojpeg, encode to video codec, log as VideoStream.

        Called from on_data_record (sequential path). Timestamp/frame_number are
        already set by the caller for metadata logging, but packet logging uses
        PTS-based timestamp lookup due to encoder buffering.
        """
        self._ensure_encoder()
        assert self._encoder is not None

        # Register PTS→timestamp mapping before encoding
        encoder_pts: int = self._encoder._frame_number
        self._pts_to_time[encoder_pts] = (timestamp_sec, self._frame_number - 1)

        if image_format in ("jpg", "png"):
            jpeg_bytes: bytes = image_block.tobytes()
            yuv_planes: list[UInt8[np.ndarray, "..."]] = _tj.decode_to_yuv_planes(jpeg_bytes)
            if len(yuv_planes) == 1:
                packets: list[tuple[int, bytes]] = self._encoder.encode_yuv_planes(yuv_planes[0])
            elif len(yuv_planes) >= 3:
                packets = self._encoder.encode_yuv_planes(yuv_planes[0], yuv_planes[1], yuv_planes[2])
            else:
                return
        else:
            decoded: UInt8[np.ndarray, "h w"] | UInt8[np.ndarray, "h w 3"] | None = self._decode_raw_image(image_spec, image_block)
            if decoded is None:
                return
            packets = self._encoder.encode_frame(decoded)

        for pts, packet_bytes in packets:
            self._log_packet_with_correct_time(pts, packet_bytes)

    def _decode_raw_image(
        self, image_spec: dict[str, object], pixel_array: UInt8[ndarray, "n"]
    ) -> UInt8[np.ndarray, "h w"] | UInt8[np.ndarray, "h w 3"] | None:
        """Reshape a raw pixel array into an image using spec dimensions."""
        width: int = int(image_spec.get("width", 0))
        height: int = int(image_spec.get("height", 0))
        pixel_format: str = str(image_spec.get("pixel_format", ""))

        if width == 0 or height == 0:
            logger.warning(f"Stream {self._stream_id}: raw image with unknown dimensions, skipping")
            return None

        channels: int = _channels_from_pixel_format(pixel_format)
        try:
            if channels == 1:
                return pixel_array.reshape(height, width)
            return pixel_array.reshape(height, width, channels)
        except ValueError:
            logger.warning(f"Stream {self._stream_id}: failed to reshape raw image, disabling")
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
            logger.info(f"Stream {self._stream_id}: passthrough VideoStream with codec {codec_name}")

        rr.log(self._entity_path, rr.VideoStream.from_fields(sample=raw_bytes.tobytes()))

    # ── Image-only path (encode_video=False, RAW format) ────────────────

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
