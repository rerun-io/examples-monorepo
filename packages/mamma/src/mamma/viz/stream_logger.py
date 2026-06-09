"""Streaming Rerun logger: per-camera H.264 VideoStreams + shared 3D scene.

The hot-loop contract is one ``log_tick_video`` call per synchronized frame
set. Frames arrive as GPU-resident RGB CHW uint8 tensors (already resized);
each camera's frames are NVENC-encoded to H.264 packets which Rerun's viewer
decodes — the RRD stays small (video packets, not images) and the logging
path never re-decodes source video.
"""

from __future__ import annotations

import rerun as rr
import torch
from jaxtyping import UInt8
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.rerun_log_utils import log_pinhole
from simplecv.video_encoder import VideoCodecChoice, VideoEncoder

from mamma.datasets.sequence import MultiViewSequence
from mamma.viz.blueprint import WORLD_TAG, camera_entity, default_blueprint, pinhole_entity

TIMELINE: str = "time"
"""Shared timeline name; ticks are elapsed seconds (``frame_idx / fps``)."""


class StreamLogger:
    """Owns all Rerun logging for one streaming run."""

    def __init__(
        self,
        sequence: MultiViewSequence,
        resize_hw: tuple[int, int],
        video_codec: VideoCodecChoice = VideoCodecChoice.H264,
    ) -> None:
        """Args:
        sequence: The capture being processed (native-resolution calibration).
        resize_hw: ``(height, width)`` of the frames the engine works on;
            pinholes are logged rescaled to this grid so video, 2D overlays,
            and reprojections all share one pixel space.
        video_codec: Codec for the per-camera VideoStreams.
        """
        self.sequence: MultiViewSequence = sequence
        self.resize_hw: tuple[int, int] = resize_hw
        self.fps: float = sequence.fps
        self._video_codec: VideoCodecChoice = video_codec
        self._encoders: dict[str, VideoEncoder] = {
            name: VideoEncoder(codec=video_codec, fps=sequence.fps) for name in sequence.camera_names
        }
        rerun_codec_by_choice: dict[VideoCodecChoice, rr.VideoCodec] = {
            VideoCodecChoice.H264: rr.VideoCodec.H264,
            VideoCodecChoice.H265: rr.VideoCodec.H265,
            VideoCodecChoice.AV1: rr.VideoCodec.AV1,
        }
        self._rerun_codec: rr.VideoCodec = rerun_codec_by_choice[video_codec]

    def setup(self) -> None:
        """Send blueprint and static scene structure (run once, before the loop)."""
        rr.send_blueprint(default_blueprint(self.sequence.camera_names))
        rr.log(WORLD_TAG, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        height: int
        width: int
        height, width = self.resize_hw
        for cam in self.sequence.cameras:
            pinhole: PinholeParameters = cam.scaled_to(height=height, width=width).to_pinhole_parameters()
            log_pinhole(pinhole, cam_log_path=camera_entity(cam.name), image_plane_distance=0.4, static=True)
            rr.log(f"{pinhole_entity(cam.name)}/video", rr.VideoStream(codec=self._rerun_codec), static=True)

    def log_tick_video(self, frame_idx: int, frames: list[UInt8[torch.Tensor, "3 h w"]]) -> None:
        """Encode and log one synchronized frame per camera at this tick."""
        for cam_name, frame_chw in zip(self.sequence.camera_names, frames, strict=True):
            rgb_hwc: UInt8[ndarray, "h w 3"] = frame_chw.permute(1, 2, 0).contiguous().cpu().numpy()
            packets: list[tuple[int, bytes]] = self._encoders[cam_name].encode_frame(rgb_hwc)
            self._log_packets(cam_name, packets)

    def flush(self) -> None:
        """Drain buffered encoder packets (run once, after the loop)."""
        for cam_name, encoder in self._encoders.items():
            self._log_packets(cam_name, encoder.flush())

    def _log_packets(self, cam_name: str, packets: list[tuple[int, bytes]]) -> None:
        entity: str = f"{pinhole_entity(cam_name)}/video"
        for pts, data in packets:
            rr.set_time(TIMELINE, duration=pts / self.fps)
            rr.log(entity, rr.VideoStream.from_fields(sample=data))

    @property
    def encoder_stats(self) -> dict[str, dict[str, object]]:
        """Per-camera encoder performance metrics."""
        return {name: enc.stats for name, enc in self._encoders.items()}


def set_tick_time(frame_idx: int, fps: float) -> None:
    """Position the shared timeline at ``frame_idx`` for non-video entities."""
    rr.set_time(TIMELINE, duration=frame_idx / fps)
