"""The Rerun logging core shared by the live capture (:mod:`metrictrail.capture`) and the finished-run logger
(:mod:`metrictrail.outputs`): times, pinholes, depth and video. Entity paths are in :mod:`metrictrail.entities`.

Video, as the EgoHand app logs it:

* the **input video** is logged once, from the file's own samples (:func:`log_input_video`, ``rr.VideoStream`` through
  ``simplecv.log_video``; no re-encode for an H.264/AV1 MP4), on ``video_time`` = the file's PTS, under the ``input``
  pinhole of the source camera (:class:`SourceRaster` maps ViPE's tracking-raster K and pixels back onto it);
* per-frame images MetricTrail computes (the DINOv3 PCA; the tracked frames of a frame-directory input) are encoded
  to one H.264 ``rr.VideoStream`` each as they are logged (:class:`FrameVideo`; NVENC, libx264 when NVENC is missing);
* depth stays a millimetre PNG ``EncodedDepthImage`` per frame (:func:`log_depth`): Rerun has no depth video yet.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import av
import depth_png
import numpy as np
import rerun as rr
from jaxtyping import Float32, Float64, Int64, UInt8, UInt16
from numpy import ndarray
from simplecv.rerun_log_utils import log_video
from simplecv.video_encoder import VideoCodecChoice, VideoEncoder

from metrictrail import entities
from metrictrail.eval.vipe_output import InputRasterRecord
from metrictrail.palette import BA_WINDOW_RGB

LOGGER: logging.Logger = logging.getLogger(__name__)
VIEWER_CODECS: tuple[str, ...] = ("h264", "av1")
"""Codecs (FFmpeg names) the input video keeps as they are: every browser's Rerun viewer decodes them (headless Linux
Chromium has no HEVC)."""
HARDWARE_ENCODER: str = "h264_nvenc"
"""The encoder :class:`FrameVideo` expects; anything else is the CPU fallback and is warned about once."""
CAMERA_PLANE_M: float = 0.5
"""Image-plane distance of the moving camera's frustum, metres: large enough to see the video playing inside it."""
_warned_cpu_encoder: list[bool] = [False]
"""Whether the CPU-encoder warning was given in this process."""


@dataclass(frozen=True, slots=True)
class SourceRaster:
    """How ViPE put a source frame on the tracking raster: a resize to ``resized_hw``, then a crop at ``crop_left_top``
    (``vipe.metrictrail.raster.plan_raster``; ``resize`` is the crop-free case). Continuous pixel coordinates."""

    source_hw: tuple[int, int]
    """Decoded source frame (height, width)."""
    resized_hw: tuple[int, int]
    """Size the source was resized to (height, width)."""
    crop_left_top: tuple[int, int]
    """Crop offset in the resized frame (left, top)."""

    @staticmethod
    def from_record(record: InputRasterRecord) -> "SourceRaster | None":
        """The run record's ``input_raster`` block; None for a raster without a plan (``unchanged`` frame directories)."""
        if record.source_hw is None or record.resized_hw is None or record.crop_left_top is None:
            return None
        return SourceRaster(
            source_hw=(record.source_hw[0], record.source_hw[1]),
            resized_hw=(record.resized_hw[0], record.resized_hw[1]),
            crop_left_top=(record.crop_left_top[0], record.crop_left_top[1]),
        )

    def scale_xy(self) -> tuple[float, float]:
        """Source pixels per resized pixel, (x, y)."""
        return self.source_hw[1] / self.resized_hw[1], self.source_hw[0] / self.resized_hw[0]

    def source_px(self, raster_px: Float32[ndarray, "n 2"]) -> Float32[ndarray, "n 2"]:
        """Raster pixel coordinates on the source frame."""
        scale_x, scale_y = self.scale_xy()
        offset: Float32[ndarray, "2"] = np.array(self.crop_left_top, dtype=np.float32)
        return ((raster_px + offset) * np.array([scale_x, scale_y], dtype=np.float32)).astype(np.float32)

    def source_intrinsics(self, k4_raster: Float64[ndarray, "4"]) -> Float64[ndarray, "4"]:
        """``fx fy cx cy`` of the source camera from the raster's."""
        scale_x, scale_y = self.scale_xy()
        left, top = self.crop_left_top
        return np.array([k4_raster[0] * scale_x, k4_raster[1] * scale_y, (k4_raster[2] + left) * scale_x, (k4_raster[3] + top) * scale_y])


def set_frame_time(rec: rr.RecordingStream, row: int, source_frame: int | None, video_time_ns: Int64[ndarray, "n"] | None) -> None:
    """Every timeline of tracked row ``row``: ``frame``, and when known ``source_frame`` and ``video_time``
    (``video_time_ns`` indexed by source frame: the input video's PTS or the benchmark's stamps)."""
    rec.set_time(entities.FRAME_TIMELINE, sequence=row)
    if source_frame is None:
        return
    rec.set_time(entities.SOURCE_FRAME_TIMELINE, sequence=source_frame)
    if video_time_ns is not None and source_frame < video_time_ns.shape[0]:
        rec.set_time(entities.VIDEO_TIMELINE, duration=np.timedelta64(int(video_time_ns[source_frame]), "ns"))


def moving_camera(k4: Float64[ndarray, "4"], hw: tuple[int, int]) -> rr.Pinhole:
    """The moving camera's frustum in the 3D views (the input video's camera, or the tracked raster's without one)."""
    return rr.Pinhole(
        focal_length=k4[:2],
        principal_point=k4[2:],
        resolution=[hw[1], hw[0]],
        image_plane_distance=CAMERA_PLANE_M,
        color=BA_WINDOW_RGB,
        line_width=-2.5,  # negative = UI points (rr.Radius.ui_points); a positive value is metres
    )


def log_pinholes(rec: rr.RecordingStream, k4_raster: Float64[ndarray, "4"], raster_hw: tuple[int, int], raster: SourceRaster | None) -> None:
    """``cam``: the tracking raster's pinhole; with ``raster``, also :data:`entities.INPUT`: the source camera the input
    video plays in, the moving camera's frustum in the 3D views."""
    height, width = raster_hw
    rec.log(entities.CAM, rr.Pinhole(focal_length=k4_raster[:2], principal_point=k4_raster[2:], resolution=[width, height]))
    if raster is not None:
        rec.log(entities.INPUT, moving_camera(raster.source_intrinsics(k4_raster), raster.source_hw))


def log_depth(rec: rr.RecordingStream, depth_m: Float32[ndarray, "h w"], depth_range_m: tuple[float, float]) -> None:
    """``cam/depth``: metric depth as a millimetre PNG (0 = no depth; NaN counts as none), turbo over ``depth_range_m``."""
    depth_mm: UInt16[ndarray, "h w"] = np.clip(np.round(np.nan_to_num(depth_m, nan=0.0) * 1000.0), 0, 65535).astype(np.uint16)
    rec.log(
        entities.CAM_DEPTH,
        rr.EncodedDepthImage(
            blob=depth_png.encode(depth_mm), media_type="image/png", meter=1000.0, colormap="turbo", depth_range=[r * 1000.0 for r in depth_range_m]
        ),
    )


class FrameVideo:
    """Per-frame RGB images of one entity as an H.264 ``rr.VideoStream``: each frame is encoded when it is logged and its
    sample lands at the recording's current time (no B-frames, one sample per frame)."""

    def __init__(self, rec: rr.RecordingStream, entity: str, fps: float) -> None:
        self.rec: rr.RecordingStream = rec
        self.entity: str = entity
        self.encoder: VideoEncoder = VideoEncoder(codec=VideoCodecChoice.H264, fps=float(max(round(fps), 1)))
        rec.log(entity, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

    def log(self, rgb: UInt8[ndarray, "h w 3"]) -> None:
        """Encode one frame (cropped to even sides) and log its sample."""
        even: UInt8[ndarray, "H W 3"] = np.ascontiguousarray(rgb[: rgb.shape[0] // 2 * 2, : rgb.shape[1] // 2 * 2])
        packets: list[tuple[int, bytes]] = self.encoder.encode_frame(even)
        if self.encoder.encoder_name != HARDWARE_ENCODER and not _warned_cpu_encoder[0]:
            _warned_cpu_encoder[0] = True
            LOGGER.warning("NVENC is unavailable; %s encodes MetricTrail's derived videos on the CPU", self.encoder.encoder_name)
        for _, sample in packets:
            self.rec.log(self.entity, rr.VideoStream.from_fields(sample=sample))

    def close(self) -> None:
        """Log what the encoder still holds (nothing for the zero-latency settings) at the current time."""
        for _, sample in self.encoder.flush():
            self.rec.log(self.entity, rr.VideoStream.from_fields(sample=sample))


def viewer_mp4(video: Path, out_dir: Path) -> Path:
    """The video as an MP4 every Rerun viewer plays: an MP4/MOV of H.264 or AV1 without a display rotation comes back
    as it is; anything else (HEVC, another container, a rotated phone clip) is re-encoded once into ``out_dir`` as H.264
    without B-frames, with the rotation applied (``Mp4Reader`` ignores it; ViPE's decode applies it)."""
    from torchcodec.decoders import VideoDecoder

    with av.open(str(video)) as container:
        is_mp4: bool = "mp4" in container.format.name.split(",")
        codec: str = container.streams.video[0].codec_context.name
    rotation: float = float(VideoDecoder(str(video), seek_mode="approximate").metadata.rotation or 0.0)
    if is_mp4 and codec in VIEWER_CODECS and round(rotation) % 360 == 0:
        return video
    out_dir.mkdir(parents=True, exist_ok=True)
    target: Path = out_dir / f"{video.stem}.mp4"
    print(f"[metrictrail] re-encoding {video.name} ({codec}, rotation {rotation:g} deg) to H.264 for the viewer")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-map", "0:v:0"]
        + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-bf", "0", "-pix_fmt", "yuv420p", str(target)],
        check=True,
    )
    return target


def log_input_video(rec: rr.RecordingStream, video: Path, scratch: Path) -> Int64[ndarray, "n"]:
    """``input/video``: the input video's samples (:func:`viewer_mp4`) on ``video_time``; returns their PTS, ns, one per
    source frame in presentation order."""
    return log_video(viewer_mp4(video, scratch), Path(entities.INPUT_VIDEO), timeline=entities.VIDEO_TIMELINE, recording=rec)
