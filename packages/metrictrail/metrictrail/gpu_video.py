"""GPU-resident video input for ViPE: NVDEC decode and the input raster on the device.

ViPE's ``raw_mp4_stream`` decodes with OpenCV on the host, normalises each frame to float32 there and copies the full
frame to the GPU; its input raster (``vipe.metrictrail.raster.prepare_input_raster``) then copies every frame back to
the host, resizes and crops it there, keeps the prepared clip in host memory and uploads it again on every pass. Its
frame count (``count_stream_frames``) is one more OpenCV decode of the whole file.

Here a video decodes once per pass on NVDEC (simplecv ``TorchCodecVideoReader``): :class:`NvdecVideoStream` yields
ViPE's float frames on the GPU, and :func:`gpu_prepare_input_raster` runs ViPE's own raster plan
(:func:`~vipe.metrictrail.raster.raster_frame`, device-agnostic) on the GPU and keeps the prepared clip there as
:class:`DevicePreparedRasterStream`. Frame directories (every benchmark) keep upstream's path unchanged.

:func:`gpu_video_input` installs the raster hook for one run; :class:`GpuVideoSession` keeps the prepared clip so the
point-cloud colours come from the same device frames as tracking instead of a third host decode.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from jaxtyping import UInt8
from numpy import ndarray
from serde import from_dict
from simplecv.video_io import TorchCodecVideoReader
from torch import Tensor
from vipe.metrictrail.raster import PreparedRasterStream
from vipe.streams.base import ProcessedVideoStream, StreamList, VideoFrame, VideoStream

from metrictrail.eval.vipe_output import InputRasterRecord
from metrictrail.rerun_log import SourceRaster


def _rotate_display(frame_3hw: UInt8[Tensor, "3 h w"], rotation_deg: int) -> UInt8[Tensor, "3 h w"]:
    """Apply a container's clockwise display rotation, as OpenCV's ``CAP_PROP_ORIENTATION_AUTO`` does."""
    turns: int = (rotation_deg // 90) % 4
    return frame_3hw if turns == 0 else torch.rot90(frame_3hw, k=-turns, dims=(1, 2))


class NvdecVideoStream(VideoStream):
    """ViPE's ``RawMp4Stream`` contract on NVDEC: float32 (H,W,3) RGB in [0, 1] on the GPU, ``raw_frame_idx`` the
    decoded frame's index.

    The file sits under ``video_path``, not ``path``: ViPE's frame count treats a stream with an ``.mp4`` ``path`` as
    OpenCV's and grabs the whole file on the host; without it, it counts one pass over this stream on the GPU.
    """

    def __init__(self, video_path: Path, seek_range: range, name: str | None = None) -> None:
        super().__init__()
        from torchcodec.decoders import VideoDecoder

        self.video_path: Path = video_path
        self._name: str = name if name is not None else video_path.stem
        metadata: Any = VideoDecoder(str(video_path), seek_mode="approximate").metadata
        self.rotation_deg: int = int(round(float(getattr(metadata, "rotation", None) or 0.0)))
        height, width = int(metadata.height), int(metadata.width)
        self._hw: tuple[int, int] = (width, height) if (self.rotation_deg // 90) % 2 else (height, width)
        frame_count: int = int(metadata.num_frames or 0)
        self.start: int = seek_range.start
        self.end: int = frame_count if seek_range.stop == -1 else min(seek_range.stop, frame_count)
        self.step: int = seek_range.step
        self._fps: float = float(metadata.average_fps or 0.0) / self.step

    def frame_size(self) -> tuple[int, int]:
        return self._hw

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(range(self.start, self.end, self.step))

    def uint8_frames(self) -> Iterator[tuple[int, UInt8[Tensor, "h w 3"]]]:
        """``(raw_frame_idx, (H,W,3) uint8 RGB on the GPU)`` for the stream's frames, in order."""
        # thread_owned: CUDA decoders are thread-affine and ViPE may advance a stream from a producer thread.
        reader: TorchCodecVideoReader = TorchCodecVideoReader(self.video_path, device="cuda", thread_owned=True)
        try:
            index: int = -1
            while (frame_3hw := reader.read()) is not None:
                index += 1
                if index >= self.end:
                    return
                if index < self.start or (index - self.start) % self.step:
                    continue
                yield index, _rotate_display(frame_3hw, self.rotation_deg).permute(1, 2, 0)
        finally:
            reader.close()

    def __iter__(self) -> Iterator[VideoFrame]:
        for index, rgb_hw3 in self.uint8_frames():
            yield VideoFrame(raw_frame_idx=index, rgb=rgb_hw3.float() / 255.0)


class NvdecStreamList(StreamList):
    """``RawMP4StreamList`` on NVDEC: one video file and ViPE's ``frame_start`` / ``frame_end`` / ``frame_skip``."""

    def __init__(self, video_path: Path, frame_range: range) -> None:
        super().__init__()
        self.video_path: Path = video_path
        self.frame_range: range = frame_range

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> VideoStream:
        if index != 0:
            raise IndexError(index)
        return NvdecVideoStream(self.video_path, self.frame_range)

    def stream_name(self, index: int) -> str:
        return self.video_path.stem


def make_stream_list(streams_config: Any) -> StreamList:
    """NVDEC streams for a single video file under ViPE's ``raw_mp4_stream`` config, else ViPE's own list."""
    path: Path = Path(str(streams_config.base_path))
    if streams_config.instance.endswith("RawMP4StreamList") and path.is_file() and not streams_config.cached:
        return NvdecStreamList(path, range(streams_config.frame_start, streams_config.frame_end, streams_config.frame_skip))
    return StreamList.make(streams_config)


class DevicePreparedRasterStream(PreparedRasterStream):
    """ViPE's prepared raster stream with the (N,h,w,3) uint8 clip already on the GPU (no pinned host copy)."""

    def __init__(self, frames: UInt8[Tensor, "n h w 3"], source_frame_ids: list[int], *, name: str, fps: float) -> None:
        VideoStream.__init__(self)
        if len(source_frame_ids) != len(frames):
            raise ValueError("one source frame id per prepared frame")
        self.device = frames.device
        self._frames = frames
        self.source_frame_ids = [int(item) for item in source_frame_ids]
        self._name = name
        self._fps = float(fps)


@dataclass
class GpuVideoSession:
    """What the GPU input of one run produced, for the stages after tracking."""

    prepared: DevicePreparedRasterStream | None = None
    """The prepared clip on the GPU, or None when the run's input took upstream's path."""
    rasters: int = 0
    """Input rasters prepared on the GPU (one per pipeline run)."""
    raster: SourceRaster | None = None
    """How the latest input raster (either path) put a source frame on the tracking raster; None before one is ready or
    without a resize plan."""

    def source_raster(self) -> SourceRaster | None:
        """:attr:`raster` (the capture's :attr:`~metrictrail.capture.CaptureSession.source_raster` callback)."""
        return self.raster

    def rgb_source(self, source_frame_ids: ndarray) -> Callable[[int], UInt8[ndarray, "h w 3"]] | None:
        """Artifact frame ``i`` -> its prepared 8-bit RGB (tracking's own bytes), copied to the host once."""
        if self.prepared is None:
            return None
        position: dict[int, int] = {source: index for index, source in enumerate(self.prepared.source_frame_ids)}
        if any(int(source) not in position for source in source_frame_ids):
            return None
        host: UInt8[ndarray, "n h w 3"] = self.prepared._frames.cpu().numpy()
        rows: list[int] = [position[int(source)] for source in source_frame_ids]
        return lambda index: host[rows[index]]


def gpu_prepare_input_raster(
    stream: NvdecVideoStream, config: Any, session: GpuVideoSession, *, frame_count: int
) -> tuple[VideoStream, dict[str, Any]]:
    """``prepare_input_raster`` for a resize policy on the GPU: same plan, same kept frames, same record keys.

    ``frame_count`` is the stream's length as ViPE counted it (:func:`~vipe.metrictrail.streams.full_sequence_stream`
    wraps a stream that declares more frames than it yields); only that many frames are read.
    """
    from vipe.metrictrail import raster as vipe_raster
    from vipe.metrictrail.streams import video_orientation

    policy: str = str(config.policy)
    source_hw: tuple[int, int] = (int(stream.frame_size()[0]), int(stream.frame_size()[1]))
    plan = vipe_raster.plan_raster(
        source_hw,
        frame_count,
        float(stream.fps()),
        target_hw=(int(config.height), int(config.width)),
        max_frames=int(config.max_frames),
        policy=policy,
        resampler=str(getattr(config, "resampler", vipe_raster.LEGACY_RESAMPLER)),
    )
    keep: list[int] = vipe_raster.fps_filter_source_indices(frame_count, plan.frame_stride)
    wanted: set[int] = set(keep)
    frames: UInt8[Tensor, "n h w 3"] = torch.empty((len(keep), *plan.target_hw, 3), dtype=torch.uint8, device="cuda")
    source_ids: list[int] = []
    for ordinal, (raw_index, rgb_hw3) in enumerate(islice(stream.uint8_frames(), frame_count)):
        if ordinal in wanted:
            frames[len(source_ids)] = vipe_raster.raster_frame(rgb_hw3.float() / 255.0, plan)
            source_ids.append(raw_index)
    if len(source_ids) != len(keep):
        raise ValueError(f"stream {stream.name()!r} yielded fewer frames than its length {frame_count}")
    if len(source_ids) < vipe_raster.MIN_PREPARED_FRAMES:
        raise ValueError(
            f"only {len(source_ids)} frames prepared from {stream.name()!r}; the pipeline needs at least {vipe_raster.MIN_PREPARED_FRAMES}"
        )
    prepared: DevicePreparedRasterStream = DevicePreparedRasterStream(frames, source_ids, name=stream.name(), fps=plan.prepared_fps)
    session.prepared = prepared
    session.rasters += 1
    record: dict[str, Any] = {
        "schema": vipe_raster.RASTER_SCHEMA,
        "reference": vipe_raster.POLICY_REFERENCE[policy],
        **plan.record(),
        "prepared_hw": list(plan.target_hw),
        "frames": len(prepared),
        "stream_frames": frame_count,
        "source_frame_count_from": "stream length",
        "source_frame_ids_rule": "constant_rate_fps_filter_rule",
        "source_frame_ids_head": source_ids[:4],
        "dropped_attributes": sorted(item.value for item in stream.attributes()),
        "interpolation": {**vipe_raster.INTERPOLATIONS[plan.resampler], "torch": str(torch.__version__)},
        "video_orientation": video_orientation(stream),
        # The prepared clip lives on the GPU (metrictrail.gpu_video), decoded on NVDEC.
        "host_bytes": 0,
        "device_bytes": int(frames.numel()),
        "decoder": f"nvdec (torchcodec), display rotation {stream.rotation_deg} deg applied",
    }
    return prepared, record


@contextmanager
def gpu_video_input(session: GpuVideoSession) -> Iterator[GpuVideoSession]:
    """Route ViPE's input raster through :func:`gpu_prepare_input_raster` for NVDEC streams during the block, and note
    every prepared raster's plan in ``session``."""
    import vipe.metrictrail.pipeline as vipe_pipeline

    upstream: Callable[..., Any] = vipe_pipeline.prepare_input_raster

    def prepare_input_raster(stream: VideoStream, config: Any, *, device: torch.device | None = None) -> tuple[VideoStream, dict[str, Any]]:
        base: VideoStream = stream
        while isinstance(base, ProcessedVideoStream):  # ViPE's ExactLengthStream around a short stream
            base = base.stream
        prepared: tuple[VideoStream, dict[str, Any]]
        if isinstance(base, NvdecVideoStream) and str(config.policy) in ("cover_crop", "resize"):
            prepared = gpu_prepare_input_raster(base, config, session, frame_count=len(stream))
        else:
            prepared = upstream(stream, config, device=device)
        session.raster = SourceRaster.from_record(from_dict(InputRasterRecord, prepared[1]))
        return prepared

    vipe_pipeline.prepare_input_raster = prepare_input_raster
    try:
        yield session
    finally:
        vipe_pipeline.prepare_input_raster = upstream
