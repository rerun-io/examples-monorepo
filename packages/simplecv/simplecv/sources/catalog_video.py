"""Camera videos as dagkit Sources: one time-ordered stream of decoded framesets after the static rig (SPEC 2.6).

``VideoSource`` samples RGB framesets from any ``FrameProvider`` (``VideoSource.catalog``: a catalog segment's);
``CatalogLumaSource`` groups every frame's raw luma by time."""

import itertools
import threading
from collections.abc import Iterator, Mapping, Sequence
from io import BytesIO
from typing import Any, Literal, Protocol, runtime_checkable

import av
import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from dagkit.catalog import read_column
from dagkit.node import Event, Fenced, PortType, Source
from dagkit.sources import Column, IterSource, SegmentRef
from dagkit.time import STATIC
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.gpu import fence_after
from simplecv.ops.framesets import FramesetIndex, group_framesets
from simplecv.types.frames import LumaFrameset, MultiViewFrames
from simplecv.types.rig import RigCameras

MP4_NOMINAL_FPS: int = 30
"""Frame rate written into the MP4 wrapper around the catalog packets: frames are addressed by sample index and the
times come from the catalog, so it only labels the container."""
KEYFRAME_LEAD_NS: int = 5_000_000_000  # a windowed cpu_luma read starts this early, so each camera's decoder begins at a keyframe
LUMA_DECODE_THREADS: int = 2  # libavcodec threads per camera for cpu_luma (every camera decodes in lockstep)
DISTORTION: str = "pinhole:simplecv.components.DistortionCoefficients"
"""A camera's distortion coefficients beside its pinhole (Brown-Conrady: OpenCV order, zero-padded to 14; dataforge's layout)."""
DISTORTION_MODEL: str = "pinhole:simplecv.components.DistortionModel"
"""Names the model of those coefficients; ``undistort`` takes ``brown_conrady`` only (a Kannala-Brandt vector would be misread)."""


class SegmentFrames:
    """Random access to one segment's camera videos (``/world/<camera>/pinhole/video``); packets fetched once per camera.

    A torchcodec NVDEC decoder only decodes on the thread that created it (the runner reads a Source's first frameset on
    the caller's thread, the rest on the Source's own), so decoders are kept per thread over the cached packets. Also a
    graph resource (exo-calib's metric rescale reads a few frames per camera); ``close`` drops everything.
    """

    def __init__(self, segment: SegmentRef, cameras: tuple[str, ...], device: Literal["cuda", "cpu"] = "cuda") -> None:
        self.segment, self.cameras, self.device = segment, cameras, device
        self._packets: list[tuple[Int64[ndarray, " n"], bytes]] = []
        self._decoders: dict[tuple[int, int], Any] = {}  # torchcodec VideoDecoder per (thread, view), imported lazily

    def times_ns(self, view: int) -> Int64[ndarray, " n"]:
        """Sample times of one camera on the segment's timeline (the first call fetches every camera's packets in one read)."""
        if not self._packets:
            dataset = self.segment.entry()  # the entry must outlive its queries
            entities: list[str] = [f"/world/{camera}/pinhole/video" for camera in self.cameras]
            for video in read_catalog_videos(dataset, self.segment.segment, entities, self.segment.timeline):
                mp4: bytes = wrap_mp4(video.samples, video.keyframes, MP4_NOMINAL_FPS, codec=video.codec)
                self._packets.append((np.asarray(video.times).astype("timedelta64[ns]").astype(np.int64), mp4))
        return self._packets[view][0]

    def frames_at(self, view: int, indices: Sequence[int]) -> UInt8[Tensor, "k 3 h w"]:
        from torchcodec.decoders import VideoDecoder  # RGB decodes only: cpu_luma users (robocap-live) ship without torchcodec

        key: tuple[int, int] = (threading.get_ident(), view)
        if key not in self._decoders:
            self.times_ns(view)
            self._decoders[key] = VideoDecoder(self._packets[view][1], device=self.device, seek_mode="exact", num_ffmpeg_threads=0)
        return self._decoders[key].get_frames_at([int(i) for i in indices]).data

    def close(self) -> None:
        self._decoders.clear()
        self._packets.clear()


def log_segment_videos(recording: rr.RecordingStream, segment: SegmentRef, entities: Mapping[str, str]) -> None:
    """Copy cameras' encoded videos from the catalog into ``recording`` (camera -> entity) on the segment's timeline as a
    duration, without re-encoding: what a viewer of the segment plays. A window starts at the keyframe before it (up to
    ``KEYFRAME_LEAD_NS`` earlier), so its first frames decode."""
    window: tuple[int, int] | None = None if segment.window_ns is None else (segment.window_ns[0] - KEYFRAME_LEAD_NS, segment.window_ns[1])
    videos = read_catalog_videos(segment.entry(), segment.segment, [f"/world/{camera}/pinhole/video" for camera in entities], segment.timeline, window)
    for entity, video in zip(entities.values(), videos, strict=True):
        t_ns: Int64[ndarray, " n"] = video.t_ns
        start: int = 0 if segment.window_ns is None else max((i for i, key in enumerate(video.keyframes) if key and t_ns[i] <= segment.window_ns[0]), default=0)
        codec = {"av1": rr.VideoCodec.AV1, "h264": rr.VideoCodec.H264, "hevc": rr.VideoCodec.H265}[video.codec]
        recording.log(entity, rr.VideoStream(codec=codec), static=True)
        samples = rr.VideoStream.columns(sample=video.samples[start:], is_keyframe=video.keyframes[start:])
        recording.send_columns(entity, indexes=[rr.TimeColumn(segment.timeline, duration=t_ns[start:].astype("timedelta64[ns]"))], columns=samples)


def luma_planes(video: CatalogVideo) -> Iterator[UInt8[ndarray, "h w"]]:
    """A camera's frames in order as libavcodec's raw Y plane, a view that keeps the decoded frame alive (no copy). No range
    conversion: full-range luma the stream leaves unlabelled stays as recorded (``gray8`` would stretch it 255/219, clip at 235)."""
    with av.open(BytesIO(wrap_mp4(video.samples, video.keyframes, MP4_NOMINAL_FPS, codec=video.codec)), mode="r") as container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        stream.thread_type, stream.thread_count = "AUTO", LUMA_DECODE_THREADS
        for frame in container.decode(stream):
            if frame.format.name not in ("yuv420p", "yuvj420p", "nv12", "gray"):
                raise ValueError(f"decoded pixel format {frame.format.name}: no plain Y plane")
            plane: av.video.plane.VideoPlane = frame.planes[0]
            yield np.frombuffer(plane, dtype=np.uint8).reshape(frame.height, plane.line_size)[:, : frame.width]


def sample_indices(
    times_ns: Int64[ndarray, " n"], sample: Literal["all"] | int | tuple[int, ...], window_ns: tuple[int, ...] | None, stride: int = 1
) -> Int64[ndarray, " t"]:
    """Sample indices of a stream: all, a budget spread evenly (rounded linspace), or explicit indices; inside ``window_ns``
    when set, and from every ``stride``-th frame of it (whole strides only: ``m // stride`` frames, as a decoder reading
    ``i * stride``)."""
    candidates: Int64[ndarray, " m"] = np.arange(times_ns.size, dtype=np.int64)
    if window_ns is not None:
        candidates = candidates[(times_ns >= window_ns[0]) & (times_ns < window_ns[1])]
    candidates = candidates[: candidates.size // stride * stride : stride]
    if candidates.size == 0:
        raise ValueError("the stream has no frames (in the window)")
    if isinstance(sample, tuple):
        return np.intersect1d(np.asarray(sample, dtype=np.int64), candidates)
    if sample == "all":
        return candidates
    if sample < 1:
        raise ValueError("a frame budget must be positive")
    return candidates[np.linspace(0, candidates.size - 1, min(sample, candidates.size)).round().astype(np.int64)]


def undistortion_maps(
    K: Float64[ndarray, "3 3"], coefficients: Float64[ndarray, " k"], width: int, height: int
) -> tuple[Float64[ndarray, "3 3"], Float32[ndarray, "h w"], Float32[ndarray, "h w"]]:
    """Intrinsics and ``cv2.remap`` tables that undistort a ``width x height`` OpenCV camera onto a pinhole of the same size:
    alpha 0 (every output pixel sees the scene) and the principal point at the centre (as MASt3R-SLAM's dataloader)."""
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, coefficients, (width, height), 0.0, (width, height), centerPrincipalPoint=True)
    map_x, map_y = cv2.initUndistortRectifyMap(K, coefficients, None, K_new, (width, height), cv2.CV_32FC1)
    return K_new, map_x, map_y


RIG: Column = Column(RigCameras, "/world", static=True)
"""The base layer's calibration, every camera at stored size: a catalog source's ``rig`` port."""


@runtime_checkable
class FrameProvider(Protocol):
    """Random access to a rig's camera videos (``SegmentFrames``: a catalog segment's; mamma's ``FileFrames``: video files)."""

    def times_ns(self, view: int) -> Int64[ndarray, " n"]: ...
    def frames_at(self, view: int, indices: Sequence[int]) -> UInt8[Tensor, "k 3 h w"]: ...
    def close(self) -> None: ...


class UndistortedFrames:
    """A FrameProvider's frames remapped onto undistorted pinholes (``undistortion_maps``).

    ``calibrate`` keeps each camera's remap tables from its stored Brown-Conrady coefficients and returns the rig with the
    undistorted pinholes; ``static_events`` calls it on the rig, before the source decodes a frame.
    """

    def __init__(self, frames: FrameProvider) -> None:
        self.frames: FrameProvider = frames
        self.maps: list[tuple[Float32[ndarray, "h w"], Float32[ndarray, "h w"]]] = []

    def calibrate(self, rig: RigCameras, table: pa.Table) -> RigCameras:
        """``rig`` with every camera's undistorted pinhole (``table``: the static read it came from, with the coefficients)."""
        intrinsics: Float64[ndarray, "v 3 3"] = rig.intrinsics.copy()
        self.maps = []
        for i, camera in enumerate(rig.names):
            column, model = f"/world/{camera}/{DISTORTION}", f"/world/{camera}/{DISTORTION_MODEL}"
            if column not in table.column_names or table.column(column)[0].as_py() is None:
                raise ValueError(f"{camera}: undistort needs the camera's {DISTORTION}")
            if model not in table.column_names or table.column(model)[0].as_py() != ["brown_conrady"]:
                raise ValueError(f"{camera}: undistort takes brown_conrady coefficients, not {table.column(model)[0].as_py() if model in table.column_names else None}")
            coefficients: Float64[ndarray, " k"] = np.asarray(table.column(column)[0].as_py(), dtype=np.float64).reshape(-1)
            width, height = (int(v) for v in rig.resolution_wh[i])
            intrinsics[i], *maps = undistortion_maps(rig.intrinsics[i], coefficients, width, height)
            self.maps.append((maps[0], maps[1]))
        return RigCameras(rig.names, intrinsics, rig.cam_T_world, rig.resolution_wh)

    def times_ns(self, view: int) -> Int64[ndarray, " n"]:
        return self.frames.times_ns(view)

    def frames_at(self, view: int, indices: Sequence[int]) -> UInt8[Tensor, "k 3 h w"]:
        native: UInt8[Tensor, "k 3 h w"] = self.frames.frames_at(view, indices)
        if view >= len(self.maps) or native.shape[-2:] != self.maps[view][0].shape:
            raise ValueError("undistort needs a calibrated rig of the decoded frame size")
        return torch.stack([torch.from_numpy(cv2.remap(frame.permute(1, 2, 0).numpy().copy(), *self.maps[view], cv2.INTER_LINEAR)).permute(2, 0, 1) for frame in native])

    def close(self) -> None:
        self.frames.close()


def static_events(segment: SegmentRef, cameras: tuple[str, ...], statics: Mapping[str, Column], undistort: UndistortedFrames | None = None) -> Iterator[Event]:
    """A catalog segment's STATIC ports, temporal ones read whole: ``rig`` (``RIG``, when every camera has it; undistorted
    through ``undistort``), then ``statics``."""
    dataset = segment.entry()  # one for every port (an entry must outlive its queries)
    for port, column in {"rig": RIG, **statics}.items():
        table = read_column(dataset, segment.segment, column.type, column.root, segment.timeline, static=column.static)
        for _, value in column.type.from_layer(table, column.root, segment.timeline):
            if port != "rig":
                yield Event(STATIC, port, value)
            elif isinstance(value, RigCameras) and set(cameras) <= set(value.names):
                rig: RigCameras = value.select(cameras)
                yield Event(STATIC, port, rig if undistort is None else undistort.calibrate(rig, table))


class VideoSource:
    """Decode the cameras' videos.

    The STATIC values of ``statics`` first, then RGB framesets: every camera's frame at sampled indices of the first camera
    inside ``window_ns``, timed by it (``Fenced[MultiViewFrames]`` when ``fenced``, else ``MultiViewFrames``). ``resize_hw``
    applies area resize; ``keep_native`` keeps the stored-size decode in the same value (``native``). ``catalog`` reads a
    catalog segment's videos and rig.
    """

    def __init__(
        self,
        frames: FrameProvider,
        cameras: tuple[str, ...],
        statics: Source,
        *,
        window_ns: tuple[int, ...] | None = None,
        fenced: bool = False,
        sample: Literal["all"] | int | tuple[int, ...] = "all",
        stride: int = 1,
        resize_hw: tuple[int, int] | None = None,
        keep_native: bool = False,
    ) -> None:
        if keep_native and resize_hw is None:
            raise NotImplementedError("keep_native needs resize_hw")
        self.frames, self.cameras, self.statics, self.window_ns, self.fenced = frames, cameras, statics, window_ns, fenced
        self.sample, self.stride, self.resize_hw, self.keep_native = sample, stride, resize_hw, keep_native
        self.clock: str = statics.clock
        self.outputs: dict[str, PortType] = {"frames": Fenced[MultiViewFrames] if fenced else MultiViewFrames} | dict(statics.outputs)

    @classmethod
    def catalog(
        cls,
        segment: SegmentRef,
        cameras: tuple[str, ...],
        *,
        decode: Literal["nvdec_rgb", "cpu_rgb"],
        sample: Literal["all"] | int | tuple[int, ...] = "all",
        stride: int = 1,
        resize_hw: tuple[int, int] | None = None,
        keep_native: bool = False,
        statics: Mapping[str, Column] | None = None,
        undistort: bool = False,
    ) -> "VideoSource":
        """A catalog segment's cameras after its ``rig`` and ``statics`` (``static_events``): ``nvdec_rgb`` decodes on the GPU
        (``Fenced`` frames), ``cpu_rgb`` on the CPU, where ``undistort`` remaps every frame with its camera's stored
        coefficients and the rig carries the undistorted pinholes."""
        if undistort and decode != "cpu_rgb":
            raise NotImplementedError("undistort remaps cpu_rgb frames only")
        columns: dict[str, Column] = dict(statics or {})
        frames = SegmentFrames(segment, cameras, "cuda" if decode == "nvdec_rgb" else "cpu")
        undistorted: UndistortedFrames | None = UndistortedFrames(frames) if undistort else None
        ports: dict[str, PortType] = {"rig": RigCameras} | {port: column.type for port, column in columns.items()}
        reads = IterSource(ports, lambda: static_events(segment, cameras, columns, undistorted), clock=segment.timeline)
        return cls(
            undistorted or frames, cameras, reads, window_ns=segment.window_ns, fenced=decode == "nvdec_rgb", sample=sample, stride=stride,
            resize_hw=resize_hw, keep_native=keep_native,
        )  # fmt: skip

    def prefetch(self) -> None:
        """Read every camera's sample times now (for a catalog segment: its packets), so a run's clock starts at the first decode."""
        for view in range(len(self.cameras)):
            self.frames.times_ns(view)

    def events(self) -> Iterator[Event]:
        yield from self.statics.events()
        times: list[Int64[ndarray, " n"]] = [self.frames.times_ns(view) for view in range(len(self.cameras))]
        shortest: int = min(t.size for t in times)  # index-aligned: every sampled index exists in every camera
        for index in sample_indices(times[0][:shortest], self.sample, self.window_ns, self.stride).tolist():
            native: UInt8[Tensor, "v 3 hh ww"] = torch.stack([self.frames.frames_at(view, [index])[0] for view in range(len(self.cameras))])
            rgb: UInt8[Tensor, "v 3 h w"] = native
            if self.resize_hw is not None:
                rgb = torch.nn.functional.interpolate(native.float(), size=self.resize_hw, mode="area").round_().clamp_(0, 255).to(torch.uint8)
            cam_t_ns: Int64[Tensor, " v"] = torch.as_tensor([int(t[index]) for t in times])
            frames: MultiViewFrames = MultiViewFrames(int(times[0][index]), index, self.cameras, rgb, cam_t_ns, native if self.keep_native else None)
            yield Event(frames.t_ns, "frames", Fenced(frames, fence_after(rgb)) if self.fenced else frames)

    def close(self) -> None:
        self.frames.close()
        self.statics.close()


class CatalogLumaSource:
    """Decode the cameras' luma planes.

    A catalog segment's ``rig`` and ``statics`` (``static_events``), then a ``LumaFrameset`` for every frame in the window:
    each camera's raw Y plane, grouped across cameras within ``group_tolerance_ns`` and timed by its opening frame.
    """

    def __init__(self, segment: SegmentRef, cameras: tuple[str, ...], *, group_tolerance_ns: int, statics: Mapping[str, Column] | None = None) -> None:
        self.segment, self.cameras, self.group_tolerance_ns = segment, cameras, group_tolerance_ns
        self.statics: dict[str, Column] = dict(statics or {})
        self.clock: str = segment.timeline
        self.outputs: dict[str, PortType] = {"frames": LumaFrameset, "rig": RigCameras} | {port: column.type for port, column in self.statics.items()}

    def events(self) -> Iterator[Event]:
        yield from static_events(self.segment, self.cameras, self.statics)
        lo, hi = self.segment.window_ns or (None, None)
        read_window: tuple[int, int] | None = None if lo is None or hi is None else (lo - KEYFRAME_LEAD_NS, hi)
        entities: list[str] = [f"/world/{camera}/pinhole/video" for camera in self.cameras]
        videos: tuple[CatalogVideo, ...] = read_catalog_videos(self.segment.entry(), self.segment.segment, entities, self.clock, read_window)
        decoders: list[Iterator[UInt8[ndarray, "h w"]]] = []
        inside: list[Int64[ndarray, " n"]] = []
        for camera, video in zip(self.cameras, videos, strict=True):
            first: int = 0 if lo is None else int(np.searchsorted(video.t_ns, lo))
            key: int | None = next((i for i in range(min(first, len(video.samples) - 1), -1, -1) if video.keyframes[i]), None)
            if first == len(video.samples) or key is None:
                raise ValueError(f"{self.segment.segment} {camera}: no frame in the window, or no keyframe {KEYFRAME_LEAD_NS} ns before it")
            planes = luma_planes(CatalogVideo(video.times[key:], video.samples[key:], video.keyframes[key:], video.codec))
            decoders.append(itertools.islice(planes, first - key, None))  # decoded from the keyframe, kept from the window's first frame
            inside.append(video.t_ns[first:])
        index: FramesetIndex = group_framesets(inside, self.group_tolerance_ns)
        for row, t_ns in enumerate(index.t_ns.tolist()):
            cam_t_ns = tuple(int(t) if f >= 0 else None for t, f in zip(index.cam_t_ns[row], index.frame[row], strict=True))  # None: absent
            luma = tuple(next(decoder, None) if t is not None else None for decoder, t in zip(decoders, cam_t_ns, strict=True))
            if any(t is not None and plane is None for t, plane in zip(cam_t_ns, luma, strict=True)):
                raise ValueError(f"{self.segment.segment}: a camera's decoder ended before its last packet")
            yield Event(t_ns, "frames", LumaFrameset(t_ns, self.cameras, luma, cam_t_ns))
        if any(next(decoder, None) is not None for decoder in decoders):
            raise ValueError(f"{self.segment.segment}: a camera decoded more frames than it has packets")

    def close(self) -> None:
        pass
