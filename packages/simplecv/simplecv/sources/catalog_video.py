"""Catalog video as a dagkit Source: one time-ordered stream of decoded framesets plus the segment's static rig (SPEC 2.6)."""

import itertools
import threading
from collections.abc import Iterator, Mapping, Sequence
from io import BytesIO
from typing import Any, Literal

import av
import cv2
import numpy as np
import pyarrow as pa
import torch
from dagkit.node import Event, Fence, Fenced, PortType
from dagkit.sources import Column, SegmentRef
from dagkit.time import STATIC
from dagkit.types import read_column
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from torch import Tensor

from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.catalog_video_codec import catalog_codec_name, wrap_mp4
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
"""A camera's OpenCV (Brown-Conrady) coefficients beside its pinhole, zero-padded to 14 (dataforge's layout)."""


class SegmentFrames:
    """Random access to one segment's camera videos (``/world/<camera>/pinhole/video``); packets fetched once per camera.

    A torchcodec NVDEC decoder only decodes on the thread that created it (the runner reads a Source's first frameset on
    the caller's thread, the rest on the Source's own), so decoders are kept per thread over the cached packets. Also a
    graph resource (exo-calib's metric rescale reads a few frames per camera); ``close`` drops everything.
    """

    def __init__(self, segment: SegmentRef, cameras: tuple[str, ...], device: Literal["cuda", "cpu"] = "cuda") -> None:
        self.segment, self.cameras, self.device = segment, cameras, device
        self._dataset: DatasetEntry | None = None
        self._packets: dict[int, tuple[Int64[ndarray, " n"], bytes]] = {}
        self._decoders: dict[tuple[int, int], Any] = {}  # torchcodec VideoDecoder per (thread, view), imported lazily

    def times_ns(self, view: int) -> Int64[ndarray, " n"]:
        """Sample times of one camera on the segment's timeline (its packets are fetched on the first call)."""
        if view not in self._packets:
            from simplecv.rerun_dataloader import read_segment_packets  # needs the rerun dataloader extra

            self._dataset = self._dataset or self.segment.entry()  # the entry must outlive its queries
            times, samples, keyframes, codec = read_segment_packets(self._dataset, self.segment.segment, f"world/{self.cameras[view]}/pinhole/video", self.segment.timeline)
            mp4: bytes = wrap_mp4(samples, keyframes, MP4_NOMINAL_FPS, codec=catalog_codec_name(codec))
            self._packets[view] = (np.asarray(times).astype("timedelta64[ns]").astype(np.int64), mp4)
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
    times_ns: Int64[ndarray, " n"], sample: Literal["all"] | int | tuple[int, ...], window_ns: tuple[int, int] | None, stride: int = 1
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


class CatalogVideoSource:
    """Decoded framesets after STATIC ``rig`` (base-layer calibration at stored size, when all cameras have it) and ``statics``.

    Temporal statics are read whole. RGB frames are index-aligned at sampled indices, timed by the first camera:
    ``nvdec_rgb`` emits ``Fenced[MultiViewFrames]``; ``cpu_rgb`` emits ``MultiViewFrames``. ``resize_hw`` applies area resize;
    ``keep_native`` emits ``frames_native`` just before ``frames`` at the same time.
    ``cpu_luma`` emits ``LumaFrameset`` for every frame in the window, grouped by ``group_tolerance_ns`` and timed by its opening.
    ``undistort`` (``cpu_rgb``): every frame is remapped with its camera's stored coefficients (``undistortion_maps``) and
    the rig carries the undistorted pinhole.
    """

    def __init__(
        self,
        segment: SegmentRef,
        cameras: tuple[str, ...],
        *,
        decode: Literal["nvdec_rgb", "cpu_rgb", "cpu_luma"],
        sample: Literal["all"] | int | tuple[int, ...] = "all",
        group_tolerance_ns: int | None = None,
        resize_hw: tuple[int, int] | None = None,
        keep_native: bool = False,
        statics: Mapping[str, Column] | None = None,
        stride: int = 1,
        undistort: bool = False,
    ) -> None:
        luma: bool = decode == "cpu_luma"
        if undistort and decode != "cpu_rgb":
            raise NotImplementedError("undistort remaps cpu_rgb frames only")
        self.undistort: bool = undistort
        self.maps: list[tuple[Float32[ndarray, "h w"], Float32[ndarray, "h w"]]] = []
        if luma != (group_tolerance_ns is not None) or (luma and (resize_hw is not None or isinstance(sample, int | tuple))) or (keep_native and resize_hw is None):
            raise NotImplementedError("cpu_luma groups every frame (group_tolerance_ns); rgb is index-aligned; keep_native needs resize_hw")
        self.segment, self.cameras, self.sample, self.stride, self.fenced = segment, cameras, sample, stride, decode == "nvdec_rgb"
        self.group_tolerance_ns: int | None = group_tolerance_ns
        self.resize_hw, self.keep_native = resize_hw, keep_native
        self.statics: dict[str, Column] = {"rig": Column(RigCameras, "/world", static=True), **(statics or {})}
        self.clock: str = segment.timeline
        frame_type: PortType = LumaFrameset if luma else Fenced[MultiViewFrames] if self.fenced else MultiViewFrames
        self.outputs: dict[str, PortType] = {"frames": frame_type} | ({"frames_native": frame_type} if keep_native else {})
        self.outputs |= {port: column.type for port, column in self.statics.items()}
        self.frames = SegmentFrames(segment, cameras, "cuda" if self.fenced else "cpu")

    def events(self) -> Iterator[Event]:
        for port, column in self.statics.items():  # the entry lives through the call (an entry must outlive its queries)
            table = read_column(self.segment.entry(), self.segment.segment, column.type, column.root, self.clock, static=column.static)
            for _, value in column.type.from_layer(table, column.root, self.clock):
                if port != "rig":
                    yield Event(STATIC, port, value)
                elif isinstance(value, RigCameras) and set(self.cameras) <= set(value.names):
                    yield Event(STATIC, port, self._undistorted(value.select(self.cameras), table) if self.undistort else value.select(self.cameras))
        if self.group_tolerance_ns is not None:
            yield from self._luma_framesets(self.group_tolerance_ns)
            return
        times: list[Int64[ndarray, " n"]] = [self.frames.times_ns(view) for view in range(len(self.cameras))]
        shortest: int = min(t.size for t in times)  # index-aligned: every sampled index exists in every camera
        for index in sample_indices(times[0][:shortest], self.sample, self.segment.window_ns, self.stride).tolist():
            native: UInt8[Tensor, "v 3 hh ww"] = torch.stack([self.frames.frames_at(view, [index])[0] for view in range(len(self.cameras))])
            if self.undistort:
                if len(self.maps) != len(self.cameras) or native.shape[-2:] != self.maps[0][0].shape:
                    raise ValueError(f"{self.segment.segment}: undistort needs a calibrated rig of the decoded frame size")
                native = torch.stack([torch.from_numpy(cv2.remap(view.permute(1, 2, 0).numpy().copy(), *maps, cv2.INTER_LINEAR)).permute(2, 0, 1) for view, maps in zip(native, self.maps, strict=True)])
            rgb: UInt8[Tensor, "v 3 h w"] = native
            if self.resize_hw is not None:
                rgb = torch.nn.functional.interpolate(native.float(), size=self.resize_hw, mode="area").round_().clamp_(0, 255).to(torch.uint8)
            cam_t_ns: Int64[Tensor, " v"] = torch.as_tensor([int(t[index]) for t in times])
            ports: dict[str, UInt8[Tensor, "v 3 _ _"]] = ({"frames_native": native} if self.keep_native else {}) | {"frames": rgb}
            fence: Fence | None = fence_after(rgb)
            for port, pixels in ports.items():
                frames = MultiViewFrames(int(times[0][index]), index, self.cameras, pixels, cam_t_ns)
                yield Event(frames.t_ns, port, Fenced(frames, fence) if self.fenced else frames)

    def _undistorted(self, rig: RigCameras, table: pa.Table) -> RigCameras:
        """The rig with every camera's undistorted pinhole; keeps the remap tables for the frames."""
        intrinsics: Float64[ndarray, "v 3 3"] = rig.intrinsics.copy()
        for i, camera in enumerate(self.cameras):
            column: str = f"/world/{camera}/{DISTORTION}"
            if column not in table.column_names or table.column(column)[0].as_py() is None:
                raise ValueError(f"{self.segment.segment} {camera}: undistort needs the camera's {DISTORTION}")
            coefficients: Float64[ndarray, " k"] = np.asarray(table.column(column)[0].as_py(), dtype=np.float64).reshape(-1)
            width, height = (int(v) for v in rig.resolution_wh[i])
            intrinsics[i], *maps = undistortion_maps(rig.intrinsics[i], coefficients, width, height)
            self.maps.append((maps[0], maps[1]))
        return RigCameras(rig.names, intrinsics, rig.cam_T_world, rig.resolution_wh)

    def _luma_framesets(self, tolerance_ns: int) -> Iterator[Event]:
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
        index: FramesetIndex = group_framesets(inside, tolerance_ns)
        for row, t_ns in enumerate(index.t_ns.tolist()):
            cam_t_ns = tuple(int(t) if f >= 0 else None for t, f in zip(index.cam_t_ns[row], index.frame[row], strict=True))  # None: absent
            luma = tuple(next(decoder, None) if t is not None else None for decoder, t in zip(decoders, cam_t_ns, strict=True))
            if any(t is not None and plane is None for t, plane in zip(cam_t_ns, luma, strict=True)):
                raise ValueError(f"{self.segment.segment}: a camera's decoder ended before its last packet")
            yield Event(t_ns, "frames", LumaFrameset(t_ns, self.cameras, luma, cam_t_ns))
        if any(next(decoder, None) is not None for decoder in decoders):
            raise ValueError(f"{self.segment.segment}: a camera decoded more frames than it has packets")

    def close(self) -> None:
        self.frames.close()
