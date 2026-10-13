"""Catalog video as a dagkit Source: one time-ordered stream of decoded framesets plus the segment's static rig (SPEC 2.6)."""

import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Literal

import numpy as np
import torch
from dagkit.node import Event, Fence, Fenced, PortType
from dagkit.sources import Column, SegmentRef
from dagkit.time import STATIC
from dagkit.types import read_column
from jaxtyping import Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from simplecv.catalog_video_codec import catalog_codec_name, wrap_mp4
from simplecv.gpu import fence_after
from simplecv.types.frames import MultiViewFrames
from simplecv.types.rig import RigCameras

MP4_NOMINAL_FPS: int = 30
"""Frame rate written into the MP4 wrapper around the catalog packets: frames are addressed by sample index and the
times come from the catalog, so it only labels the container."""


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
        self._decoders: dict[tuple[int, int], VideoDecoder] = {}

    def _stream(self, view: int) -> tuple[Int64[ndarray, " n"], VideoDecoder]:
        if view not in self._packets:
            from simplecv.rerun_dataloader import read_segment_packets  # needs the rerun dataloader extra

            self._dataset = self._dataset or self.segment.entry()  # the entry must outlive its queries
            times, samples, keyframes, codec = read_segment_packets(self._dataset, self.segment.segment, f"world/{self.cameras[view]}/pinhole/video", self.segment.timeline)
            mp4: bytes = wrap_mp4(samples, keyframes, MP4_NOMINAL_FPS, codec=catalog_codec_name(codec))
            self._packets[view] = (np.asarray(times).astype("timedelta64[ns]").astype(np.int64), mp4)
        key: tuple[int, int] = (threading.get_ident(), view)
        if key not in self._decoders:
            self._decoders[key] = VideoDecoder(self._packets[view][1], device=self.device, seek_mode="exact", num_ffmpeg_threads=0)
        return self._packets[view][0], self._decoders[key]

    def times_ns(self, view: int) -> Int64[ndarray, " n"]:
        """Sample times of one camera on the segment's timeline."""
        return self._stream(view)[0]

    def frames_at(self, view: int, indices: Sequence[int]) -> UInt8[Tensor, "k 3 h w"]:
        return self._stream(view)[1].get_frames_at([int(i) for i in indices]).data

    def close(self) -> None:
        self._decoders.clear()
        self._packets.clear()


def sample_indices(times_ns: Int64[ndarray, " n"], sample: Literal["all"] | int | tuple[int, ...], window_ns: tuple[int, int] | None) -> Int64[ndarray, " t"]:
    """Sample indices of a stream: all, a budget spread evenly (rounded linspace), or explicit indices; inside ``window_ns`` when set."""
    candidates: Int64[ndarray, " m"] = np.arange(times_ns.size, dtype=np.int64)
    if window_ns is not None:
        candidates = candidates[(times_ns >= window_ns[0]) & (times_ns < window_ns[1])]
    if candidates.size == 0:
        raise ValueError("the stream has no frames (in the window)")
    if isinstance(sample, tuple):
        return np.intersect1d(np.asarray(sample, dtype=np.int64), candidates)
    if sample == "all":
        return candidates
    if sample < 1:
        raise ValueError("a frame budget must be positive")
    return candidates[np.linspace(0, candidates.size - 1, min(sample, candidates.size)).round().astype(np.int64)]


class CatalogVideoSource:
    """Framesets of ``cameras`` at the sampled indices (index-aligned; ``t_ns`` = the first camera's time), after the STATIC ports.

    Ports: ``frames`` (``Fenced[MultiViewFrames]`` for ``nvdec_rgb``, ``MultiViewFrames`` for ``cpu_rgb``), ``rig`` (STATIC
    ``RigCameras`` of the base layer at the stored size, only when every camera has a calibration) and one STATIC port per
    ``statics`` key. With ``resize_hw`` the frames are area-resized after decode; ``keep_native`` adds ``frames_native``
    (same type, decoded size), emitted just before ``frames`` at the same time.
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
    ) -> None:
        if decode == "cpu_luma" or group_tolerance_ns is not None or (keep_native and resize_hw is None):
            raise NotImplementedError("cpu_luma decode and frameset grouping land with robocap-live; keep_native needs resize_hw")
        self.segment, self.cameras, self.sample, self.fenced = segment, cameras, sample, decode == "nvdec_rgb"
        self.resize_hw, self.keep_native = resize_hw, keep_native
        self.statics: dict[str, Column] = {"rig": Column(RigCameras, "/world", static=True), **(statics or {})}
        self.clock: str = segment.timeline
        frame_type: PortType = Fenced[MultiViewFrames] if self.fenced else MultiViewFrames
        self.outputs: dict[str, PortType] = {"frames": frame_type} | ({"frames_native": frame_type} if keep_native else {})
        self.outputs |= {port: column.type for port, column in self.statics.items()}
        self.frames = SegmentFrames(segment, cameras, "cuda" if self.fenced else "cpu")

    def events(self) -> Iterator[Event]:
        for port, column in self.statics.items():  # the entry lives through the call (an entry must outlive its queries)
            table = read_column(self.segment.entry(), self.segment.segment, column.type, column.root, self.clock, static=True)
            for _, value in column.type.from_layer(table, column.root, self.clock):
                if port != "rig":
                    yield Event(STATIC, port, value)
                elif isinstance(value, RigCameras) and set(self.cameras) <= set(value.names):
                    yield Event(STATIC, port, value.select(self.cameras))
        times: list[Int64[ndarray, " n"]] = [self.frames.times_ns(view) for view in range(len(self.cameras))]
        shortest: int = min(t.size for t in times)  # index-aligned: every sampled index exists in every camera
        for index in sample_indices(times[0][:shortest], self.sample, self.segment.window_ns).tolist():
            native: UInt8[Tensor, "v 3 hh ww"] = torch.stack([self.frames.frames_at(view, [index])[0] for view in range(len(self.cameras))])
            rgb: UInt8[Tensor, "v 3 h w"] = native
            if self.resize_hw is not None:
                rgb = torch.nn.functional.interpolate(native.float(), size=self.resize_hw, mode="area").round_().clamp_(0, 255).to(torch.uint8)
            cam_t_ns: Int64[Tensor, " v"] = torch.as_tensor([int(t[index]) for t in times])
            ports: dict[str, UInt8[Tensor, "v 3 _ _"]] = ({"frames_native": native} if self.keep_native else {}) | {"frames": rgb}
            fence: Fence | None = fence_after(rgb)
            for port, pixels in ports.items():
                frames = MultiViewFrames(int(times[0][index]), index, self.cameras, pixels, cam_t_ns)
                yield Event(frames.t_ns, port, Fenced(frames, fence) if self.fenced else frames)

    def close(self) -> None:
        self.frames.close()
