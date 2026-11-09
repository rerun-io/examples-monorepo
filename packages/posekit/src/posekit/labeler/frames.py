"""Decoded camera frames of one catalog segment on the labeler's 5 fps grid.

A ``FrameSource`` reads the compressed video of every rig camera of one segment from the catalog in one
round trip, wraps each camera's packets into an in-memory MP4 and drops the Arrow copy. Two decode paths
share it:

- ``frame(cam, t_ns)``: random access for the backend (CPU torchcodec, one decoder per camera behind a lock,
  seeking to the previous keyframe; a small in-memory LRU of decoded frames, nothing on disk).
- ``iter_grid(...)``: the proposer's batch path, one decoder per camera that walks the camera's grid frames in
  time order (NVDEC on ``cuda``) and yields them as GPU batches.

The grid: every ``period_ns`` (200 ms) of the reference camera's real frame times, gaps in the recording
skipped. One grid time names a *frame set*: each camera's nearest frame within ``FRAMESET_TOLERANCE_NS``. The
grid time is the latest frame time of its set, so a latest-at query at the grid time (what the Rerun viewer
does) lands on every camera's frame of the set, never on the frame before it.
"""

import re
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import torch
from jaxtyping import Bool, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from posekit.labeler.paths import pinhole_path, video_path
from posekit.labeler.schema import CameraInfo, SegmentInfo, SegmentSplit

DecodeDevice: TypeAlias = Literal["cpu", "cuda"]
"""Where torchcodec decodes: FFmpeg on the CPU, or NVDEC."""

TIMELINE: str = "video_time"
"""Timeline of the RoboCap video streams (a duration timeline, ns)."""
GRID_PERIOD_NS: int = 200_000_000
"""Label grid spacing: 5 fps."""
KEYFRAME_STRIDE: int = 5
"""Every fifth grid time is a review keyframe: 1 fps (and the last grid time, so range fill reaches the end)."""
GRID_TOLERANCE_NS: int = 50_000_000
"""Largest gap between a grid target and the reference frame that fills it; larger gaps in the recording are skipped."""
FRAMESET_TOLERANCE_NS: int = 10_000_000
"""Largest gap between frames of one frame set (RoboCap cameras sit within ~50 us of each other)."""
REFERENCE_CAM: int = 2
"""``cam_02`` (left_eye) sets the grid times."""
VIDEO_ENTITY: re.Pattern[str] = re.compile(r"^/world/rig_00/cam_(\d{2})/pinhole/video$")
"""Camera video entities of rig 0."""


@dataclass(frozen=True, slots=True)
class CameraVideo:
    """One camera's whole video for a segment: its frame times and the packets muxed into an MP4."""

    info: CameraInfo
    """Index, rig name and native size."""
    times_ns: Int64[ndarray, "n"]
    """``video_time`` of every frame, ns, strictly increasing."""
    mp4: bytes
    """All packets in an in-memory MP4 with positional pts (frame ``i`` has pts ``i``)."""


@dataclass(frozen=True, slots=True)
class GridBatch:
    """Consecutive grid frames of one camera."""

    cam: int
    """Camera index."""
    times_ns: Int64[ndarray, "b"]
    """Grid time of each frame, ns."""
    frames_rgb: UInt8[Tensor, "b h w 3"]
    """Decoded RGB frames on the decode device."""


def nearest_indices(times_ns: Int64[ndarray, "n"], targets_ns: Int64[ndarray, "m"], tolerance_ns: int) -> Int64[ndarray, "m"]:
    """Index of the nearest ``times_ns`` entry per target; -1 when none lies within ``tolerance_ns``.

    Args:
        times_ns: Sorted times, ns.
        targets_ns: Query times, ns.
        tolerance_ns: Largest accepted distance, ns.
    """
    if not len(times_ns):
        return np.full(len(targets_ns), -1, dtype=np.int64)
    right: Int64[ndarray, "m"] = np.clip(np.searchsorted(times_ns, targets_ns), 0, len(times_ns) - 1).astype(np.int64)
    left: Int64[ndarray, "m"] = np.clip(right - 1, 0, len(times_ns) - 1)
    best: Int64[ndarray, "m"] = np.where(np.abs(times_ns[left] - targets_ns) <= np.abs(times_ns[right] - targets_ns), left, right)
    return np.where(np.abs(times_ns[best] - targets_ns) <= tolerance_ns, best, -1).astype(np.int64)


def grid_indices(times_ns: Int64[ndarray, "n"], period_ns: int, tolerance_ns: int) -> Int64[ndarray, "g"]:
    """Frames of one camera that fall on a fixed-period grid; grid targets inside recording gaps are skipped.

    The targets are ``times_ns[0] + k * period_ns`` and each takes its nearest frame, so dropped frames do not
    make the grid drift.

    Args:
        times_ns: The camera's frame times, ns, strictly increasing.
        period_ns: Grid spacing, ns.
        tolerance_ns: Largest distance between a target and the frame that fills it, ns.

    Returns:
        Increasing, unique frame indices.
    """
    if not len(times_ns):
        return np.zeros(0, dtype=np.int64)
    targets: Int64[ndarray, "t"] = np.arange(int(times_ns[0]), int(times_ns[-1]) + 1, period_ns, dtype=np.int64)
    hits: Int64[ndarray, "t"] = nearest_indices(times_ns, targets, tolerance_ns)
    return np.unique(hits[hits >= 0]).astype(np.int64)


def frameset_samples(videos: Sequence[CameraVideo], reference_cam: int, period_ns: int, tolerance_ns: int) -> tuple[Int64[ndarray, "g"], Int64[ndarray, "c g"]]:
    """The grid times and, per camera, the frame each grid time shows (-1 when the camera has none).

    Args:
        videos: Every camera, rig order.
        reference_cam: Position in ``videos`` of the camera whose frames set the grid.
        period_ns: Grid spacing, ns.
        tolerance_ns: Largest distance between the reference frame and another camera's frame of the same set, ns.

    Returns:
        Grid times (the latest frame time of each set) and the frame index per (camera, grid time).
    """
    reference: Int64[ndarray, "n"] = videos[reference_cam].times_ns
    targets: Int64[ndarray, "g"] = reference[grid_indices(reference, period_ns, GRID_TOLERANCE_NS)]
    samples: Int64[ndarray, "c g"] = np.stack([nearest_indices(video.times_ns, targets, tolerance_ns) for video in videos]).astype(np.int64)
    frame_times: Int64[ndarray, "c g"] = np.stack(
        [np.where(row >= 0, video.times_ns[np.clip(row, 0, None)], np.iinfo(np.int64).min) for video, row in zip(videos, samples, strict=True)]
    )
    grid_ns: Int64[ndarray, "g"] = frame_times.max(axis=0).astype(np.int64)
    return grid_ns, samples


class FrameSource:
    """Cameras, label grid and RGB frames of one catalog segment."""

    def __init__(
        self,
        segment_id: str,
        videos: Sequence[CameraVideo],
        *,
        reference_cam: int = REFERENCE_CAM,
        period_ns: int = GRID_PERIOD_NS,
        tolerance_ns: int = FRAMESET_TOLERANCE_NS,
        cache_frames: int = 36,
    ) -> None:
        """Build the grid from the cameras' frame times.

        Args:
            segment_id: Catalog segment id.
            videos: Every camera, rig order (``videos[i].info.cam == i``).
            reference_cam: Camera whose frames set the grid.
            period_ns: Grid spacing, ns.
            tolerance_ns: Largest distance between frames of one frame set, ns.
            cache_frames: Decoded frames kept in memory for ``frame``.
        """
        if [video.info.cam for video in videos] != list(range(len(videos))):
            raise ValueError(f"{segment_id}: cameras must be indexed 0..{len(videos) - 1} in order")
        self.segment_id: str = segment_id
        self.videos: tuple[CameraVideo, ...] = tuple(videos)
        self.tolerance_ns: int = tolerance_ns
        grid_ns, samples = frameset_samples(self.videos, reference_cam, period_ns, tolerance_ns)
        self.grid_ns: Int64[ndarray, "g"] = grid_ns
        self.grid_samples: Int64[ndarray, "c g"] = samples
        self._grid_column: dict[int, int] = {int(t): column for column, t in enumerate(grid_ns.tolist())}
        self._cache_frames: int = cache_frames
        self._cache: OrderedDict[tuple[int, int], UInt8[ndarray, "h w 3"]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._decoders: dict[int, VideoDecoder] = {}
        self._decoder_locks = tuple(threading.Lock() for _ in self.videos)

    @classmethod
    def from_catalog(cls, dataset: DatasetEntry, segment_id: str, *, reference_cam: int = REFERENCE_CAM, cache_frames: int = 36) -> "FrameSource":
        """Read every rig camera of ``segment_id``: names and sizes (statics) and all video packets (one round trip each).

        Raises:
            ValueError: If the segment has no rig-0 camera videos or a camera lacks its name or resolution.
        """
        cams: list[int] = sorted(
            int(match.group(1))
            for column in dataset.filter_segments(segment_id).schema().component_columns()
            if column.component == "VideoStream:sample" and (match := VIDEO_ENTITY.match(column.entity_path))
        )
        if cams != list(range(len(cams))) or not cams:
            raise ValueError(f"{segment_id}: expected rig-0 cameras cam_00..cam_NN, found {cams}")
        rig_entities: list[str] = [pinhole_path(cam).removesuffix("/pinhole") for cam in cams]
        statics: pa.Table = dataset.filter_segments(segment_id).filter_contents(rig_entities + [pinhole_path(cam) for cam in cams]).reader(index=None).to_arrow_table()
        infos: list[CameraInfo] = []
        for cam, rig_entity in zip(cams, rig_entities, strict=True):
            names: list = statics[f"{rig_entity}:name"].drop_null().to_pylist()
            resolutions: list = statics[f"{pinhole_path(cam)}:Pinhole:resolution"].drop_null().to_pylist()
            if not names or not resolutions:
                raise ValueError(f"{segment_id} cam_{cam:02d}: missing the camera name or the pinhole resolution")
            width, height = (int(round(value)) for value in resolutions[0][0])
            infos.append(CameraInfo(cam=cam, name=str(names[0][0]), width=width, height=height))
        catalog_videos: tuple[CatalogVideo, ...] = read_catalog_videos(dataset, segment_id, [video_path(cam) for cam in cams], TIMELINE)
        videos: list[CameraVideo] = []
        for info, video in zip(infos, catalog_videos, strict=True):
            times: Int64[ndarray, "n"] = video.t_ns.astype(np.int64)
            fps: int = round(1e9 / float(np.median(np.diff(times)))) if len(times) > 1 else 30
            videos.append(CameraVideo(info, times, wrap_mp4(video.samples, video.keyframes, fps, codec=video.codec)))
        return cls(segment_id, videos, reference_cam=reference_cam, cache_frames=cache_frames)

    @property
    def cams(self) -> list[CameraInfo]:
        """Cameras in rig order."""
        return [video.info for video in self.videos]

    @property
    def times_ns(self) -> list[int]:
        """The 5 fps label grid on ``video_time``, ns."""
        return self.grid_ns.tolist()

    @property
    def keyframes_ns(self) -> list[int]:
        """The 1 fps review keyframes: every ``KEYFRAME_STRIDE``-th grid time, plus the last grid time.

        The last one closes the final run: range fill only covers times strictly between two keyframes, so
        without it the grid times after the last stride keyframe would never be labelled.
        """
        keyframes: list[int] = self.grid_ns[::KEYFRAME_STRIDE].tolist()
        if len(self.grid_ns) and keyframes[-1] != int(self.grid_ns[-1]):
            keyframes.append(int(self.grid_ns[-1]))
        return keyframes

    def segment_info(self, split: SegmentSplit, viewer_url: str) -> SegmentInfo:
        """What the browser needs to lay out this segment."""
        return SegmentInfo(segment_id=self.segment_id, split=split, cams=self.cams, times_ns=self.times_ns, keyframes_ns=self.keyframes_ns, viewer_url=viewer_url)

    def sample_index(self, cam: int, t_ns: int) -> int:
        """The frame of ``cam`` shown at ``t_ns``: the frame-set member for a grid time, else the nearest frame.

        Raises:
            KeyError: If the camera has no frame within the frame-set tolerance of ``t_ns``.
        """
        column: int | None = self._grid_column.get(t_ns)
        if column is not None:
            index: int = int(self.grid_samples[cam, column])
        else:
            index = int(nearest_indices(self.videos[cam].times_ns, np.asarray([t_ns], dtype=np.int64), self.tolerance_ns)[0])
        if index < 0:
            raise KeyError(f"{self.segment_id} cam_{cam:02d}: no frame within {self.tolerance_ns / 1e6:.0f} ms of {t_ns}")
        return index

    def frame(self, cam: int, t_ns: int) -> UInt8[ndarray, "h w 3"]:
        """Native-size RGB frame of ``cam`` at ``t_ns`` (CPU decode; thread-safe; cached).

        Raises:
            KeyError: If the camera has no frame near ``t_ns``.
        """
        key: tuple[int, int] = (cam, self.sample_index(cam, t_ns))
        with self._cache_lock:
            cached: UInt8[ndarray, "h w 3"] | None = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        with self._decoder_locks[cam]:
            decoder: VideoDecoder | None = self._decoders.get(cam)
            if decoder is None:
                decoder = VideoDecoder(self.videos[cam].mp4, device="cpu", seek_mode="exact", dimension_order="NHWC")
                self._decoders[cam] = decoder
            rgb: UInt8[ndarray, "h w 3"] = decoder.get_frame_at(key[1]).data.numpy()
        with self._cache_lock:
            self._cache[key] = rgb
            while len(self._cache) > self._cache_frames:
                self._cache.popitem(last=False)
        return rgb

    def iter_grid(
        self, cams: Sequence[int] | None = None, *, batch_size: int = 32, device: DecodeDevice = "cuda", max_frames: int | None = None
    ) -> Iterator[GridBatch]:
        """Every grid frame of each camera, camera after camera, in time order.

        Each camera gets a fresh decoder created here, on the consuming thread (NVDEC decoders are per-thread);
        the decoder only moves forward, so each GOP is decoded once.

        Args:
            cams: Cameras to walk; all by default.
            batch_size: Frames per yielded batch.
            device: ``cuda`` for NVDEC, ``cpu`` for FFmpeg.
            max_frames: Walk only the first ``max_frames`` grid times (smoke runs).
        """
        if device == "cuda":
            # torchcodec's CUDA stream set-up runs out of memory on the GB10 when torch has not created its context yet.
            torch.zeros(1, device="cuda")
        for cam in range(len(self.videos)) if cams is None else cams:
            present: Bool[ndarray, "k"] = self.grid_samples[cam, :max_frames] >= 0
            columns: Int64[ndarray, "m"] = np.flatnonzero(present).astype(np.int64)
            decoder: VideoDecoder = VideoDecoder(self.videos[cam].mp4, device=device, seek_mode="exact", dimension_order="NHWC")
            for start in range(0, len(columns), batch_size):
                chunk: Int64[ndarray, "b"] = columns[start : start + batch_size]
                frames: UInt8[Tensor, "b h w 3"] = decoder.get_frames_at(self.grid_samples[cam, chunk].tolist()).data.contiguous()
                yield GridBatch(cam, self.grid_ns[chunk], frames)
            del decoder
