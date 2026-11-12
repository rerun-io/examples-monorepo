"""MoGe-2's depth and normals of every tracked frame, logged off the tracking thread.

DPVO tracks with MoGe-2's metric depth on every frame, so showing it costs no inference, only logging; done on the
tracking thread, that logging is what costs (a blocking device-to-host copy waits for all of DPVO's queued GPU work,
then a PNG encode and an H.264 encode). :class:`PriorLog` takes each frame's depth and normals as asynchronous copies
into page-locked host memory and an event, and a thread of its own waits for the copy, then logs the depth as a
metric millimetre PNG (``rr.EncodedDepthImage`` through the Rust ``depth_png`` codec) and the normals as a frame of an
NVENC H.264 video, at that frame's times.

What stays on the tracking thread is cheap: the copies go into a ring of page-locked buffers allocated once (a fresh
page-locked tensor per frame cost about 0.9 ms of the thread's time), and the logger waits on a blocking event, so it
sleeps instead of spinning on the driver while DPVO launches kernels.
"""

import queue
import threading
from dataclasses import dataclass, field

import numpy as np
import rerun as rr
import torch
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from metrictrail import entities
from metrictrail.rerun_log import FrameVideo, log_depth, set_frame_time

MAX_PENDING: int = 16
"""Frames queued for the logger before ``submit`` waits for it (0.4 s of tracking at 40 fps)."""
RING_SLOTS: int = MAX_PENDING + 2
"""Page-locked buffers per frame shape: a slot is written again only after the frame that used it is logged (at most
:data:`MAX_PENDING` frames queued and one being logged)."""


@dataclass(frozen=True, slots=True)
class PriorFrame:
    """One tracked frame's MoGe-2 outputs on their way to the host."""

    rec: rr.RecordingStream
    """The recording of the pass."""
    frame: int
    """Tracked-frame index (``frame``)."""
    source_frame: int | None
    """Its source frame (``source_frame``, ``video_time``), if known."""
    depth_m: Float32[Tensor, "h w"] | None
    """Metric depth on the host (0 = none), filled by an asynchronous copy."""
    normals: UInt8[Tensor, "h w 3"] | None
    """8-bit normals on the host, filled by an asynchronous copy."""
    ready: torch.cuda.Event | None
    """Recorded after the copies; None for CPU tensors (copied already)."""


@dataclass
class HostRing:
    """Page-locked buffers of one shape and dtype, reused in turn (:data:`RING_SLOTS`)."""

    buffers: list[Tensor] = field(default_factory=list)
    """Allocated on first use, up to :data:`RING_SLOTS`."""
    turn: int = 0
    """Copies made so far."""

    def copy(self, tensor: Tensor) -> Tensor:
        """``tensor`` copied asynchronously into the next buffer."""
        slot: int = self.turn % RING_SLOTS
        self.turn += 1
        if slot == len(self.buffers):
            self.buffers.append(torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True))
        return self.buffers[slot].copy_(tensor, non_blocking=True)


@dataclass
class PriorLog:
    """Logs MoGe-2's depth and normals on its own thread (module docstring); :meth:`drain` before a pass's recording
    closes, :meth:`close` at the end. A logging failure is raised on the tracking thread at the next call."""

    depth_range_m: tuple[float, float]
    """Colour range of the depth images."""
    video_time_ns: Int64[ndarray, "s"] | None
    """``video_time`` of each source frame, or None."""
    fps: float
    """Rate the normals video is encoded at (its times come from the timelines)."""
    pending: "queue.Queue[PriorFrame | None]" = field(default_factory=lambda: queue.Queue(maxsize=MAX_PENDING))
    """Frames waiting for the logger thread; None stops it."""
    rings: dict[tuple[tuple[int, ...], torch.dtype], HostRing] = field(default_factory=dict)
    """The page-locked buffers, per frame shape and dtype (the tracking thread's)."""
    videos: dict[int, FrameVideo] = field(default_factory=dict)
    """The normals video of each pass recording (by ``id``); touched by the logger thread only, or after :meth:`drain`."""
    error: BaseException | None = None
    """The first failure of the logger thread."""
    thread: threading.Thread | None = None
    """The logger thread, started on the first frame."""

    def submit(self, rec: rr.RecordingStream, frame: int, source_frame: int | None, depth_m: Float32[Tensor, "h w"] | None, normals: UInt8[Tensor, "h w 3"] | None) -> None:
        """Queue one frame's depth (0 = none) and normals; copies them without waiting for the GPU."""
        self.raise_error()
        if depth_m is None and normals is None:
            return
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, name="metrictrail-priors", daemon=True)
            self.thread.start()
        on_gpu: bool = any(tensor is not None and tensor.is_cuda for tensor in (depth_m, normals))
        host_depth: Tensor | None = self.to_host(depth_m) if depth_m is not None else None
        host_normals: Tensor | None = self.to_host(normals) if normals is not None else None
        ready: torch.cuda.Event | None = None
        if on_gpu:
            ready = torch.cuda.Event(blocking=True)
            ready.record()
        self.pending.put(PriorFrame(rec, frame, source_frame, host_depth, host_normals, ready))

    def to_host(self, tensor: Tensor) -> Tensor:
        """An asynchronous copy into the ring of ``tensor``'s shape for a CUDA tensor; a plain copy otherwise."""
        if not tensor.is_cuda:
            return tensor.detach().clone()
        key: tuple[tuple[int, ...], torch.dtype] = (tuple(tensor.shape), tensor.dtype)
        ring: HostRing | None = self.rings.get(key)
        if ring is None:
            ring = self.rings[key] = HostRing()
        return ring.copy(tensor)

    def run(self) -> None:
        """The logger thread: each frame at its own times, in order."""
        while True:
            item: PriorFrame | None = self.pending.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    self.log(item)
            except BaseException as error:  # kept, and raised on the tracking thread as it is (beartype's included)
                self.error = error
            finally:
                self.pending.task_done()

    def log(self, item: PriorFrame) -> None:
        """Log one frame (the logger thread's own time context)."""
        if item.ready is not None:
            item.ready.synchronize()
        set_frame_time(item.rec, item.frame, item.source_frame, self.video_time_ns)
        if item.depth_m is not None:
            log_depth(item.rec, item.depth_m.numpy(), self.depth_range_m)
        if item.normals is not None:
            video: FrameVideo | None = self.videos.get(id(item.rec))
            if video is None:
                video = self.videos[id(item.rec)] = FrameVideo(item.rec, entities.CAM_NORMALS, self.fps)
            normals_rgb: UInt8[ndarray, "h w 3"] = np.asarray(item.normals.numpy())
            video.log(normals_rgb)

    def drain(self) -> None:
        """Wait until every queued frame is logged, then flush the normals videos (a pass ends)."""
        self.pending.join()
        for video in self.videos.values():
            video.close()
        self.videos = {}
        self.raise_error()

    def close(self) -> None:
        """:meth:`drain`, then stop the logger thread; it is stopped and joined even when draining raises."""
        try:
            self.drain()
        finally:
            if self.thread is not None:
                self.pending.put(None)
                self.thread.join()
                self.thread = None

    def raise_error(self) -> None:
        """Raise the logger thread's failure, if any (once)."""
        if self.error is not None:
            error, self.error = self.error, None
            raise error
