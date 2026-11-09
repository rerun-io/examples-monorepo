"""HTTP backend of the hand labeler (stdlib ``ThreadingHTTPServer``, JSON via pyserde).

API (all JSON bodies are the pyserde records of ``posekit.labeler.schema`` and ``posekit.labeler.sheet``):

- ``GET  /`` -> ``static/index.html``; ``GET /static/<file>``
- ``GET  /api/config`` -> ``ServerConfig`` (default reviewer, dataset)
- ``GET  /api/segments`` -> ``list[SegmentSummary]``
- ``GET  /api/segment/<sid>`` -> ``SegmentInfo``
- ``GET  /api/frame/<sid>/<cam>/<t_ns>.jpg?max_side=960`` -> JPEG of the native frame, downscaled
- ``GET  /api/frameset/<sid>/<t_ns>`` -> ``FrameState``
- ``GET  /api/sheet/<sid>/<t_ns>.jpg`` -> 3x2 contact sheet; ``GET /api/sheet/<sid>/<t_ns>.json`` -> ``SheetInfo``
- ``POST /api/labels`` body ``FrameLabels`` -> ``{"stored": n}``
- ``POST /api/range`` body ``RangeRequest`` -> ``RangeResult``
- ``GET  /api/queue?sid=<sid>&limit=50`` -> ``list[QueueItem]``
- ``POST /api/commit/<sid>`` -> ``{"path": ..., "n_labels": n}`` (writes and registers ``hand_boxes``)

MEgATrack loop (``posekit.labeler.tracker``, ``posekit.labeler.actions``):

- ``GET  /api/track/<sid>?t0=<t_ns>&n=<count>`` -> ``list[TrackFrame]``, ``count`` grid times from ``t0`` (default: the
  first grid time, 10)
- ``POST /api/accept_range`` body ``AcceptRange`` -> ``AcceptResult``: the machine box of every (t, cam, side) in
  [t0, t1] without a label becomes a ``tracker`` label (``accepted``, or ``no_hand`` where the machine has none)
- ``GET  /api/playback/<sid>/<t0_ns>.jpg?n=10&stride=1`` -> filmstrip, ``n`` rows (every ``stride``-th grid time) x 6 cameras with the machine boxes;
  ``.json`` -> ``PlaybackInfo``
- ``GET  /api/actions/<sid>`` -> ``list[ActionEvent]``; ``GET /api/effort/<sid>`` -> ``EffortSummary``

With ``open_carrier`` (``--visual-tracker``), a visual tracker carries reviewed boxes through EgoHand misses (origin
``tracked``, ``posekit.labeler.carry``); without it the machine track is unchanged. With ``open_propagator``
(``--propagator megatrack``) the machine track comes from that propagator instead (``posekit.labeler.megatrack``).

``POST /api/labels`` also logs one ``ActionEvent`` per posted label that differs from the machine track (see
``posekit.labeler.actions``). Every label write is serialised per segment and forgets the cached machine track from
its time on. A playback request decodes its cells on six threads (one per camera) and then schedules the next window.

Bad input answers 400 with ``{"error": message}``; an unknown segment, camera or file answers 404.

Segments open lazily (frames + labels from the catalog) on first use. Decoded frames sit in a small cache;
after a queue request a background worker decodes the first queue items, so the next frame sets are warm.
Every ``autosave_every`` new labels the store is written to its autosave file (never registered). ``close``
refuses new writes (503), waits for the writes in flight and autosaves every store with unsaved labels.
"""

from __future__ import annotations

import mimetypes
import re
import threading
import time
import traceback
from _thread import LockType
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable
from urllib.parse import parse_qs, urlparse

import numpy as np
from beartype.roar import BeartypeException
from jaxtyping import UInt8
from numpy import ndarray
from serde import SerdeError, serde
from serde.json import from_json, to_json

from posekit.labeler.actions import classify_labels, effort_summary
from posekit.labeler.queue import ProposalProvider, ReviewQueue
from posekit.labeler.schema import (
    AcceptRange,
    AcceptResult,
    ActionEvent,
    CameraInfo,
    EffortSummary,
    FrameLabels,
    FrameState,
    Proposal,
    QueueItem,
    RangeRequest,
    RangeResult,
    SegmentInfo,
    SegmentSplit,
    SegmentSummary,
    TrackFrame,
    ViewLabel,
    ViewState,
)
from posekit.labeler.sheet import PlaybackCell, PlaybackInfo, SheetInfo, contact_sheet, encode_jpeg, filmstrip, playback_info, sheet_info, strip_cell
from posekit.labeler.store import LabelRow, LabelStore, NewWrite
from posekit.labeler.tracker import BoxCarrier, Propagator, TrackCache


@runtime_checkable
class FrameProvider(Protocol):
    """Frames of one segment; ``posekit.labeler.frames.FrameSource`` satisfies it."""

    @property
    def cams(self) -> list[CameraInfo]:
        """Cameras in rig order."""
        ...

    @property
    def times_ns(self) -> list[int]:
        """The 5 fps label grid on ``video_time``, ns, ascending."""
        ...

    @property
    def keyframes_ns(self) -> list[int]:
        """The 1 fps keyframes plus the last grid time, a subset of ``times_ns``."""
        ...

    def frame(self, cam: int, t_ns: int) -> UInt8[ndarray, "h w 3"]:
        """Native RGB frame of camera ``cam`` at grid time ``t_ns``; thread-safe; KeyError when there is none."""
        ...


@runtime_checkable
class LayerBackend(Protocol):
    """Where a segment's labels load from and commit to (``posekit.labeler.store.CatalogLayers``)."""

    def open_store(self, segment_id: str, split: SegmentSplit) -> LabelStore:
        """A store filled with the segment's saved labels."""
        ...

    def layer_path(self, segment_id: str) -> Path:
        """Committed layer file."""
        ...

    def autosave_path(self, segment_id: str) -> Path:
        """Autosave file (never registered)."""
        ...

    def register(self, path: Path) -> None:
        """Register a committed layer file as ``hand_boxes`` (REPLACE)."""
        ...


class NotFoundError(LookupError):
    """An unknown segment, camera, time or file; answered with 404."""


class ShuttingDownError(RuntimeError):
    """A write that arrived after ``LabelApp.close`` started; answered with 503."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class StoredResponse:
    """Answer of ``POST /api/labels``."""

    stored: int
    """Labels stored."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CommitResponse:
    """Answer of ``POST /api/commit/<sid>``."""

    path: str
    """Layer file written and registered."""
    n_labels: int
    """Labels in it."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Answer of ``GET /api/config``."""

    reviewer: str
    """Default reviewer name for the browser."""
    dataset: str
    """Catalog dataset."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ErrorResponse:
    """Body of a 4xx/5xx answer."""

    error: str
    """What went wrong."""


@dataclass(slots=True)
class Segment:
    """One opened segment."""

    info: SegmentInfo
    """Layout sent to the browser."""
    frames: FrameProvider
    """Frame decoder."""
    store: LabelStore
    """Labels."""
    queue: ReviewQueue
    """Review queue and range fill."""
    tracker: TrackCache
    """Machine track."""
    period_ns: int
    """Grid period, ns."""
    save_lock: LockType
    """Serialises autosave and commit writes."""
    label_lock: LockType
    """Serialises label writes (each write and the track invalidation after it are one step)."""
    autosave_pending: bool = False
    """An autosave is scheduled and has not run yet."""


FrameKey: TypeAlias = tuple[str, int, int]
"""(segment id, cam, t_ns)."""
CellKey: TypeAlias = tuple[int, int]
"""(cam, t_ns) of one filmstrip cell."""
MAX_PLAYBACK_ROWS: int = 50
"""Most rows one filmstrip may have."""
MAX_PLAYBACK_STRIDE: int = 10
"""Most grid steps between two filmstrip rows."""


class LabelApp:
    """The labeler's state and operations, independent of HTTP."""

    def __init__(
        self,
        *,
        segment_ids: list[str],
        open_frames: Callable[[str], FrameProvider],
        proposals: ProposalProvider,
        layers: LayerBackend,
        viewer_url: Callable[[str], str],
        eval_segments: frozenset[str],
        static_dir: Path,
        reviewer: str = "pablo",
        dataset: str = "robocap-labeling",
        autosave_every: int = 50,
        cached_frames: int = 72,
        prefetch_items: int = 2,
        cached_cells: int = 720,
        open_carrier: Callable[[str], BoxCarrier] | None = None,
        open_propagator: Callable[[str], Propagator] | None = None,
    ) -> None:
        """Bind the labeler to its providers.

        Args:
            segment_ids: Segments offered, in list order.
            open_frames: Opens the frame provider of one segment (called once per segment, lazily).
            proposals: Proposal source for every segment.
            layers: Label persistence.
            viewer_url: Segment id -> URL the embedded web viewer opens.
            eval_segments: Held-out segments; all others are train.
            static_dir: Directory with ``index.html`` and the frontend files.
            reviewer: Default reviewer name for the browser.
            dataset: Catalog dataset name, reported by ``/api/config``.
            autosave_every: New labels between autosaves.
            cached_frames: Decoded frames kept in memory.
            prefetch_items: Queue items whose frames are decoded ahead after a queue request.
            cached_cells: Downscaled filmstrip cells kept in memory (six windows of ten rows by default).
            open_carrier: Opens the visual tracker of one segment (called after ``open_frames``), which carries reviewed
                boxes in the machine track (origin ``tracked``); None = chains only hold.
            open_propagator: Opens the machine propagator of one segment (called after ``open_frames``); None = the chain
                tracker (``posekit.labeler.tracker.ChainPropagator`` with ``open_carrier``'s carrier).
        """
        self.segment_ids: list[str] = segment_ids
        self.open_frames: Callable[[str], FrameProvider] = open_frames
        self.proposals: ProposalProvider = proposals
        self.layers: LayerBackend = layers
        self.viewer_url: Callable[[str], str] = viewer_url
        self.eval_segments: frozenset[str] = eval_segments
        self.static_dir: Path = static_dir
        self.config: ServerConfig = ServerConfig(reviewer, dataset)
        self.autosave_every: int = autosave_every
        self.cached_frames: int = cached_frames
        self.prefetch_items: int = prefetch_items
        self.open_carrier: Callable[[str], BoxCarrier] | None = open_carrier
        self.open_propagator: Callable[[str], Propagator] | None = open_propagator
        self._segments: dict[str, Segment] = {}
        self._open_locks: dict[str, LockType] = {sid: threading.Lock() for sid in segment_ids}
        self._frames: OrderedDict[FrameKey, Future[UInt8[ndarray, "h w 3"]]] = OrderedDict()
        self._frames_lock = threading.Lock()
        self._prefetcher: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="labeler-prefetch")
        self._writer: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="labeler-autosave")
        self.cached_cells: int = cached_cells
        self._cells: OrderedDict[FrameKey, Future[UInt8[ndarray, "ch cw 3"]]] = OrderedDict()
        self._cells_lock = threading.Lock()
        self._cell_decoder: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="labeler-cells")
        self._warmer: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="labeler-track")
        self._closing: bool = False
        self._writes_in_flight: int = 0
        self._writes_changed = threading.Condition()

    def close(self) -> None:
        """Refuse new writes, wait for those in flight, autosave every store with unsaved labels, stop the workers."""
        with self._writes_changed:
            if self._closing:
                return
            self._closing = True
            self._writes_changed.wait_for(lambda: self._writes_in_flight == 0)
        self._prefetcher.shutdown(wait=False, cancel_futures=True)
        self._cell_decoder.shutdown(wait=False, cancel_futures=True)
        self._warmer.shutdown(wait=False, cancel_futures=True)
        dirty: list[Segment] = [segment for segment in list(self._segments.values()) if segment.store.unsaved]
        print(f"[labeler] shutting down: saving {len(dirty)} segment(s) with unsaved labels")
        saves: list[Future[None]] = [self._writer.submit(self._autosave, segment) for segment in dirty]
        self._writer.shutdown(wait=True)
        for save in saves:
            save.result()

    @contextmanager
    def _write(self) -> Iterator[None]:
        """Run one label write; refused once ``close`` has started, and ``close`` waits for it to finish."""
        with self._writes_changed:
            if self._closing:
                raise ShuttingDownError("the labeler is shutting down; nothing was stored")
            self._writes_in_flight += 1
        try:
            yield
        finally:
            with self._writes_changed:
                self._writes_in_flight -= 1
                self._writes_changed.notify_all()

    # -- segments -------------------------------------------------------------------------------------------

    def split(self, segment_id: str) -> SegmentSplit:
        """train or eval."""
        return "eval" if segment_id in self.eval_segments else "train"

    def segment(self, segment_id: str) -> Segment:
        """The opened segment, opening it on first use."""
        lock: LockType | None = self._open_locks.get(segment_id)
        if lock is None:
            raise NotFoundError(f"unknown segment {segment_id!r}")
        with lock:
            opened: Segment | None = self._segments.get(segment_id)
            if opened is not None:
                return opened
            frames: FrameProvider = self.open_frames(segment_id)
            split: SegmentSplit = self.split(segment_id)
            info: SegmentInfo = SegmentInfo(
                segment_id,
                split,
                list(frames.cams),
                [int(t) for t in frames.times_ns],
                [int(t) for t in frames.keyframes_ns],
                self.viewer_url(segment_id),
            )
            store: LabelStore = self.layers.open_store(segment_id, split)
            queue: ReviewQueue = ReviewQueue(
                segment_id,
                times_ns=info.times_ns,
                keyframes_ns=info.keyframes_ns,
                cams=[camera.cam for camera in info.cams],
                store=store,
                proposals=self.proposals,
            )
            cams: list[int] = [camera.cam for camera in info.cams]
            carrier: BoxCarrier | None = None if self.open_carrier is None else self.open_carrier(segment_id)
            propagator: Propagator | None = None if self.open_propagator is None else self.open_propagator(segment_id)
            tracker: TrackCache = TrackCache(
                segment_id, times_ns=info.times_ns, cams=cams, proposals=self.proposals, store=store, carrier=carrier, propagator=propagator
            )
            period_ns: int = int(np.median(np.diff(info.times_ns))) if len(info.times_ns) > 1 else 200_000_000
            opened = Segment(info, frames, store, queue, tracker, period_ns, threading.Lock(), threading.Lock())
            self._segments[segment_id] = opened
            if info.times_ns:
                self._warmer.submit(self._warm_track, opened)
            print(f"[labeler] opened {segment_id}: {len(info.times_ns)} grid times, {len(store)} stored labels, {len(store.actions())} actions")
            return opened

    def _warm_track(self, segment: Segment) -> None:
        """Compute the whole machine track once in the background, so the first track requests are cache hits."""
        try:
            started: float = time.perf_counter()
            segment.tracker.frames(segment.info.times_ns[0], len(segment.info.times_ns))
            print(f"[labeler] machine track of {segment.info.segment_id}: {time.perf_counter() - started:.2f} s")
        except BeartypeException:
            raise
        except Exception as error:  # the request that needs the track recomputes it and reports the error
            print(f"[labeler] warming the track of {segment.info.segment_id} failed: {error!r}")

    def summaries(self) -> list[SegmentSummary]:
        """One row per segment (opens every segment)."""
        rows: list[SegmentSummary] = []
        for segment_id in self.segment_ids:
            segment: Segment = self.segment(segment_id)
            times: list[int] = segment.info.times_ns
            duration_s: float = (times[-1] - times[0]) / 1e9 if times else 0.0
            rows.append(
                SegmentSummary(
                    segment_id,
                    segment.info.split,
                    duration_s,
                    len(segment.info.keyframes_ns),
                    segment.queue.n_reviewed_keyframes(),
                    len(segment.store),
                )
            )
        return rows

    def _grid_time(self, segment: Segment, t_ns: int) -> int:
        if not segment.queue.on_grid(t_ns):
            raise ValueError(f"t_ns {t_ns} is not on the 5 fps grid of {segment.info.segment_id}")
        return t_ns

    def _camera(self, segment: Segment, cam: int) -> CameraInfo:
        for camera in segment.info.cams:
            if camera.cam == cam:
                return camera
        raise NotFoundError(f"segment {segment.info.segment_id} has no camera {cam}")

    # -- frames ---------------------------------------------------------------------------------------------

    def frame(self, segment_id: str, cam: int, t_ns: int) -> UInt8[ndarray, "h w 3"]:
        """Decoded RGB frame, from the cache or decoded now (waits for a prefetch already decoding it)."""
        segment: Segment = self.segment(segment_id)
        self._camera(segment, cam)
        self._grid_time(segment, t_ns)
        key: FrameKey = (segment_id, cam, t_ns)
        owner: bool = False
        with self._frames_lock:
            future: Future[UInt8[ndarray, "h w 3"]] | None = self._frames.get(key)
            if future is None:
                future = Future()
                self._frames[key] = future
                owner = True
                while len(self._frames) > self.cached_frames:
                    self._frames.popitem(last=False)
            else:
                self._frames.move_to_end(key)
        if owner:
            try:
                future.set_result(segment.frames.frame(cam, t_ns))
            except BaseException as error:
                with self._frames_lock:
                    self._frames.pop(key, None)
                future.set_exception(NotFoundError(f"no frame: {error}") if isinstance(error, KeyError) else error)
        return future.result()

    def prefetch(self, segment_id: str, times_ns: list[int]) -> None:
        """Decode every camera of these times in the background."""
        segment: Segment = self.segment(segment_id)
        for t_ns in times_ns:
            for camera in segment.info.cams:
                self._prefetcher.submit(self._prefetch_one, segment_id, camera.cam, t_ns)

    def _prefetch_one(self, segment_id: str, cam: int, t_ns: int) -> None:
        try:
            self.frame(segment_id, cam, t_ns)
        except BeartypeException:
            raise
        except Exception as error:  # a failed prefetch is retried (and reported) by the request that needs it
            print(f"[labeler] prefetch {segment_id} cam {cam} t {t_ns} failed: {error!r}")

    def frame_jpeg(self, segment_id: str, cam: int, t_ns: int, max_side: int | None) -> bytes:
        """JPEG of one frame, longer side at most ``max_side``."""
        if max_side is not None and max_side < 16:
            raise ValueError("max_side must be at least 16")
        return encode_jpeg(self.frame(segment_id, cam, t_ns), max_side)

    # -- frame sets -----------------------------------------------------------------------------------------

    def frame_state(self, segment_id: str, t_ns: int) -> FrameState:
        """Proposals, labels and flags of one time."""
        segment: Segment = self.segment(segment_id)
        self._grid_time(segment, t_ns)
        proposals: list[Proposal] = self.proposals.proposals(segment_id, t_ns)
        labels: list[ViewLabel] = segment.store.get(t_ns)
        views: list[ViewState] = [
            ViewState(camera.cam, [p for p in proposals if p.cam == camera.cam], [label for label in labels if label.cam == camera.cam])
            for camera in segment.info.cams
        ]
        return FrameState(segment_id, t_ns, views, segment.queue.flags(t_ns))

    def sheet_info(self, segment_id: str, t_ns: int) -> SheetInfo:
        """Layout and numbering of the contact sheet and the machine track at ``t_ns`` (no decoding)."""
        segment: Segment = self.segment(segment_id)
        self._grid_time(segment, t_ns)
        return sheet_info(segment_id, t_ns, segment.info.cams, self.proposals.proposals(segment_id, t_ns), segment.tracker.frame(t_ns).boxes)

    def sheet_jpeg(self, segment_id: str, t_ns: int) -> bytes:
        """The rendered contact sheet."""
        segment: Segment = self.segment(segment_id)
        self._grid_time(segment, t_ns)
        images: list[UInt8[ndarray, "h w 3"]] = [self._frame_or_black(segment_id, camera, t_ns) for camera in segment.info.cams]
        return contact_sheet(
            segment_id,
            t_ns,
            segment.info.cams,
            images,
            self.proposals.proposals(segment_id, t_ns),
            segment.store.get(t_ns),
            segment.tracker.frame(t_ns).boxes,
        ).jpeg

    def _frame_or_black(self, segment_id: str, camera: CameraInfo, t_ns: int) -> UInt8[ndarray, "h w 3"]:
        """The frame, or a black image when this camera has none at ``t_ns`` (cameras start and stop a few frames apart).

        A reviewer sees the black cell and marks that camera unjudgeable (``rejected``).
        """
        try:
            return self.frame(segment_id, camera.cam, t_ns)
        except NotFoundError:
            return np.zeros((camera.height, camera.width, 3), dtype=np.uint8)

    # -- labels ---------------------------------------------------------------------------------------------

    def put_labels(self, frame: FrameLabels) -> int:
        """Store a reviewer's decisions and log what they changed against the machine track; autosaves every ``autosave_every`` writes."""
        segment: Segment = self.segment(frame.segment_id)
        self._grid_time(segment, frame.t_ns)
        for label in frame.labels:
            self._camera(segment, label.cam)
        with self._write(), segment.label_lock:
            machine: TrackFrame = segment.tracker.frame(frame.t_ns)
            events: list[ActionEvent] = classify_labels(
                frame, machine, self.proposals.proposals(frame.segment_id, frame.t_ns), segment.store.get(frame.t_ns), time.time_ns()
            )
            stored: int = segment.store.put(frame)
            segment.tracker.invalidate_from(frame.t_ns)
            segment.store.log_actions(events)
            self._maybe_autosave(segment)
        return stored

    def accept_range(self, request: AcceptRange) -> AcceptResult:
        """Accept the machine track of every grid time in [t0_ns, t1_ns]: labels where there are none, reviewer labels kept."""
        segment: Segment = self.segment(request.segment_id)
        first: int = segment.tracker.index(request.t0_ns)
        last: int = segment.tracker.index(request.t1_ns)
        if last < first:
            raise ValueError("t0_ns must not be after t1_ns")
        with self._write(), segment.label_lock:
            frames: list[TrackFrame] = segment.tracker.frames(request.t0_ns, last - first + 1)
            rows: list[LabelRow] = [
                LabelRow(frame.t_ns, ViewLabel(box.cam, box.side, "no_hand" if box.box is None else "accepted", box.box, "tracker", request.reviewer))
                for frame in frames
                for box in frame.boxes
                if box.origin != "reviewed"
            ]
            reviewed: int = sum(1 for frame in frames for box in frame.boxes if box.origin == "reviewed")
            written: NewWrite = segment.store.put_new(rows)
            segment.tracker.invalidate_from(request.t0_ns)
            segment.store.log_actions(
                [
                    ActionEvent(
                        time.time_ns(),
                        request.segment_id,
                        request.t0_ns,
                        request.t1_ns,
                        None,
                        None,
                        "accept_range",
                        request.reviewer,
                        written.written,
                    )
                ]
            )
            self._maybe_autosave(segment)
        return AcceptResult(written.written, written.kept + reviewed)

    def track(self, segment_id: str, t0_ns: int | None, n: int) -> list[TrackFrame]:
        """The machine track of ``n`` grid times from ``t0_ns`` (the first grid time when None)."""
        segment: Segment = self.segment(segment_id)
        if not segment.info.times_ns:
            return []
        return segment.tracker.frames(segment.info.times_ns[0] if t0_ns is None else t0_ns, n)

    def actions(self, segment_id: str) -> list[ActionEvent]:
        """The segment's action log, in logging order."""
        return self.segment(segment_id).store.actions()

    def effort(self, segment_id: str) -> EffortSummary:
        """Effort summary of the segment."""
        segment: Segment = self.segment(segment_id)
        return effort_summary(segment.store.actions(), segment.store.rows(), segment.period_ns)

    # -- playback -------------------------------------------------------------------------------------------

    def _playback_layout(self, segment: Segment, t0_ns: int, n: int, stride: int = 1) -> PlaybackInfo:
        if not 1 <= n <= MAX_PLAYBACK_ROWS:
            raise ValueError(f"n must be in [1, {MAX_PLAYBACK_ROWS}]")
        if not 1 <= stride <= MAX_PLAYBACK_STRIDE:
            raise ValueError(f"stride must be in [1, {MAX_PLAYBACK_STRIDE}]")
        start: int = segment.tracker.index(t0_ns)
        return playback_info(segment.info.segment_id, segment.info.cams, segment.info.times_ns[start : start + n * stride : stride])

    def playback_info(self, segment_id: str, t0_ns: int, n: int, stride: int = 1) -> PlaybackInfo:
        """Layout of the filmstrip of ``n`` grid times from ``t0_ns``, every ``stride``-th grid time (no decoding)."""
        return self._playback_layout(self.segment(segment_id), t0_ns, n, stride)

    def playback_jpeg(self, segment_id: str, t0_ns: int, n: int, stride: int = 1) -> bytes:
        """The filmstrip with the machine boxes; the next window's cells are decoded in the background.

        ``stride`` > 1 shows every ``stride``-th grid time, so one look covers a longer stretch of video.
        """
        segment: Segment = self.segment(segment_id)
        info: PlaybackInfo = self._playback_layout(segment, t0_ns, n, stride)
        cells: dict[CellKey, Future[UInt8[ndarray, "ch cw 3"]]] = self._strip_cells(segment, info)
        after: int = segment.tracker.index(info.rows[-1].t_ns) + 1
        if after < len(segment.info.times_ns):
            self._strip_cells(segment, self._playback_layout(segment, segment.info.times_ns[after], n, stride))
        track: list[TrackFrame] = segment.tracker.frames(t0_ns, (len(info.rows) - 1) * stride + 1)[::stride]
        images: list[list[UInt8[ndarray, "ch cw 3"]]] = [[cells[(cell.cam, row.t_ns)].result() for cell in info.cells] for row in info.rows]
        return filmstrip(info, images, track, segment.info.times_ns[0])

    def _strip_cells(self, segment: Segment, info: PlaybackInfo) -> dict[CellKey, Future[UInt8[ndarray, "ch cw 3"]]]:
        """Futures of every cell of a filmstrip; missing cells are decoded on the cell pool, one task per camera in time order."""
        futures: dict[CellKey, Future[UInt8[ndarray, "ch cw 3"]]] = {}
        todo: dict[int, list[int]] = {}
        with self._cells_lock:
            for cell in info.cells:
                for row in info.rows:
                    key: FrameKey = (info.segment_id, cell.cam, row.t_ns)
                    future: Future[UInt8[ndarray, "ch cw 3"]] | None = self._cells.get(key)
                    if future is None:
                        future = Future()
                        self._cells[key] = future
                        todo.setdefault(cell.cam, []).append(row.t_ns)
                    else:
                        self._cells.move_to_end(key)
                    futures[(cell.cam, row.t_ns)] = future
            while len(self._cells) > self.cached_cells:
                self._cells.popitem(last=False)
        for cell in info.cells:
            if cell.cam in todo:
                self._cell_decoder.submit(self._decode_cells, segment, cell, {t_ns: futures[(cell.cam, t_ns)] for t_ns in todo[cell.cam]})
        return futures

    def _decode_cells(self, segment: Segment, cell: PlaybackCell, futures: dict[int, Future[UInt8[ndarray, "ch cw 3"]]]) -> None:
        """Decode one camera's cells in time order; a camera without a frame at a time gets a black cell."""
        for t_ns, future in futures.items():
            try:
                image: UInt8[ndarray, "ch cw 3"] = strip_cell(segment.frames.frame(cell.cam, t_ns), cell)
            except KeyError:
                image = np.zeros((cell.h, cell.w, 3), dtype=np.uint8)
            except BaseException as error:  # handed to the request that waits on this cell (BeartypeException included)
                with self._cells_lock:
                    self._cells.pop((segment.info.segment_id, cell.cam, t_ns), None)
                future.set_exception(error)
                continue
            future.set_result(image)

    def _maybe_autosave(self, segment: Segment) -> None:
        with segment.save_lock:
            if segment.autosave_pending or segment.store.unsaved < self.autosave_every:
                return
            segment.autosave_pending = True
        self._writer.submit(self._autosave, segment)

    def _autosave(self, segment: Segment) -> None:
        with segment.save_lock:
            segment.autosave_pending = False
            if segment.store.unsaved == 0:  # a commit saved everything since this autosave was scheduled
                return
            path: Path = self.layers.autosave_path(segment.info.segment_id)
            count: int = segment.store.save_layer(path, self.effort(segment.info.segment_id))
        print(f"[labeler] autosaved {count} labels of {segment.info.segment_id} to {path}")

    def fill_range(self, request: RangeRequest) -> RangeResult:
        """Range fill between two reviewed keyframes."""
        segment: Segment = self.segment(request.segment_id)
        with self._write(), segment.label_lock:
            result: RangeResult = segment.queue.fill_range(request)
            segment.tracker.invalidate_from(request.t0_ns)
            self._maybe_autosave(segment)
        return result

    def queue(self, segment_id: str, limit: int) -> list[QueueItem]:
        """Frame sets to review next; the first ``prefetch_items`` are decoded in the background."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        items: list[QueueItem] = self.segment(segment_id).queue.items(limit)
        self.prefetch(segment_id, [item.t_ns for item in items[: self.prefetch_items]])
        return items

    def commit(self, segment_id: str) -> CommitResponse:
        """Write the ``hand_boxes`` layer file and register it on the catalog (REPLACE)."""
        segment: Segment = self.segment(segment_id)
        with self._write(), segment.save_lock:
            path: Path = self.layers.layer_path(segment_id)
            count: int = segment.store.save_layer(path, self.effort(segment_id))
            self.layers.register(path)
            self.layers.autosave_path(segment_id).unlink(missing_ok=True)
        print(f"[labeler] committed {count} labels of {segment_id} to {path}")
        return CommitResponse(str(path), count)


ROUTE_FRAME: re.Pattern[str] = re.compile(r"^/api/frame/([^/]+)/(-?\d+)/(-?\d+)\.jpg$")
ROUTE_FRAMESET: re.Pattern[str] = re.compile(r"^/api/frameset/([^/]+)/(-?\d+)$")
ROUTE_SHEET: re.Pattern[str] = re.compile(r"^/api/sheet/([^/]+)/(-?\d+)\.(jpg|json)$")
ROUTE_SEGMENT: re.Pattern[str] = re.compile(r"^/api/segment/([^/]+)$")
ROUTE_COMMIT: re.Pattern[str] = re.compile(r"^/api/commit/([^/]+)$")
ROUTE_TRACK: re.Pattern[str] = re.compile(r"^/api/track/([^/]+)$")
ROUTE_PLAYBACK: re.Pattern[str] = re.compile(r"^/api/playback/([^/]+)/(-?\d+)\.(jpg|json)$")
ROUTE_ACTIONS: re.Pattern[str] = re.compile(r"^/api/actions/([^/]+)$")
ROUTE_EFFORT: re.Pattern[str] = re.compile(r"^/api/effort/([^/]+)$")
MAX_BODY: int = 4 * 1024 * 1024
"""Largest accepted request body, bytes."""


def make_handler(app: LabelApp) -> type[BaseHTTPRequestHandler]:
    """A request handler class bound to ``app``."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            if "/api/frame/" not in self.path:
                super().log_message(format, *args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: object, status: int = 200) -> None:
            self._send(status, to_json(payload).encode(), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json(ErrorResponse(message), status)

        def _body(self) -> str:
            length: int = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY:
                raise ValueError(f"body of {length} bytes is larger than {MAX_BODY}")
            return self.rfile.read(length).decode()

        def _dispatch(self, method: str) -> None:
            try:
                if method == "GET":
                    self._get()
                else:
                    self._post()
            except NotFoundError as error:
                self._error(404, str(error))
            except ShuttingDownError as error:
                self._error(503, str(error))
            except BeartypeException as error:
                self._error(500, f"type violation: {error}")
                raise
            except (ValueError, SerdeError) as error:
                self._error(400, str(error))
            except Exception as error:
                traceback.print_exc()
                self._error(500, f"{type(error).__name__}: {error}")

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def _get(self) -> None:
            url = urlparse(self.path)
            path: str = url.path
            query: dict[str, list[str]] = parse_qs(url.query)
            if path in ("/", "/index.html"):
                self._static("index.html")
            elif path.startswith("/static/"):
                self._static(path.removeprefix("/static/"))
            elif path == "/api/config":
                self._json(app.config)
            elif path == "/api/segments":
                self._json(app.summaries())
            elif path == "/api/queue":
                if "sid" not in query:
                    raise ValueError("missing query parameter sid")
                self._json(app.queue(query["sid"][0], int(query.get("limit", ["50"])[0])))
            elif match := ROUTE_SEGMENT.match(path):
                self._json(app.segment(match.group(1)).info)
            elif match := ROUTE_FRAME.match(path):
                max_side: int | None = int(query["max_side"][0]) if "max_side" in query else None
                self._send(200, app.frame_jpeg(match.group(1), int(match.group(2)), int(match.group(3)), max_side), "image/jpeg")
            elif match := ROUTE_FRAMESET.match(path):
                self._json(app.frame_state(match.group(1), int(match.group(2))))
            elif match := ROUTE_SHEET.match(path):
                if match.group(3) == "json":
                    self._json(app.sheet_info(match.group(1), int(match.group(2))))
                else:
                    self._send(200, app.sheet_jpeg(match.group(1), int(match.group(2))), "image/jpeg")
            elif match := ROUTE_TRACK.match(path):
                t0_ns: int | None = int(query["t0"][0]) if "t0" in query else None
                self._json(app.track(match.group(1), t0_ns, int(query.get("n", ["10"])[0])))
            elif match := ROUTE_PLAYBACK.match(path):
                rows: int = int(query.get("n", ["10"])[0])
                stride: int = int(query.get("stride", ["1"])[0])
                if match.group(3) == "json":
                    self._json(app.playback_info(match.group(1), int(match.group(2)), rows, stride))
                else:
                    self._send(200, app.playback_jpeg(match.group(1), int(match.group(2)), rows, stride), "image/jpeg")
            elif match := ROUTE_ACTIONS.match(path):
                self._json(app.actions(match.group(1)))
            elif match := ROUTE_EFFORT.match(path):
                self._json(app.effort(match.group(1)))
            else:
                raise NotFoundError(f"no route GET {path}")

        def _post(self) -> None:
            path: str = urlparse(self.path).path
            body: str = self._body()
            if path == "/api/labels":
                self._json(StoredResponse(app.put_labels(from_json(FrameLabels, body))))
            elif path == "/api/range":
                self._json(app.fill_range(from_json(RangeRequest, body)))
            elif path == "/api/accept_range":
                self._json(app.accept_range(from_json(AcceptRange, body)))
            elif match := ROUTE_COMMIT.match(path):
                self._json(app.commit(match.group(1)))
            else:
                raise NotFoundError(f"no route POST {path}")

        def _static(self, name: str) -> None:
            root: Path = app.static_dir.resolve()
            target: Path = (root / name).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise NotFoundError(f"no static file {name!r}")
            content_type: str = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix == ".js":
                content_type = "text/javascript"
            self._send(200, target.read_bytes(), content_type)

    return Handler


def make_server(app: LabelApp, host: str, port: int) -> ThreadingHTTPServer:
    """A threading HTTP server for ``app`` (port 0 picks a free port); call ``serve_forever`` on it."""
    server: ThreadingHTTPServer = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    return server
