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

Bad input answers 400 with ``{"error": message}``; an unknown segment, camera or file answers 404.

Segments open lazily (frames + labels from the catalog) on first use. Decoded frames sit in a small cache;
after a queue request a background worker decodes the first queue items, so the next frame sets are warm.
Every ``autosave_every`` new labels the store is written to its autosave file (never registered).
"""

from __future__ import annotations

import mimetypes
import re
import threading
import traceback
from _thread import LockType
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable
from urllib.parse import parse_qs, urlparse

from beartype.roar import BeartypeException
from jaxtyping import UInt8
from numpy import ndarray
from serde import SerdeError, serde
from serde.json import from_json, to_json

from posekit.labeler.queue import ProposalProvider, ReviewQueue
from posekit.labeler.schema import (
    CameraInfo,
    FrameLabels,
    FrameState,
    Proposal,
    QueueItem,
    RangeRequest,
    RangeResult,
    SegmentInfo,
    SegmentSplit,
    SegmentSummary,
    ViewLabel,
    ViewState,
)
from posekit.labeler.sheet import SheetInfo, contact_sheet, encode_jpeg, sheet_info
from posekit.labeler.store import LabelStore


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
        """The 1 fps keyframes, a subset of ``times_ns``."""
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
    save_lock: LockType
    """Serialises autosave and commit writes."""
    autosave_pending: bool = False
    """An autosave is scheduled and has not run yet."""


FrameKey: TypeAlias = tuple[str, int, int]
"""(segment id, cam, t_ns)."""


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
        self._segments: dict[str, Segment] = {}
        self._open_locks: dict[str, LockType] = {sid: threading.Lock() for sid in segment_ids}
        self._frames: OrderedDict[FrameKey, Future[UInt8[ndarray, "h w 3"]]] = OrderedDict()
        self._frames_lock = threading.Lock()
        self._prefetcher: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="labeler-prefetch")
        self._writer: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="labeler-autosave")

    def close(self) -> None:
        """Stop the background workers (pending autosaves still run)."""
        self._prefetcher.shutdown(wait=False, cancel_futures=True)
        self._writer.shutdown(wait=True)

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
            opened = Segment(info, frames, store, queue, threading.Lock())
            self._segments[segment_id] = opened
            print(f"[labeler] opened {segment_id}: {len(info.times_ns)} grid times, {len(store)} stored labels")
            return opened

    def summaries(self) -> list[SegmentSummary]:
        """One row per segment (opens every segment)."""
        rows: list[SegmentSummary] = []
        for segment_id in self.segment_ids:
            segment: Segment = self.segment(segment_id)
            times: list[int] = segment.info.times_ns
            duration_s: float = (times[-1] - times[0]) / 1e9 if times else 0.0
            rows.append(
                SegmentSummary(
                    segment_id, segment.info.split, duration_s, len(segment.info.keyframes_ns), segment.queue.n_reviewed_keyframes(), len(segment.store)
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
        """Layout and numbering of the contact sheet (no decoding)."""
        segment: Segment = self.segment(segment_id)
        self._grid_time(segment, t_ns)
        return sheet_info(segment_id, t_ns, segment.info.cams, self.proposals.proposals(segment_id, t_ns))

    def sheet_jpeg(self, segment_id: str, t_ns: int) -> bytes:
        """The rendered contact sheet."""
        segment: Segment = self.segment(segment_id)
        self._grid_time(segment, t_ns)
        images: list[UInt8[ndarray, "h w 3"]] = [self.frame(segment_id, camera.cam, t_ns) for camera in segment.info.cams]
        return contact_sheet(
            segment_id, t_ns, segment.info.cams, images, self.proposals.proposals(segment_id, t_ns), segment.store.get(t_ns)
        ).jpeg

    # -- labels ---------------------------------------------------------------------------------------------

    def put_labels(self, frame: FrameLabels) -> int:
        """Store a reviewer's decisions; schedules an autosave every ``autosave_every`` new labels."""
        segment: Segment = self.segment(frame.segment_id)
        self._grid_time(segment, frame.t_ns)
        for label in frame.labels:
            self._camera(segment, label.cam)
        stored: int = segment.store.put(frame)
        self._maybe_autosave(segment)
        return stored

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
            count: int = segment.store.save_layer(path)
        print(f"[labeler] autosaved {count} labels of {segment.info.segment_id} to {path}")

    def fill_range(self, request: RangeRequest) -> RangeResult:
        """Range fill between two reviewed keyframes."""
        segment: Segment = self.segment(request.segment_id)
        result: RangeResult = segment.queue.fill_range(request)
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
        with segment.save_lock:
            path: Path = self.layers.layer_path(segment_id)
            count: int = segment.store.save_layer(path)
            self.layers.register(path)
            self.layers.autosave_path(segment_id).unlink(missing_ok=True)
        print(f"[labeler] committed {count} labels of {segment_id} to {path}")
        return CommitResponse(str(path), count)


ROUTE_FRAME: re.Pattern[str] = re.compile(r"^/api/frame/([^/]+)/(-?\d+)/(-?\d+)\.jpg$")
ROUTE_FRAMESET: re.Pattern[str] = re.compile(r"^/api/frameset/([^/]+)/(-?\d+)$")
ROUTE_SHEET: re.Pattern[str] = re.compile(r"^/api/sheet/([^/]+)/(-?\d+)\.(jpg|json)$")
ROUTE_SEGMENT: re.Pattern[str] = re.compile(r"^/api/segment/([^/]+)$")
ROUTE_COMMIT: re.Pattern[str] = re.compile(r"^/api/commit/([^/]+)$")
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
            else:
                raise NotFoundError(f"no route GET {path}")

        def _post(self) -> None:
            path: str = urlparse(self.path).path
            body: str = self._body()
            if path == "/api/labels":
                self._json(StoredResponse(app.put_labels(from_json(FrameLabels, body))))
            elif path == "/api/range":
                self._json(app.fill_range(from_json(RangeRequest, body)))
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
