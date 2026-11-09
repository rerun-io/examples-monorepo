"""The labeler HTTP backend on an ephemeral port, with fake frames, proposals and layer storage."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest
from jaxtyping import UInt8
from numpy import ndarray
from serde.json import from_json, to_json

from posekit.labeler.queue import ProposalProvider
from posekit.labeler.schema import (
    Box,
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
)
from posekit.labeler.server import FrameProvider, LabelApp, make_server
from posekit.labeler.sheet import SheetInfo
from posekit.labeler.store import LabelStore, load_layer_file

SEGMENTS: list[str] = ["seg_a", "seg_b"]
STEP: int = 200_000_000
WIDTH: int = 192
HEIGHT: int = 108


@dataclass
class FakeFrames:
    """Six cameras of solid colour; 11 grid times, keyframes every 5th."""

    cams: list[CameraInfo] = field(default_factory=lambda: [CameraInfo(cam, f"name{cam}", WIDTH, HEIGHT) for cam in range(6)])
    times_ns: list[int] = field(default_factory=lambda: [1_000 + i * STEP for i in range(11)])
    keyframes_ns: list[int] = field(default_factory=lambda: [1_000, 1_000 + 5 * STEP, 1_000 + 10 * STEP])
    decoded: list[tuple[int, int]] = field(default_factory=list)

    def frame(self, cam: int, t_ns: int) -> UInt8[ndarray, "h w 3"]:
        self.decoded.append((cam, t_ns))
        return np.full((HEIGHT, WIDTH, 3), 40 * cam, dtype=np.uint8)


class FakeProposals:
    def proposals(self, segment_id: str, t_ns: int) -> list[Proposal]:
        return [Proposal(0, "left", Box(10.0, 10.0, 60.0, 60.0), 0.9, "egohand"), Proposal(3, "right", Box(100.0, 20.0, 150.0, 90.0), 0.4, "egohand")]


class FakeLayers:
    def __init__(self, root: Path) -> None:
        self.root: Path = root
        self.registered: list[Path] = []

    def open_store(self, segment_id: str, split: SegmentSplit) -> LabelStore:
        return LabelStore(segment_id, split=split, application_id="test")

    def layer_path(self, segment_id: str) -> Path:
        return self.root / f"{segment_id}.rrd"

    def autosave_path(self, segment_id: str) -> Path:
        return self.root / "autosave" / f"{segment_id}.rrd"

    def register(self, path: Path) -> None:
        self.registered.append(path)


@dataclass
class Running:
    url: str
    app: LabelApp
    layers: FakeLayers
    frames: dict[str, FakeFrames]


@pytest.fixture
def running(tmp_path: Path) -> Iterator[Running]:
    static: Path = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>labeler</html>")
    (static / "app.js").write_text("console.log('hi')")
    (tmp_path / "secret.txt").write_text("nope")
    frames: dict[str, FakeFrames] = {}

    def open_frames(segment_id: str) -> FrameProvider:
        frames[segment_id] = FakeFrames()
        return frames[segment_id]

    proposals: ProposalProvider = FakeProposals()
    layers: FakeLayers = FakeLayers(tmp_path / "layers")
    app: LabelApp = LabelApp(
        segment_ids=SEGMENTS,
        open_frames=open_frames,
        proposals=proposals,
        layers=layers,
        viewer_url=lambda sid: f"rerun+http://example:9897/dataset/x?segment_id={sid}",
        eval_segments=frozenset({"seg_b"}),
        static_dir=static,
        reviewer="tester",
        autosave_every=3,
    )
    server: ThreadingHTTPServer = make_server(app, "127.0.0.1", 0)
    thread: threading.Thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Running(f"http://127.0.0.1:{server.server_address[1]}", app, layers, frames)
    server.shutdown()
    server.server_close()
    app.close()


def decode_jpeg(jpeg: bytes) -> UInt8[ndarray, "h w 3"]:
    bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None
    return bgr


def fetch(url: str, body: str | None = None) -> tuple[int, str, bytes]:
    """(status, content type, body) of a GET, or of a POST when ``body`` is given."""
    request: urllib.request.Request = urllib.request.Request(url, data=None if body is None else body.encode(), method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.headers["Content-Type"], response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers["Content-Type"], error.read()


def get_json(url: str) -> str:
    status, content_type, body = fetch(url)
    assert status == 200, body
    assert content_type == "application/json"
    return body.decode()


def test_static_files(running: Running) -> None:
    assert fetch(running.url + "/")[2] == b"<html>labeler</html>"
    status, content_type, body = fetch(running.url + "/static/app.js")
    assert (status, content_type, body) == (200, "text/javascript", b"console.log('hi')")
    assert fetch(running.url + "/static/../secret.txt")[0] == 404
    assert fetch(running.url + "/static/%2e%2e/secret.txt")[0] == 404
    assert fetch(running.url + "/static/missing.css")[0] == 404
    assert fetch(running.url + "/nope")[0] == 404


def test_config_segments_and_segment(running: Running) -> None:
    assert json.loads(get_json(running.url + "/api/config")) == {"reviewer": "tester", "dataset": "robocap-labeling"}
    summaries: list[SegmentSummary] = from_json(list[SegmentSummary], get_json(running.url + "/api/segments"))
    assert [(row.segment_id, row.split, row.n_keyframes, row.n_reviewed_keyframes, row.n_labels) for row in summaries] == [
        ("seg_a", "train", 3, 0, 0),
        ("seg_b", "eval", 3, 0, 0),
    ]
    assert summaries[0].duration_s == pytest.approx(2.0)
    info: SegmentInfo = from_json(SegmentInfo, get_json(running.url + "/api/segment/seg_b"))
    assert info.split == "eval" and len(info.cams) == 6 and info.keyframes_ns[1] == 1_000 + 5 * STEP
    assert info.viewer_url.endswith("segment_id=seg_b")
    status, _, body = fetch(running.url + "/api/segment/zzz")
    assert status == 404 and "unknown segment" in json.loads(body)["error"]


def test_frame_jpeg(running: Running) -> None:
    status, content_type, body = fetch(running.url + f"/api/frame/seg_a/2/{1_000 + STEP}.jpg?max_side=96")
    assert (status, content_type) == (200, "image/jpeg")
    image: UInt8[ndarray, "h w 3"] = decode_jpeg(body)
    assert image.shape == (54, 96, 3)
    assert abs(int(image[20, 20, 0]) - 80) < 4
    # A second request is served from the cache.
    fetch(running.url + f"/api/frame/seg_a/2/{1_000 + STEP}.jpg")
    assert running.frames["seg_a"].decoded.count((2, 1_000 + STEP)) == 1
    assert fetch(running.url + f"/api/frame/seg_a/9/{1_000}.jpg")[0] == 404
    assert fetch(running.url + "/api/frame/seg_a/0/5.jpg")[0] == 400
    assert fetch(running.url + "/api/frame/seg_a/0/1000.jpg?max_side=abc")[0] == 400


def test_frameset_and_sheet(running: Running) -> None:
    state: FrameState = from_json(FrameState, get_json(running.url + "/api/frameset/seg_a/1000"))
    assert [view.cam for view in state.views] == list(range(6))
    assert len(state.views[0].proposals) == 1 and state.views[0].labels == []
    assert state.flags == ["low_score"]
    info: SheetInfo = from_json(SheetInfo, get_json(running.url + "/api/sheet/seg_a/1000.json"))
    assert [p.k for p in info.proposals] == [1, 2]
    assert (info.width, info.cells[0].scale) == (1920, pytest.approx(10 / 3))
    status, content_type, body = fetch(running.url + "/api/sheet/seg_a/1000.jpg")
    assert (status, content_type) == (200, "image/jpeg")
    sheet: UInt8[ndarray, "h w 3"] = decode_jpeg(body)
    assert sheet.shape == (info.height, info.width, 3)
    assert fetch(running.url + "/api/frameset/seg_a/7")[0] == 400


def test_labels_queue_range_commit(running: Running) -> None:
    keyframes: list[int] = [1_000, 1_000 + 5 * STEP]
    queue: list[QueueItem] = from_json(list[QueueItem], get_json(running.url + "/api/queue?sid=seg_a&limit=2"))
    assert [item.t_ns for item in queue] == keyframes
    assert queue[0].reasons == ["keyframe", "low_score"]
    for t_ns in keyframes:
        labels: list[ViewLabel] = [
            ViewLabel(cam, side, "accepted", Box(10.0, 10.0, 60.0, 60.0), "egohand", "tester")
            if (cam, side) == (0, "left")
            else ViewLabel(cam, side, "no_hand", None, "person", "tester")
            for cam in range(6)
            for side in ("left", "right")
        ]
        status, _, body = fetch(running.url + "/api/labels", to_json(FrameLabels("seg_a", t_ns, labels)))
        assert (status, json.loads(body)) == (200, {"stored": 12})
    result: RangeResult = from_json(RangeResult, fetch(running.url + "/api/range", to_json(RangeRequest("seg_a", keyframes[0], keyframes[1], "tester")))[2])
    assert result == RangeResult(filled=4 * 12, queued=[])
    remaining: list[QueueItem] = from_json(list[QueueItem], get_json(running.url + "/api/queue?sid=seg_a"))
    assert [item.t_ns for item in remaining] == [1_000 + 10 * STEP]
    state: FrameState = from_json(FrameState, get_json(running.url + f"/api/frameset/seg_a/{1_000 + STEP}"))
    assert state.views[0].labels[0].source == "range"
    # The autosave runs in the background after every few labels.
    deadline: float = time.monotonic() + 30.0
    while not running.layers.autosave_path("seg_a").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert running.layers.autosave_path("seg_a").exists()
    status, _, body = fetch(running.url + "/api/commit/seg_a", "")
    assert status == 200
    committed: dict[str, object] = json.loads(body)
    assert committed == {"path": str(running.layers.layer_path("seg_a")), "n_labels": 6 * 12}
    assert running.layers.registered == [running.layers.layer_path("seg_a")]
    assert not running.layers.autosave_path("seg_a").exists()
    summaries: list[SegmentSummary] = from_json(list[SegmentSummary], get_json(running.url + "/api/segments"))
    assert (summaries[0].n_reviewed_keyframes, summaries[0].n_labels) == (2, 72)
    assert fetch(running.url + "/api/commit/zzz", "")[0] == 404


def test_close_saves_labels_below_the_autosave_threshold(running: Running) -> None:
    labels: list[ViewLabel] = [ViewLabel(0, side, "no_hand", None, "person", "tester") for side in ("left", "right")]
    assert fetch(running.url + "/api/labels", to_json(FrameLabels("seg_a", 1_000, labels)))[0] == 200
    store: LabelStore = running.app.segment("seg_a").store
    assert store.unsaved == 2  # below autosave_every=3: no autosave yet
    running.app.close()
    assert running.layers.autosave_path("seg_a").exists()
    assert store.unsaved == 0
    # After close, writes are refused instead of being acknowledged and lost.
    status, _, body = fetch(running.url + "/api/labels", to_json(FrameLabels("seg_a", 1_000 + STEP, labels)))
    assert status == 503 and "shutting down" in json.loads(body)["error"]
    assert store.get(1_000 + STEP) == []


@pytest.mark.integration
def test_committed_file_reads_back(running: Running) -> None:
    labels: list[ViewLabel] = [ViewLabel(0, "left", "accepted", Box(10.0, 10.0, 60.0, 60.0), "egohand", "tester")]
    assert fetch(running.url + "/api/labels", to_json(FrameLabels("seg_a", 1_000, labels)))[0] == 200
    assert fetch(running.url + "/api/commit/seg_a", "")[0] == 200
    rows = load_layer_file(running.layers.layer_path("seg_a"))
    assert [(row.t_ns, row.label) for row in rows] == [(1_000, labels[0])]


@pytest.mark.parametrize(
    ("path", "body", "message"),
    [
        ("/api/labels", "{not json", ""),
        ("/api/labels", json.dumps({"segment_id": "seg_a", "t_ns": 1000, "labels": [], "extra": 1}), "extra"),
        (
            "/api/labels",
            json.dumps({"segment_id": "seg_a", "t_ns": 1000, "labels": [{"cam": 0, "side": "left", "status": "accepted", "box": None, "source": "person", "reviewer": "x"}]}),
            "needs a box",
        ),
        (
            "/api/labels",
            json.dumps({"segment_id": "seg_a", "t_ns": 1000, "labels": [{"cam": 0, "side": "up", "status": "no_hand", "box": None, "source": "person", "reviewer": "x"}]}),
            "",
        ),
        ("/api/labels", json.dumps({"segment_id": "seg_a", "t_ns": 3, "labels": []}), "grid"),
        ("/api/range", json.dumps({"segment_id": "seg_a", "t0_ns": 1000, "t1_ns": 1000 + 10 * STEP, "reviewer": "x"}), "reviewed"),
    ],
)
def test_bad_input_is_400(running: Running, path: str, body: str, message: str) -> None:
    status, content_type, answer = fetch(running.url + path, body)
    assert (status, content_type) == (400, "application/json")
    assert message in json.loads(answer)["error"]


def test_bad_queue_requests(running: Running) -> None:
    assert fetch(running.url + "/api/queue")[0] == 400
    assert fetch(running.url + "/api/queue?sid=seg_a&limit=0")[0] == 400
    assert fetch(running.url + "/api/queue?sid=nope")[0] == 404
