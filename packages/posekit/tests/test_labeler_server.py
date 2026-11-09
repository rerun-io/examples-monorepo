"""The labeler HTTP backend on an ephemeral port, with fake frames, proposals and layer storage."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
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
from posekit.labeler.rig import Kb4Rig
from posekit.labeler.schema import (
    AcceptRange,
    AcceptResult,
    ActionEvent,
    Box,
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
    SolveResult,
    TrackFrame,
    ViewLabel,
)
from posekit.labeler.server import FrameProvider, LabelApp, make_server
from posekit.labeler.sheet import SIDE_RGB, PlaybackInfo, SheetInfo
from posekit.labeler.sphere import HandSphere, box_iou, sphere_box
from posekit.labeler.store import LabelStore, LayerContents, load_layer_file, load_layer_file_contents
from posekit.labeler.tracker import BoxCarrier

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


def serve(
    tmp_path: Path, open_carrier: Callable[[str], BoxCarrier] | None = None, open_rig: Callable[[str], Kb4Rig] | None = None
) -> Iterator[Running]:
    static: Path = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>labeler</html>")
    (static / "app.js").write_text("console.log('hi')")
    (static / "assets").mkdir()
    (static / "assets" / "index-a1b2.js").write_text("export {}")
    (static / "assets" / "index-a1b2.css").write_text("body{}")
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
        open_carrier=open_carrier,
        open_rig=open_rig,
    )
    server: ThreadingHTTPServer = make_server(app, "127.0.0.1", 0)
    thread: threading.Thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Running(f"http://127.0.0.1:{server.server_address[1]}", app, layers, frames)
    server.shutdown()
    server.server_close()
    app.close()


@pytest.fixture
def running(tmp_path: Path) -> Iterator[Running]:
    yield from serve(tmp_path)


def decode_jpeg(jpeg: bytes) -> UInt8[ndarray, "h w 3"]:
    bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None
    return bgr


def fetch(url: str, body: str | None = None) -> tuple[int, str, bytes]:
    """(status, content type, body) of a GET, or of a POST when ``body`` is given."""
    request: urllib.request.Request = urllib.request.Request(
        url, data=None if body is None else body.encode(), method="GET" if body is None else "POST"
    )
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


def test_built_ui_assets(running: Running) -> None:
    """Vite's build (``base: './'``) loads ``./assets/<hash>.js|css`` from the server root."""
    assert fetch(running.url + "/assets/index-a1b2.js") == (200, "text/javascript", b"export {}")
    assert fetch(running.url + "/assets/index-a1b2.css") == (200, "text/css", b"body{}")
    assert fetch(running.url + "/assets/../../secret.txt")[0] == 404
    assert fetch(running.url + "/api/nope")[0] == 404


def look(yaw_deg: float, pitch_deg: float) -> ndarray:
    """Rig-from-camera rotation of a camera looking at yaw (about +y) and pitch (down); rig x right, y down, z forward."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    forward: ndarray = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    right: ndarray = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(forward, right), forward], axis=-1)


def six_camera_rig() -> Kb4Rig:
    """The labeler's six cameras at the fake frames' size, looking out and down around a 9 cm head."""
    yaws: tuple[float, ...] = (-70.0, -25.0, 25.0, 70.0, 180.0, 180.0)
    cam_from_rig: ndarray = np.tile(np.eye(4), (6, 1, 1))
    for cam, yaw in enumerate(yaws):
        rig_from_cam: ndarray = look(yaw, 35.0 if yaw != 180.0 else 0.0)
        origin: ndarray = 0.09 * np.array([np.sin(np.radians(yaw)), 0.0, np.cos(np.radians(yaw))])
        cam_from_rig[cam, :3, :3] = rig_from_cam.T
        cam_from_rig[cam, :3, 3] = -rig_from_cam.T @ origin
    return Kb4Rig(
        tuple(f"name{cam}" for cam in range(6)),
        np.tile([float(WIDTH), float(HEIGHT)], (6, 1)),
        np.full((6, 2), 38.0),
        np.tile([WIDTH / 2.0, HEIGHT / 2.0], (6, 1)),
        np.zeros((6, 4)),
        cam_from_rig,
    )


def test_solve_derives_every_view(tmp_path: Path) -> None:
    rig: Kb4Rig = six_camera_rig()
    hand: HandSphere = HandSphere(np.array([0.04, 0.22, 0.32]), 0.09)
    for running in serve(tmp_path, open_rig=lambda sid: rig):
        t_ns: int = FakeFrames().times_ns[2]
        labeled: list[dict[str, object]] = []
        for cam in (1, 2):
            box: Box | None = sphere_box(rig, cam, hand)
            assert box is not None
            labeled.append({"cam": cam, "box": {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1}})
        status, _, body = fetch(running.url + "/api/solve", json.dumps({"segment_id": "seg_a", "t_ns": t_ns, "side": "left", "labeled": labeled}))
        assert status == 200, body
        result: SolveResult = from_json(SolveResult, body.decode())
        assert (result.side, result.mode, result.labeled) == ("left", "triangulated", [1, 2])
        assert [item.cam for item in result.boxes] == list(range(6))
        derived: Box | None = result.boxes[3].box
        want: Box | None = sphere_box(rig, 3, hand)
        assert derived is not None and want is not None and box_iou(derived, want) > 0.9
        assert result.boxes[4].box is None and result.boxes[5].box is None
        empty: dict[str, object] = {"segment_id": "seg_a", "t_ns": t_ns, "side": "left", "labeled": []}
        assert fetch(running.url + "/api/solve", json.dumps(empty))[0] == 400


def test_solve_without_a_rig_is_not_found(running: Running) -> None:
    request: dict[str, object] = {
        "segment_id": "seg_a",
        "t_ns": FakeFrames().times_ns[0],
        "side": "left",
        "labeled": [{"cam": 0, "box": {"x0": 1.0, "y0": 1.0, "x1": 9.0, "y1": 9.0}}],
    }
    assert fetch(running.url + "/api/solve", json.dumps(request))[0] == 404


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
    result: RangeResult = from_json(
        RangeResult, fetch(running.url + "/api/range", to_json(RangeRequest("seg_a", keyframes[0], keyframes[1], "tester")))[2]
    )
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
    labels: list[ViewLabel] = [ViewLabel(1, "left", "no_hand", None, "person", "tester")]
    assert fetch(running.url + "/api/labels", to_json(FrameLabels("seg_a", 1_000, labels)))[0] == 200
    store: LabelStore = running.app.segment("seg_a").store
    assert store.unsaved == 2  # one label + its accept_frame action, below autosave_every=3: no autosave yet
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


@pytest.mark.integration
def test_committed_file_holds_the_action_log(running: Running) -> None:
    accept: AcceptRange = AcceptRange("seg_a", 1_000, 1_000 + 2 * STEP, "tester")
    assert fetch(running.url + "/api/accept_range", to_json(accept))[0] == 200
    moved: list[ViewLabel] = [ViewLabel(0, "left", "corrected", Box(15.0, 10.0, 65.0, 60.0), "agent", "tester")]
    assert fetch(running.url + "/api/labels", to_json(FrameLabels("seg_a", 1_000 + STEP, moved)))[0] == 200
    assert fetch(running.url + "/api/commit/seg_a", "")[0] == 200
    contents: LayerContents = load_layer_file_contents(running.layers.layer_path("seg_a"))
    assert contents.actions == running.app.actions("seg_a")
    assert [event.kind for event in contents.actions] == ["accept_range", "move_box"]
    assert len(contents.rows) == 36


@pytest.mark.parametrize(
    ("path", "body", "message"),
    [
        ("/api/labels", "{not json", ""),
        ("/api/labels", json.dumps({"segment_id": "seg_a", "t_ns": 1000, "labels": [], "extra": 1}), "extra"),
        (
            "/api/labels",
            json.dumps(
                {
                    "segment_id": "seg_a",
                    "t_ns": 1000,
                    "labels": [{"cam": 0, "side": "left", "status": "accepted", "box": None, "source": "person", "reviewer": "x"}],
                }
            ),
            "needs a box",
        ),
        (
            "/api/labels",
            json.dumps(
                {
                    "segment_id": "seg_a",
                    "t_ns": 1000,
                    "labels": [{"cam": 0, "side": "up", "status": "no_hand", "box": None, "source": "person", "reviewer": "x"}],
                }
            ),
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


def track_box(frames: list[TrackFrame], index: int, cam: int, side: str) -> tuple[str, Box | None]:
    found = [box for box in frames[index].boxes if (box.cam, box.side) == (cam, side)]
    assert len(found) == 1
    return found[0].origin, found[0].box


def test_track_accept_range_actions_and_effort(running: Running) -> None:
    frames: list[TrackFrame] = from_json(list[TrackFrame], get_json(running.url + "/api/track/seg_a?t0=1000&n=3"))
    assert [frame.t_ns for frame in frames] == [1_000, 1_000 + STEP, 1_000 + 2 * STEP]
    assert [(box.cam, box.side) for box in frames[0].boxes] == [(cam, side) for cam in range(6) for side in ("left", "right")]
    assert track_box(frames, 0, 0, "left") == ("detected", Box(10.0, 10.0, 60.0, 60.0))
    assert track_box(frames, 1, 0, "left")[0] == "chained"
    assert track_box(frames, 0, 3, "right") == ("none", None)  # score 0.4
    assert len(from_json(list[TrackFrame], get_json(running.url + "/api/track/seg_a"))) == 10

    accept: AcceptRange = AcceptRange("seg_a", 1_000, 1_000 + 2 * STEP, "tester")
    assert from_json(AcceptResult, fetch(running.url + "/api/accept_range", to_json(accept))[2]) == AcceptResult(accepted=36, kept=0)
    store: LabelStore = running.app.segment("seg_a").store
    assert store.get_view(1_000 + STEP, 0, "left") == ViewLabel(0, "left", "accepted", Box(10.0, 10.0, 60.0, 60.0), "tracker", "tester")
    assert store.get_view(1_000 + STEP, 3, "right") == ViewLabel(3, "right", "no_hand", None, "tracker", "tester")

    moved: ViewLabel = ViewLabel(0, "left", "corrected", Box(15.0, 10.0, 65.0, 60.0), "agent", "tester")
    assert fetch(running.url + "/api/labels", to_json(FrameLabels("seg_a", 1_000 + STEP, [moved])))[0] == 200
    drawn: ViewLabel = ViewLabel(1, "left", "corrected", Box(100.0, 50.0, 150.0, 100.0), "agent", "tester")
    assert fetch(running.url + "/api/labels", to_json(FrameLabels("seg_a", 1_000 + 3 * STEP, [drawn])))[0] == 200
    # The machine now starts from the fix; a second accept keeps every label (the agent's included).
    after: list[TrackFrame] = from_json(list[TrackFrame], get_json(running.url + f"/api/track/seg_a?t0={1_000 + 3 * STEP}&n=2"))
    assert track_box(after, 0, 1, "left") == ("reviewed", drawn.box)
    assert track_box(after, 1, 1, "left") == ("held", drawn.box)
    assert from_json(AcceptResult, fetch(running.url + "/api/accept_range", to_json(accept))[2]) == AcceptResult(accepted=0, kept=36)
    assert store.get_view(1_000 + STEP, 0, "left") == moved

    actions: list[ActionEvent] = from_json(list[ActionEvent], get_json(running.url + "/api/actions/seg_a"))
    assert [(event.kind, event.t_ns, event.cam, event.n_labels) for event in actions] == [
        ("accept_range", 1_000, None, 36),
        ("move_box", 1_000 + STEP, 0, 1),
        ("draw_box", 1_000 + 3 * STEP, 1, 1),
        ("accept_range", 1_000, None, 0),
    ]
    assert actions[0].t1_ns == 1_000 + 2 * STEP
    assert [event.wall_ns for event in actions] == sorted({event.wall_ns for event in actions})
    effort: EffortSummary = from_json(EffortSummary, get_json(running.url + "/api/effort/seg_a"))
    assert (effort.frames, effort.n_actions, effort.n_interventions, effort.frames_corrected) == (4, 4, 2, 2)
    assert effort.video_s == pytest.approx(0.8)
    assert effort.interventions_per_min == pytest.approx(150.0)
    assert effort.boxes_accepted_by_watching == 2
    assert effort.share_corrected == pytest.approx(0.5)


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"segment_id": "seg_a", "t0_ns": 1_000 + STEP, "t1_ns": 1_000, "reviewer": "x"}),
        json.dumps({"segment_id": "seg_a", "t0_ns": 1_000, "t1_ns": 7, "reviewer": "x"}),
        json.dumps({"segment_id": "seg_a", "t0_ns": 1_000, "t1_ns": 1_000}),
    ],
)
def test_bad_accept_range_is_400(running: Running, body: str) -> None:
    assert fetch(running.url + "/api/accept_range", body)[0] == 400


def test_playback_filmstrip(running: Running) -> None:
    info: PlaybackInfo = from_json(PlaybackInfo, get_json(running.url + "/api/playback/seg_a/1000.json?n=3"))
    assert [(row.r, row.t_ns) for row in info.rows] == [(0, 1_000), (1, 1_000 + STEP), (2, 1_000 + 2 * STEP)]
    assert [(cell.cam, cell.name, cell.w, cell.h) for cell in info.cells] == [(cam, f"name{cam}", 320, 180) for cam in range(6)]
    status, content_type, body = fetch(running.url + "/api/playback/seg_a/1000.jpg?n=3")
    assert (status, content_type) == (200, "image/jpeg")
    rgb: UInt8[ndarray, "h w 3"] = cv2.cvtColor(decode_jpeg(body), cv2.COLOR_BGR2RGB)
    assert rgb.shape == (info.height, info.width, 3)
    # cam 0's machine left box (native x0=10 -> 10 * 320/192 px into the cell) is blue in row 1.
    cell = info.cells[0]
    pixel: UInt8[ndarray, "3"] = rgb[info.rows[1].y + round(35 * cell.scale), cell.x + round(10 * cell.scale)]
    assert np.abs(pixel.astype(np.int64) - np.array(SIDE_RGB["left"])).max() < 70
    # The next window is decoded in the background.
    deadline: float = time.monotonic() + 30.0
    while (5, 1_000 + 5 * STEP) not in running.frames["seg_a"].decoded and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (5, 1_000 + 5 * STEP) in running.frames["seg_a"].decoded
    tail: PlaybackInfo = from_json(PlaybackInfo, get_json(running.url + f"/api/playback/seg_a/{1_000 + 9 * STEP}.json"))
    assert len(tail.rows) == 2
    assert fetch(running.url + "/api/playback/seg_a/1000.jpg?n=0")[0] == 400
    assert fetch(running.url + "/api/playback/seg_a/7.jpg")[0] == 400
    assert fetch(running.url + "/api/playback/zzz/1000.jpg")[0] == 404


def test_sheet_json_carries_the_machine_track(running: Running) -> None:
    info: SheetInfo = from_json(SheetInfo, get_json(running.url + "/api/sheet/seg_a/1000.json"))
    assert [(box.cam, box.side, box.origin) for box in info.track if box.box is not None] == [(0, "left", "detected")]
    assert fetch(running.url + "/api/sheet/seg_a/1000.jpg")[0] == 200


class Shift:
    """Fake visual tracker: the start box moves 5 px right per grid step."""

    def carry(self, cam: int, start_ns: int, start_box: Box, t_ns: int) -> Box | None:
        shift: float = 5.0 * (t_ns - start_ns) / STEP
        return Box(start_box.x0 + shift, start_box.y0, start_box.x1 + shift, start_box.y1)


def test_a_visual_tracker_carries_reviewed_boxes_and_the_strip_tags_them(tmp_path: Path) -> None:
    for running in serve(tmp_path, lambda segment_id: Shift()):
        drawn: ViewLabel = ViewLabel(1, "left", "corrected", Box(100.0, 50.0, 150.0, 100.0), "agent", "tester")
        assert fetch(running.url + "/api/labels", to_json(FrameLabels("seg_a", 1_000 + STEP, [drawn])))[0] == 200
        frames: list[TrackFrame] = from_json(list[TrackFrame], get_json(running.url + f"/api/track/seg_a?t0={1_000 + STEP}&n=3"))
        assert [track_box(frames, i, 1, "left") for i in range(3)] == [
            ("reviewed", drawn.box),
            ("tracked", Box(105.0, 50.0, 155.0, 100.0)),
            ("tracked", Box(110.0, 50.0, 160.0, 100.0)),
        ]
        assert track_box(frames, 1, 0, "left")[0] == "chained"  # detected chains are not carried
        assert fetch(running.url + f"/api/playback/seg_a/{1_000 + STEP}.jpg?n=3")[0] == 200
        assert fetch(running.url + f"/api/sheet/seg_a/{1_000 + 2 * STEP}.jpg")[0] == 200
        accept: AcceptRange = AcceptRange("seg_a", 1_000 + 2 * STEP, 1_000 + 2 * STEP, "tester")
        assert from_json(AcceptResult, fetch(running.url + "/api/accept_range", to_json(accept))[2]).accepted == 12
        stored: ViewLabel | None = running.app.segment("seg_a").store.get_view(1_000 + 2 * STEP, 1, "left")
        assert stored == ViewLabel(1, "left", "accepted", Box(105.0, 50.0, 155.0, 100.0), "tracker", "tester")
