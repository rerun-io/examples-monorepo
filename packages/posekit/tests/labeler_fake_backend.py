"""Fake hand-labeler backend for driving the static frontend by hand or with playwright (not a pytest module).

Serves ``src/posekit/labeler/static`` at ``/`` and ``/static/<file>``, canned JSON for every ``/api`` route of the
spec, and generated JPEG frames (two skin-coloured "hands" per camera that drift over time, so proposals and
corrections can be checked by eye). Every POST body is decoded with the real pyserde schema; a body that does
not decode gets a 400 and is logged. Valid bodies are appended to ``--post-log`` (JSON lines).

The v2 MEgATrack loop (SPEC-v2-megatrack-loop.md) is here too: ``/api/track`` runs a small machine tracker over
the canned EgoHand proposals and the stored labels (reviewed > chained > held > detected > none),
``/api/accept_range`` writes the machine boxes as ``tracker`` labels, ``/api/labels`` logs one ActionEvent per
label that differs from the machine track, and ``/api/actions`` + ``/api/effort`` report them.

Run: ``python tests/labeler_fake_backend.py --port 9911``
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from jaxtyping import UInt8
from numpy import ndarray
from serde import SerdeError
from serde.json import from_json, to_json

from posekit.labeler.schema import (
    HAND_SIDES,
    AcceptRange,
    AcceptResult,
    ActionEvent,
    ActionKind,
    Box,
    CameraInfo,
    EffortSummary,
    FrameLabels,
    FrameState,
    HandSide,
    Proposal,
    QueueItem,
    RangeRequest,
    RangeResult,
    SegmentInfo,
    SegmentSummary,
    TrackBox,
    TrackFrame,
    ViewLabel,
    ViewState,
)

__test__ = False

STATIC_DIR: Path = Path(__file__).resolve().parents[1] / "src" / "posekit" / "labeler" / "static"
CAM_NAMES: tuple[str, ...] = ("left_front", "right_front", "left_eye", "right_eye", "left", "right")
WIDTH: int = 1920
HEIGHT: int = 1080
STEP_NS: int = 200_000_000
"""5 fps label grid."""
KEYFRAME_EVERY: int = 5
"""1 fps keyframes."""


@dataclass(slots=True)
class FakeSegment:
    """One canned segment."""

    segment_id: str
    split: str
    start_ns: int
    n_times: int
    viewer_url: str

    @property
    def times_ns(self) -> list[int]:
        return [self.start_ns + i * STEP_NS for i in range(self.n_times)]

    @property
    def keyframes_ns(self) -> list[int]:
        return self.times_ns[::KEYFRAME_EVERY]


@dataclass(slots=True)
class FakeState:
    """Everything the fake keeps in memory."""

    segments: dict[str, FakeSegment]
    labels: dict[tuple[str, int], dict[tuple[int, HandSide], ViewLabel]] = field(default_factory=dict)
    range_queued: dict[str, set[int]] = field(default_factory=dict)
    actions: list[ActionEvent] = field(default_factory=list)
    post_log: Path | None = None


def true_hand(cam: int, side: HandSide, t_ns: int) -> Box:
    """Where the drawn hand really is (native px)."""
    s: float = t_ns / 1e9
    phase: float = cam * 0.9 + (0.0 if side == "left" else 2.1)
    cx: float = (620.0 if side == "left" else 1300.0) + 180.0 * math.sin(0.7 * s + phase)
    cy: float = 640.0 + 120.0 * math.cos(0.5 * s + phase)
    return Box(x0=cx - 120.0, y0=cy - 110.0, x1=cx + 120.0, y1=cy + 110.0)


def proposals_for(cam: int, t_ns: int) -> list[Proposal]:
    """Canned proposals: EgoHand mostly right but offset; gaps on cams 4/5; hands layer on cams 2/3."""
    out: list[Proposal] = []
    k: int = t_ns // STEP_NS
    for side in HAND_SIDES:
        if cam == 5 or (cam == 4 and side == "left"):
            continue
        if cam == 0 and k % 11 == 5:
            continue  # a one-step dropout: the machine track holds the previous box
        b: Box = true_hand(cam, side, t_ns)
        off: float = 25.0 * math.sin(k + cam)
        if cam == 3 and side == "right" and k % 13 == 0:
            b = Box(x0=b.x0 - 520.0, y0=b.y0, x1=b.x1 - 520.0, y1=b.y1)  # a false detection far off: the chain breaks there
        egohand_side: HandSide = side
        if cam == 1 and k % 7 == 0:
            egohand_side = "right" if side == "left" else "left"  # a side flip to correct
        out.append(
            Proposal(
                cam=cam,
                side=egohand_side,
                box=Box(x0=b.x0 + off, y0=b.y0 + off, x1=b.x1 + off * 0.5, y1=b.y1 - off * 0.5),
                score=round(0.55 + 0.4 * abs(math.cos(k * 0.3 + cam)), 3),
                model="egohand",
            )
        )
        if cam in (2, 3):
            out.append(
                Proposal(
                    cam=cam,
                    side=side,
                    box=Box(x0=b.x0 - 30.0, y0=b.y0 - 20.0, x1=b.x1 + 30.0, y1=b.y1 + 40.0),
                    score=round(0.5 + 0.3 * abs(math.sin(k * 0.2 + cam)), 3),
                    model="hands_layer",
                )
            )
    return out


@lru_cache(maxsize=256)
def frame_jpeg(cam: int, t_ns: int, max_side: int) -> bytes:
    """A 1920x1080 synthetic frame with grid, labels and two hands, downscaled to ``max_side``."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    img: UInt8[ndarray, "h w 3"] = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[..., 0] = (40 + 30 * cam + (xx * 60) // WIDTH).astype(np.uint8)
    img[..., 1] = (35 + (yy * 50) // HEIGHT).astype(np.uint8)
    img[..., 2] = 50 + 10 * cam
    for x in range(0, WIDTH, 100):
        cv2.line(img, (x, 0), (x, HEIGHT - 1), (90, 90, 90) if x % 500 else (160, 160, 160), 1)
    for y in range(0, HEIGHT, 100):
        cv2.line(img, (0, y), (WIDTH - 1, y), (90, 90, 90) if y % 500 else (160, 160, 160), 1)
    for side in HAND_SIDES:
        b: Box = true_hand(cam, side, t_ns)
        center: tuple[int, int] = (int((b.x0 + b.x1) / 2), int((b.y0 + b.y1) / 2))
        cv2.ellipse(img, center, (int((b.x1 - b.x0) / 2), int((b.y1 - b.y0) / 2)), 0, 0, 360, (120, 160, 220), -1)
        cv2.putText(img, side[0].upper(), (center[0] - 25, center[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (40, 40, 40), 6)
    text: str = f"cam_{cam:02d} {CAM_NAMES[cam]}  t={t_ns / 1e9:.2f}s"
    cv2.putText(img, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    scale: float = min(1.0, max_side / max(WIDTH, HEIGHT))
    if scale < 1.0:
        img = cv2.resize(img, (round(WIDTH * scale), round(HEIGHT * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def is_reviewed(state: FakeState, sid: str, t_ns: int) -> bool:
    return len(state.labels.get((sid, t_ns), {})) == len(CAM_NAMES) * len(HAND_SIDES)


def summary(state: FakeState, seg: FakeSegment) -> SegmentSummary:
    return SegmentSummary(
        segment_id=seg.segment_id,
        split="eval" if seg.split == "eval" else "train",
        duration_s=seg.n_times * STEP_NS / 1e9,
        n_keyframes=len(seg.keyframes_ns),
        n_reviewed_keyframes=sum(is_reviewed(state, seg.segment_id, t) for t in seg.keyframes_ns),
        n_labels=sum(len(v) for (sid, _), v in state.labels.items() if sid == seg.segment_id),
    )


def queue(state: FakeState, seg: FakeSegment, limit: int) -> list[QueueItem]:
    items: list[QueueItem] = []
    for i, t in enumerate(seg.keyframes_ns):
        if not is_reviewed(state, seg.segment_id, t):
            reasons: list[str] = ["keyframe"] + (["side_flip"] if (t // STEP_NS) % 7 == 0 else []) + (["views_disagree"] if i % 4 == 3 else [])
            items.append(QueueItem(segment_id=seg.segment_id, t_ns=t, reasons=reasons))
    for t in sorted(state.range_queued.get(seg.segment_id, set())):
        if not is_reviewed(state, seg.segment_id, t):
            items.append(QueueItem(segment_id=seg.segment_id, t_ns=t, reasons=["range_gap"]))
    return items[:limit]


def range_fill(state: FakeState, req: RangeRequest) -> RangeResult:
    seg: FakeSegment = state.segments[req.segment_id]
    filled: int = 0
    queued: list[int] = []
    for t in seg.times_ns:
        if not (req.t0_ns < t < req.t1_ns) or is_reviewed(state, seg.segment_id, t):
            continue
        if (t // STEP_NS) % 3 == 0:
            queued.append(t)
            continue
        by_side: dict[tuple[int, HandSide], ViewLabel] = {}
        for cam in range(len(CAM_NAMES)):
            props: list[Proposal] = [p for p in proposals_for(cam, t) if p.model == "egohand"]
            for side in HAND_SIDES:
                p: Proposal | None = next((q for q in props if q.side == side), None)
                by_side[(cam, side)] = ViewLabel(
                    cam=cam,
                    side=side,
                    status="accepted" if p else "no_hand",
                    box=p.box if p else None,
                    source="range",
                    reviewer=req.reviewer,
                )
        state.labels[(seg.segment_id, t)] = by_side
        filled += len(by_side)
    state.range_queued.setdefault(seg.segment_id, set()).update(queued)
    return RangeResult(filled=filled, queued=queued)


def iou(a: Box, b: Box) -> float:
    ix: float = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy: float = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter: float = ix * iy
    union: float = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union if union > 0.0 else 0.0


def track_side(state: FakeState, seg: FakeSegment, cam: int, side: HandSide) -> dict[int, TrackBox]:
    """The spec's propagation rule for one (cam, side) over the whole segment."""
    out: dict[int, TrackBox] = {}
    prev: Box | None = None
    held: bool = False
    for t in seg.times_ns:
        label: ViewLabel | None = state.labels.get((seg.segment_id, t), {}).get((cam, side))
        if label is not None:
            out[t] = TrackBox(cam=cam, side=side, box=label.box, origin="reviewed", score=None)
            if label.status != "rejected":
                prev, held = label.box, False
            continue
        props: list[Proposal] = [p for p in proposals_for(cam, t) if p.model == "egohand"]
        if prev is not None:
            chain_from: Box = prev
            best: Proposal | None = max((p for p in props if iou(p.box, chain_from) >= 0.3), key=lambda p: iou(p.box, chain_from), default=None)
            if best is not None:
                out[t] = TrackBox(cam=cam, side=side, box=best.box, origin="chained", score=best.score)
                prev, held = best.box, False
                continue
            if not held:
                out[t] = TrackBox(cam=cam, side=side, box=prev, origin="held", score=None)
                held = True
                continue
            prev, held = None, False
        det: Proposal | None = max((p for p in props if p.side == side and p.score >= 0.5), key=lambda p: p.score, default=None)
        if det is not None:
            out[t] = TrackBox(cam=cam, side=side, box=det.box, origin="detected", score=det.score)
            prev = det.box
        else:
            out[t] = TrackBox(cam=cam, side=side, box=None, origin="none", score=None)
    return out


def track(state: FakeState, seg: FakeSegment) -> list[TrackFrame]:
    sides: list[dict[int, TrackBox]] = [track_side(state, seg, cam, side) for cam in range(len(CAM_NAMES)) for side in HAND_SIDES]
    return [TrackFrame(t_ns=t, boxes=[s[t] for s in sides]) for t in seg.times_ns]


def accept_range(state: FakeState, req: AcceptRange) -> AcceptResult:
    seg: FakeSegment = state.segments[req.segment_id]
    accepted: int = 0
    kept: int = 0
    for frame in track(state, seg):
        if not (req.t0_ns <= frame.t_ns <= req.t1_ns):
            continue
        bucket: dict[tuple[int, HandSide], ViewLabel] = state.labels.setdefault((seg.segment_id, frame.t_ns), {})
        for tb in frame.boxes:
            if (tb.cam, tb.side) in bucket:
                kept += 1
                continue
            bucket[(tb.cam, tb.side)] = ViewLabel(
                cam=tb.cam, side=tb.side, status="accepted" if tb.box else "no_hand", box=tb.box, source="tracker", reviewer=req.reviewer
            )
            accepted += 1
    state.actions.append(
        ActionEvent(
            wall_ns=time.time_ns(),
            segment_id=req.segment_id,
            t_ns=req.t0_ns,
            t1_ns=req.t1_ns,
            cam=None,
            side=None,
            kind="accept_range",
            reviewer=req.reviewer,
            n_labels=accepted,
        )
    )
    return AcceptResult(accepted=accepted, kept=kept)


def classify(machine: TrackBox, label: ViewLabel, props: list[Proposal]) -> ActionKind | None:
    """The action a posted label stands for, against the machine box before the post (None = unchanged)."""
    if label.status == "rejected":
        return "reject"
    if label.box is None:
        return "no_hand" if machine.box is not None else None
    if machine.box is None or iou(machine.box, label.box) < 0.5:
        near: bool = any(
            max(abs(p.box.x0 - label.box.x0), abs(p.box.y0 - label.box.y0), abs(p.box.x1 - label.box.x1), abs(p.box.y1 - label.box.y1)) <= 1.0
            for p in props
        )
        return "pick_proposal" if near else "draw_box"
    if machine.box == label.box:
        return None
    return "move_box"


def log_label_actions(state: FakeState, fl: FrameLabels) -> None:
    """Classify a post against the machine track before it is stored (the spec's set_side needs the other side's track)."""
    seg: FakeSegment = state.segments[fl.segment_id]
    frame: TrackFrame = next(f for f in track(state, seg) if f.t_ns == fl.t_ns)
    machine: dict[tuple[int, HandSide], TrackBox] = {(b.cam, b.side): b for b in frame.boxes}
    wall: int = time.time_ns()
    events: list[ActionEvent] = []
    for label in fl.labels:
        kind: ActionKind | None = classify(machine[(label.cam, label.side)], label, proposals_for(label.cam, fl.t_ns))
        other: TrackBox = machine[(label.cam, "right" if label.side == "left" else "left")]
        if kind in ("draw_box", "move_box") and label.box is not None and other.box is not None and iou(other.box, label.box) >= 0.9:
            kind = "set_side"
        if kind is not None:
            events.append(
                ActionEvent(
                    wall_ns=wall,
                    segment_id=fl.segment_id,
                    t_ns=fl.t_ns,
                    t1_ns=None,
                    cam=label.cam,
                    side=label.side,
                    kind=kind,
                    reviewer=label.reviewer,
                    n_labels=1,
                )
            )
    if not events and fl.labels:
        events.append(
            ActionEvent(
                wall_ns=wall,
                segment_id=fl.segment_id,
                t_ns=fl.t_ns,
                t1_ns=None,
                cam=None,
                side=None,
                kind="accept_frame",
                reviewer=fl.labels[0].reviewer,
                n_labels=len(fl.labels),
            )
        )
    state.actions.extend(events)


def effort(state: FakeState, seg: FakeSegment) -> EffortSummary:
    acts: list[ActionEvent] = [a for a in state.actions if a.segment_id == seg.segment_id]
    interventions: list[ActionEvent] = [a for a in acts if a.kind not in ("accept_range", "accept_frame")]
    frames: int = sum(1 for (sid, _), v in state.labels.items() if sid == seg.segment_id and v)
    video_s: float = frames * STEP_NS / 1e9
    frames_corrected: int = len({a.t_ns for a in interventions})
    by_watching: int = sum(
        1 for (sid, _), v in state.labels.items() if sid == seg.segment_id for lb in v.values() if lb.source == "tracker" and lb.box is not None
    )
    wall_s: float = (max(a.wall_ns for a in acts) - min(a.wall_ns for a in acts)) / 1e9 if acts else 0.0
    n_boxes: int = sum(len(v) for (sid, _), v in state.labels.items() if sid == seg.segment_id)
    return EffortSummary(
        video_s=video_s,
        n_actions=len(acts),
        n_interventions=len(interventions),
        interventions_per_min=len(interventions) / (video_s / 60.0) if video_s > 0.0 else None,
        frames=frames,
        frames_corrected=frames_corrected,
        share_corrected=frames_corrected / frames if frames else None,
        boxes_accepted_by_watching=by_watching,
        reviewer_wall_s=wall_s,
        boxes_per_reviewer_s=n_boxes / wall_s if wall_s > 0.0 else None,
    )


def make_handler(state: FakeState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(f"[fake] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

        def send_bytes(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, text: str, code: int = 200) -> None:
            self.send_bytes(code, text.encode(), "application/json")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            path: str = unquote(url.path)
            query: dict[str, list[str]] = parse_qs(url.query)
            if path in ("/", "/index.html"):
                self.send_bytes(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                return
            if m := re.fullmatch(r"/static/([\w.-]+)", path):
                f: Path = STATIC_DIR / m[1]
                ctype: str = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}.get(f.suffix, "application/octet-stream")
                if f.is_file():
                    self.send_bytes(200, f.read_bytes(), ctype)
                else:
                    self.send_bytes(404, b"no such file", "text/plain")
                return
            if path == "/api/config":
                self.send_json(json.dumps({"reviewer": "fake-default", "dataset": "robocap-labeling (fake)"}))
                return
            if path == "/api/segments":
                self.send_json(to_json([summary(state, s) for s in state.segments.values()]))
                return
            if m := re.fullmatch(r"/api/segment/([\w-]+)", path):
                seg = state.segments[m[1]]
                info = SegmentInfo(
                    segment_id=seg.segment_id,
                    split="eval" if seg.split == "eval" else "train",
                    cams=[CameraInfo(cam=i, name=n, width=WIDTH, height=HEIGHT) for i, n in enumerate(CAM_NAMES)],
                    times_ns=seg.times_ns,
                    keyframes_ns=seg.keyframes_ns,
                    viewer_url=seg.viewer_url,
                )
                self.send_json(to_json(info))
                return
            if m := re.fullmatch(r"/api/frame/([\w-]+)/(\d)/(\d+)\.jpg", path):
                max_side: int = int(query.get("max_side", ["960"])[0])
                self.send_bytes(200, frame_jpeg(int(m[2]), int(m[3]), max_side), "image/jpeg")
                return
            if m := re.fullmatch(r"/api/frameset/([\w-]+)/(\d+)", path):
                sid, t = m[1], int(m[2])
                stored = state.labels.get((sid, t), {})
                views = [
                    ViewState(cam=cam, proposals=proposals_for(cam, t), labels=[v for (c, _), v in stored.items() if c == cam])
                    for cam in range(len(CAM_NAMES))
                ]
                flags: list[str] = next((q.reasons for q in queue(state, state.segments[sid], 1000) if q.t_ns == t), [])
                self.send_json(to_json(FrameState(segment_id=sid, t_ns=t, views=views, flags=flags)))
                return
            if m := re.fullmatch(r"/api/track/([\w-]+)", path):
                seg = state.segments[m[1]]
                t0: int = int(query.get("t0", [str(seg.start_ns)])[0])
                n_frames: int = int(query.get("n", ["10"])[0])
                self.send_json(to_json([f for f in track(state, seg) if f.t_ns >= t0][:n_frames]))
                return
            if m := re.fullmatch(r"/api/actions/([\w-]+)", path):
                self.send_json(to_json([a for a in state.actions if a.segment_id == m[1]]))
                return
            if m := re.fullmatch(r"/api/effort/([\w-]+)", path):
                self.send_json(to_json(effort(state, state.segments[m[1]])))
                return
            if path == "/api/queue":
                sid = query["sid"][0]
                self.send_json(to_json(queue(state, state.segments[sid], int(query.get("limit", ["50"])[0]))))
                return
            self.send_bytes(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            path: str = unquote(urlparse(self.path).path)
            body: str = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
            try:
                if path == "/api/labels":
                    fl: FrameLabels = from_json(FrameLabels, body)
                    log_label_actions(state, fl)
                    bucket = state.labels.setdefault((fl.segment_id, fl.t_ns), {})
                    for label in fl.labels:
                        bucket[(label.cam, label.side)] = label
                    self.log_post(path, body)
                    self.send_json(json.dumps({"stored": len(fl.labels)}))
                    return
                if path == "/api/accept_range":
                    ar: AcceptRange = from_json(AcceptRange, body)
                    self.log_post(path, body)
                    self.send_json(to_json(accept_range(state, ar)))
                    return
                if path == "/api/range":
                    req: RangeRequest = from_json(RangeRequest, body)
                    self.log_post(path, body)
                    self.send_json(to_json(range_fill(state, req)))
                    return
                if m := re.fullmatch(r"/api/commit/([\w-]+)", path):
                    n: int = sum(len(v) for (sid, _), v in state.labels.items() if sid == m[1])
                    self.log_post(path, body)
                    self.send_json(json.dumps({"path": f"/fake/layers/hand_boxes/{m[1]}.rrd", "n_labels": n}))
                    return
            except (SerdeError, ValueError, TypeError) as e:
                print(f"[fake] INVALID {path}: {e}\n{body}", flush=True)
                self.send_bytes(400, f"invalid body: {e}".encode(), "text/plain")
                return
            self.send_bytes(404, b"not found", "text/plain")

        def log_post(self, path: str, body: str) -> None:
            if state.post_log is not None:
                with state.post_log.open("a") as f:
                    f.write(json.dumps({"path": path, "body": json.loads(body) if body else None}) + "\n")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=9911)
    parser.add_argument("--post-log", type=Path, default=Path("/tmp/hl-f/posted.jsonl"))
    parser.add_argument("--viewer-url", default="rerun+http://127.0.0.1:1/dataset/fake?segment_id={sid}", help="{sid} is replaced")
    args = parser.parse_args()
    args.post_log.parent.mkdir(parents=True, exist_ok=True)
    segments: list[FakeSegment] = [
        FakeSegment("fake_s21", "train", 99_000_000_000, 100, args.viewer_url.format(sid="fake_s21")),
        FakeSegment("fake_s13", "eval", 54_000_000_000, 50, args.viewer_url.format(sid="fake_s13")),
    ]
    state = FakeState(segments={s.segment_id: s for s in segments}, post_log=args.post_log)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"[fake] serving {STATIC_DIR} on http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
