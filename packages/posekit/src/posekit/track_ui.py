"""Click-to-track: one object, clicks on any frame of the video inside the Rerun viewer (Kineo's interaction model).

Every session owns one :class:`ClickTracker` (clip + SAM2 memory state) and one
Rerun recording; each viewer event appends to that recording through
``RecordingStream.binary_stream()``:

* ``selection_change`` → a click on the video at the viewer's current frame
  adds a positive/negative point or removes the nearest one (mode radio), and
  the re-prompted mask appears on that frame.
* ``time_update`` (paused scrubbing) → the memory-conditioned preview mask for
  the frame under the cursor, without writing memory.
* **Track** → bidirectional propagation from the first prompted frame, streamed in.
"""

from __future__ import annotations

import functools
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
from gradio_rerun import Rerun
from gradio_rerun.events import Pause, Play, SelectionChange, TimeUpdate
from jaxtyping import Int, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import log_video

from posekit.apis.click_tracker import ClickTracker, MaskResult, Point
from posekit.models.sam2_video import Sam2Variant, Sam2VideoSegmenterConfig

APP_ID: str = "posekit_click_to_track"
VIDEO: str = "video"
TIMELINE: str = "video_time"
Mode: TypeAlias = Literal["+ Include", "− Exclude", "✕ Remove"]
MODES: list[Mode] = ["+ Include", "− Exclude", "✕ Remove"]
POSITIVE_COLOR: tuple[int, int, int] = (76, 220, 96)
NEGATIVE_COLOR: tuple[int, int, int] = (255, 87, 51)
MASK_COLOR: tuple[int, int, int] = (51, 178, 255)
MAX_SESSIONS: int = 4
RRD_DIR: Path = Path("/tmp/posekit-rrd")
_EXAMPLES_DIR: Path = Path(__file__).resolve().parents[2] / "assets" / "examples"
EXAMPLE_VIDEOS: list[str] = [
    str(path)
    for path in (
        # robocap segment robocap__f408193e6447b3b0__s00000021, /world/rig_00/cam_0{0,1,2}/pinhole/video
        # (catalog H.264 samples remuxed, cut from t=2s after sensor warm-up, 10 s @ 720p).
        *sorted(_EXAMPLES_DIR.glob("robocap_*.mp4")),
        Path(__file__).resolve().parents[3] / "wilor-nano" / "assets" / "video.mp4",
        Path(__file__).resolve().parents[3] / "zipdepth" / "assets" / "examples" / "clip.mp4",
    )
    if path.is_file()
]


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Server settings for the click-to-track app."""

    port: int = 7870
    """Port to serve on."""
    root_path: str = ""
    """Root path when mounted under a reverse-proxy subpath (empty when served at ``/``)."""
    variant: Sam2Variant = "efficienttam-s-512"
    """SAM2-family checkpoint. -s (Kineo's pick) holds a point-seeded track through the clip where -ti drifts."""


@dataclass(slots=True)
class Session:
    """Per-browser-session state: the tracker, its recording, and the viewer's current frame."""

    tracker: ClickTracker
    """Clip-local prompts, decoder, and SAM memory state."""
    video_path: Path
    """Input video used by the live and downloadable recordings."""
    recording_id: str
    """Rerun recording ID and session-table key."""
    frame_timestamps_ns: Int[ndarray, "n"]
    """Viewer timestamps returned by the logged video stream."""
    current_frame: int = 0
    """Frame under the viewer cursor: 0 at load (the viewer starts paused there), then updated by time_update."""
    last_time_update: float = 0.0
    """``time.monotonic()`` of the last time_update, so a click can wait for the scrub to go quiet."""
    last_stamp: float = -1.0
    """Browser ``performance.now()`` of the newest time_update applied; unqueued requests can finish out of order."""
    playing: bool = False
    """Set by the viewer's play/pause events; clicks while playing have no known frame and are refused."""
    preview_visible: bool = False
    """A static preview mask is currently shown on ``video/preview``."""
    masked_frames: set[int] = field(default_factory=set)
    """Frames that carry a mask; anything else must show none (latest-at would otherwise keep the last one)."""
    pointed_frames: set[int] = field(default_factory=set)
    """Frames that carry temporal point data."""
    tracked_frames: dict[int, TrackedFrame] = field(default_factory=dict)
    """Completed or in-progress Track outputs kept on CPU for RRD export."""


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    """One tracked frame retained for the downloadable recording."""

    mask_bits: UInt8[ndarray, "n"]
    """Binary object mask, ``np.packbits`` of the flattened h*w mask (8x smaller than a byte per pixel)."""
    score: float
    """SAM decoder predicted IoU."""
    object_score: float
    """Sigmoid object-presence score."""


_SESSIONS: dict[str, Session] = {}
_SESSIONS_LOCK = threading.Lock()  # no annotation: beartype rejects the builtin lock type hint
VARIANT: Sam2Variant = "efficienttam-s-512"


@functools.cache
def _predictor(variant: Sam2Variant):
    """Load one SAM2-streaming predictor per variant."""
    return Sam2VideoSegmenterConfig(variant=variant).setup().predictor


def _blueprint() -> rrb.Blueprint:
    """Viewer layout shared by the live stream and downloadable recording."""
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial2DView(name="Click to track", origin=VIDEO),
            rrb.TimeSeriesView(name="Confidence", origin=VIDEO),
            row_shares=[4, 1],
        ),
        rrb.BlueprintPanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
        rrb.TimePanel(state="expanded", timeline=TIMELINE, play_state=rrb.components.PlayState.Paused),
    )


def _store_session(session: Session, previous_recording_id: str | None) -> None:
    """Replace this browser's old session and evict the oldest excess sessions."""
    removed: list[Session] = []
    with _SESSIONS_LOCK:
        if previous_recording_id and previous_recording_id != session.recording_id:
            previous: Session | None = _SESSIONS.pop(previous_recording_id, None)
            if previous is not None:
                removed.append(previous)
        _SESSIONS[session.recording_id] = session
        while len(_SESSIONS) > MAX_SESSIONS:
            oldest_id: str = next(iter(_SESSIONS))
            removed.append(_SESSIONS.pop(oldest_id))
    for old_session in removed:
        _close_session(old_session)


def _close_session(session: Session) -> None:
    """Release a session's decoder thread and its downloadable recording."""
    session.tracker.close()
    (RRD_DIR / f"{session.recording_id}.rrd").unlink(missing_ok=True)


def _drop_session(recording_id: str | None) -> None:
    """Close and remove one session if it still exists."""
    with _SESSIONS_LOCK:
        session: Session | None = _SESSIONS.pop(recording_id or "", None)
    if session is not None:
        _close_session(session)


def _rec(session: Session) -> rr.RecordingStream:
    return rr.RecordingStream(application_id=APP_ID, recording_id=session.recording_id)


def _viewer_time_s(session: Session, frame_idx: int) -> float:
    """The viewer's ``video_time`` for a frame — from the logged video stream, never torchcodec pts.

    A clip cut with ``-ss`` carries an edit-list offset (torchcodec frame 0 at 0.033 s
    while the stream's frame 0 is at 0.0), so overlays stamped with decoder pts
    land after the cursor and only show up later.
    """
    return float(session.frame_timestamps_ns[frame_idx]) / 1e9


def _clear_preview(rec: rr.RecordingStream, session: Session) -> None:
    """Remove the static scrub preview (no-op when none is shown)."""
    if session.preview_visible:
        rec.log(f"{VIDEO}/preview", rr.Clear(recursive=False), static=True)
        session.preview_visible = False


def _bound_next_frame(rec: rr.RecordingStream, session: Session, entity: str, frame_idx: int, frames_with_data: set[int]) -> None:
    """Rerun shows the latest logged value until the next one: clear the entity on the following frame unless it has its own data."""
    nxt: int = frame_idx + 1
    if nxt < session.tracker.num_frames and nxt not in frames_with_data:
        rec.set_time(TIMELINE, duration=_viewer_time_s(session, nxt))
        rec.log(entity, rr.Clear(recursive=False))


def _log_points(rec: rr.RecordingStream, session: Session, frame_idx: int) -> None:
    """Log the frame's points (or a clear) at that frame's time so markers show only where they belong."""
    entity: str = f"{VIDEO}/points"
    rec.set_time(TIMELINE, duration=_viewer_time_s(session, frame_idx))
    points: list[Point] = session.tracker.points_on(frame_idx)
    if not points:
        session.pointed_frames.discard(frame_idx)
        rec.log(entity, rr.Clear(recursive=False))
        return
    rec.log(
        entity,
        rr.Points2D(
            np.asarray([[p.x, p.y] for p in points], dtype=np.float32),
            colors=[POSITIVE_COLOR if p.positive else NEGATIVE_COLOR for p in points],
            radii=6,
            labels=["+" if p.positive else "−" for p in points],
        ),
    )
    session.pointed_frames.add(frame_idx)
    _bound_next_frame(rec, session, entity, frame_idx, session.pointed_frames)


def _log_mask(rec: rr.RecordingStream, session: Session, mask_hw: UInt8[ndarray, "h w"] | None, frame_idx: int, *, bound: bool = True) -> None:
    """Log the mask at its frame (or clear it) on the ``video_time`` timeline.

    ``bound`` writes the next-frame Clear that keeps a lone mask on its own frame;
    dense tracking skips it because the next frame gets its own mask anyway.
    """
    entity: str = f"{VIDEO}/mask"
    rec.set_time(TIMELINE, duration=_viewer_time_s(session, frame_idx))
    if mask_hw is None:
        session.masked_frames.discard(frame_idx)
        rec.log(entity, rr.Clear(recursive=False))
        return
    rec.log(entity, rr.SegmentationImage(mask_hw))
    session.masked_frames.add(frame_idx)
    if bound:
        _bound_next_frame(rec, session, entity, frame_idx, session.masked_frames)


def _mask_hw(result: MaskResult | None) -> UInt8[ndarray, "h w"] | None:
    """One GPU→CPU copy of a result's mask, shared by live logging and the export cache."""
    return None if result is None else result.mask.cpu().numpy().astype(np.uint8)


def _log_confidence(rec: rr.RecordingStream, session: Session, result: MaskResult) -> None:
    """Log decoder IoU and object-presence confidence at one frame."""
    rec.set_time(TIMELINE, duration=_viewer_time_s(session, result.frame_idx))
    rec.log(f"{VIDEO}/confidence", rr.Scalars(result.score))
    rec.log(f"{VIDEO}/object_score", rr.Scalars(result.object_score))


def _remember_result(session: Session, result: MaskResult, mask_hw: UInt8[ndarray, "h w"]) -> None:
    """Keep one result on the CPU (bit-packed) for the completed downloadable recording."""
    session.tracked_frames[result.frame_idx] = TrackedFrame(
        mask_bits=np.packbits(mask_hw.reshape(-1)),
        score=result.score,
        object_score=result.object_score,
    )


def _invalidate_track(rec: rr.RecordingStream, session: Session) -> bool:
    """Clear propagated overlays and confidence after a point edit."""
    if not session.tracked_frames:
        return False
    prompted: set[int] = set(session.tracker.prompted_frames())
    for frame_idx in sorted(session.masked_frames - prompted):
        rec.set_time(TIMELINE, duration=_viewer_time_s(session, frame_idx))
        rec.log(f"{VIDEO}/mask", rr.Clear(recursive=False))
    for frame_idx in sorted(session.tracked_frames):
        rec.set_time(TIMELINE, duration=_viewer_time_s(session, frame_idx))
        rec.log(f"{VIDEO}/confidence", rr.Clear(recursive=False))
        rec.log(f"{VIDEO}/object_score", rr.Clear(recursive=False))
    session.masked_frames = prompted
    session.tracked_frames.clear()
    return True


def _save_recording(session: Session) -> Path:
    """Write the video, prompts, tracked masks, and confidence to one RRD."""
    RRD_DIR.mkdir(parents=True, exist_ok=True)
    output: Path = RRD_DIR / f"{session.recording_id}.rrd"
    rec: rr.RecordingStream = rr.RecordingStream(application_id=APP_ID, recording_id=session.recording_id)
    rec.save(output, default_blueprint=_blueprint())
    rec.log(
        VIDEO,
        rr.AnnotationContext(
            [
                rr.AnnotationInfo(id=0, label="background", color=(0, 0, 0, 0)),
                rr.AnnotationInfo(id=1, label="object", color=MASK_COLOR),
            ]
        ),
        static=True,
    )
    log_video(session.video_path, Path(VIDEO), timeline=TIMELINE, recording=rec)
    prompted: set[int] = set(session.tracker.prompted_frames())
    for frame_idx in sorted(prompted):
        rec.set_time(TIMELINE, duration=_viewer_time_s(session, frame_idx))
        points: list[Point] = session.tracker.points_on(frame_idx)
        rec.log(
            f"{VIDEO}/points",
            rr.Points2D(
                np.asarray([[point.x, point.y] for point in points], dtype=np.float32),
                colors=[POSITIVE_COLOR if point.positive else NEGATIVE_COLOR for point in points],
                radii=6,
                labels=["+" if point.positive else "−" for point in points],
            ),
        )
        nxt: int = frame_idx + 1
        if nxt < session.tracker.num_frames and nxt not in prompted:
            rec.set_time(TIMELINE, duration=_viewer_time_s(session, nxt))
            rec.log(f"{VIDEO}/points", rr.Clear(recursive=False))
    for frame_idx, tracked in sorted(session.tracked_frames.items()):
        rec.set_time(TIMELINE, duration=_viewer_time_s(session, frame_idx))
        h, w = session.tracker.frame_hw
        mask_hw: UInt8[ndarray, "h w"] = np.unpackbits(tracked.mask_bits)[: h * w].reshape(h, w)
        rec.log(f"{VIDEO}/mask", rr.SegmentationImage(mask_hw))
        rec.log(f"{VIDEO}/confidence", rr.Scalars(tracked.score))
        rec.log(f"{VIDEO}/object_score", rr.Scalars(tracked.object_score))
    rec.disconnect()
    return output


def frame_at(frame_timestamps_ns: Int[ndarray, "n"], time_ns: float) -> int:
    """The frame the viewer shows at a time: the last frame whose pts <= t (latest-at), like the viewer itself.

    The viewer reports duration timelines in nanoseconds.
    """
    idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right")) - 1
    return max(0, min(idx, int(frame_timestamps_ns.shape[0]) - 1))


def _status(session: Session, prefix: str) -> str:
    frames: list[int] = session.tracker.prompted_frames()
    return f"{prefix} {len(session.tracker.points)} point(s) on frame(s) {frames} · viewer at frame {session.current_frame}."


# ── callbacks ─────────────────────────────────────────────────────────────


def load_video(video: str | None, previous_recording_id: str | None) -> Iterator[tuple[bytes | None, str, str, str, Any, Any]]:
    """Start a fresh session + recording for the clip (generators only: the streaming viewer rejects plain bytes)."""
    _drop_session(previous_recording_id)
    if video is None:
        yield None, "", "Upload a video to begin.", "Upload a video to begin.", gr.Tabs(selected="input"), gr.DownloadButton(visible=False)
        return
    recording_id: str = str(uuid.uuid4())
    video_path: Path = Path(video)
    tracker = ClickTracker(video_path, _predictor(VARIANT))
    rec = rr.RecordingStream(application_id=APP_ID, recording_id=recording_id)
    stream: rr.BinaryStream = rec.binary_stream()
    rec.send_blueprint(_blueprint())
    rec.log(
        VIDEO,
        rr.AnnotationContext([rr.AnnotationInfo(id=0, label="background", color=(0, 0, 0, 0)), rr.AnnotationInfo(id=1, label="object", color=MASK_COLOR)]),
        static=True,
    )
    timestamps_ns = log_video(video_path, Path(VIDEO), timeline=TIMELINE, recording=rec)
    session = Session(tracker=tracker, video_path=video_path, recording_id=recording_id, frame_timestamps_ns=timestamps_ns)
    _store_session(session, None)
    stream.flush()
    yield (
        stream.read(),
        recording_id,
        "Click the object on frame 0, or scrub (drag / ← →) to another frame first. Add − to exclude; Remove deletes a point.",
        "Click the object on frame 0, or scrub (drag / ← →) to another frame first. Add − to exclude; Remove deletes a point.",
        gr.Tabs(selected="input"),
        gr.DownloadButton(visible=False),
    )


def record_time(recording_id: str | None, stamp: float | None, evt: TimeUpdate) -> None:
    """Record the viewer's frame instantly (no queue, no outputs) so a click right after a scrub reads the right frame.

    ``stamp`` is the browser's ``performance.now()`` at dispatch (see the ``js`` hook): unqueued
    requests can complete out of order, so an older event must never overwrite a newer one.
    """
    session: Session | None = _SESSIONS.get(recording_id or "")
    if session is None:
        return
    stamp_value: float = float(stamp) if stamp is not None else session.last_stamp + 1.0
    if stamp_value < session.last_stamp:
        return
    session.last_stamp = stamp_value
    session.current_frame = frame_at(session.frame_timestamps_ns, float(evt.payload.time))
    session.last_time_update = time.monotonic()


def on_play(recording_id: str | None, evt: Play) -> None:
    """Playback emits no time_update: remember that the frame is unknown until the viewer pauses."""
    session: Session | None = _SESSIONS.get(recording_id or "")
    if session is not None:
        session.playing = True


def on_pause(recording_id: str | None, evt: Pause) -> None:
    """The viewer paused; the next time_update (or the pre-play frame) is authoritative again."""
    session: Session | None = _SESSIONS.get(recording_id or "")
    if session is not None:
        session.playing = False


def on_time_update(recording_id: str | None, evt: TimeUpdate) -> Iterator[tuple[bytes | None, str, str]]:
    """When prompts exist, preview the memory-conditioned mask on the frame under the cursor."""
    session: Session | None = _SESSIONS.get(recording_id or "")
    if session is None:
        yield b"", gr.skip(), gr.skip()
        return
    if session.playing:
        # Playback emits ~10 time updates a second; previewing and re-writing the
        # status on each would spam the UI and the GPU. Just drop any preview.
        if not session.preview_visible:
            yield b"", gr.skip(), gr.skip()
            return
        rec = _rec(session)
        stream: rr.BinaryStream = rec.binary_stream()
        _clear_preview(rec, session)
        stream.flush()
        yield stream.read(), gr.skip(), gr.skip()
        return
    # This handler is queued (always_last) and may run late; it must never write
    # session.current_frame — only the unqueued record_time does, or a backlog of
    # previews would overwrite the frame with stale times right before a click.
    frame_idx: int = frame_at(session.frame_timestamps_ns, float(evt.payload.time))
    frame_text: str = f"Viewer at frame {frame_idx} — a click prompts this frame."
    # The preview is *static* (no timeline → no ticks, overwritten in place), so it
    # must go away the moment the cursor leaves the frame it was computed for.
    rec = _rec(session)
    stream: rr.BinaryStream = rec.binary_stream()
    _clear_preview(rec, session)
    stamp: float = session.last_stamp
    wants_preview: bool = bool(session.tracker.points) and not session.tracker.points_on(frame_idx)
    if wants_preview:
        time.sleep(0.15)  # rest-throttle: skip frames the cursor only passed through
    if wants_preview and session.last_stamp == stamp:
        result: MaskResult | None = session.tracker.preview(frame_idx)
        if result is not None and session.last_stamp == stamp and not session.playing:  # discard late results
            rec.log(f"{VIDEO}/preview", rr.SegmentationImage(result.mask.cpu().numpy().astype(np.uint8)), static=True)
            session.preview_visible = True
    stream.flush()
    yield stream.read(), frame_text, frame_text


def on_select(recording_id: str | None, mode: str, resegment: bool, evt: SelectionChange) -> Iterator[tuple[bytes | None, str, str]]:
    """A click on the video at the current frame: add a +/− point or remove the nearest one, then show the mask."""
    # The viewer fires selection_change on recording open and on every click
    # anywhere; ignored selections must still yield a chunk for the streaming
    # viewer output (gr.skip() there crashes Gradio's end_stream).
    # A click on a masked pixel selects both /video/mask and /video, so take the
    # first positioned entity under the video rather than demanding exactly one.
    session: Session | None = _SESSIONS.get(recording_id or "")
    hits = [i for i in evt.payload.items if i.type == "entity" and i.position is not None and i.entity_path.startswith(f"/{VIDEO}")]
    if session is None or not hits:
        yield b"", gr.skip(), gr.skip()
        return
    position: list[float] | None = hits[0].position
    assert position is not None
    if session.playing:
        raise gr.Error("Pause the viewer first — while it plays, the click's frame is unknown.")
    x, y = float(position[0]), float(position[1])
    # The viewer sends the final time_update of a scrub just before the click; wait for the stream to go quiet.
    deadline: float = time.monotonic() + 0.4
    while time.monotonic() - session.last_time_update < 0.12 and time.monotonic() < deadline:
        time.sleep(0.02)
    frame_idx: int = session.current_frame
    rec = _rec(session)
    stream: rr.BinaryStream = rec.binary_stream()
    _clear_preview(rec, session)
    changed: bool = True
    if mode == "✕ Remove":
        removed, result = session.tracker.remove_point_near(frame_idx, x, y)
        changed = removed is not None
        prefix: str = "Removed a point." if changed else f"No point within {session.tracker.remove_radius_px:.0f}px on this frame."
    else:
        positive: bool = mode != "− Exclude"
        result = session.tracker.add_point(frame_idx, x, y, positive=positive, resegment=resegment)
        action: str = "Re-segmented +" if resegment else ("Added +" if positive else "Added −")
        prefix = f"{action} at ({x:.0f}, {y:.0f}) on frame {frame_idx}."
    stale: bool = _invalidate_track(rec, session) if changed else False
    _log_mask(rec, session, _mask_hw(result), frame_idx)
    _log_points(rec, session, frame_idx)
    stream.flush()
    if stale:
        prefix += " The previous track is stale — Track again."
    status_text: str = _status(session, prefix)
    yield stream.read(), status_text, status_text


def undo(recording_id: str | None) -> Iterator[tuple[bytes | None, str, str]]:
    """Remove the most recently added point."""
    session: Session | None = _SESSIONS.get(recording_id or "")
    if session is None:
        raise gr.Error("Upload a video first.")
    rec = _rec(session)
    stream: rr.BinaryStream = rec.binary_stream()
    _clear_preview(rec, session)
    last, result = session.tracker.undo()
    if last is None:
        status_text: str = _status(session, "Nothing to undo.")
        yield b"", status_text, status_text
        return
    stale: bool = _invalidate_track(rec, session)
    _log_mask(rec, session, _mask_hw(result), last.frame_idx)
    _log_points(rec, session, last.frame_idx)
    stream.flush()
    prefix: str = f"Undid the point on frame {last.frame_idx}."
    if stale:
        prefix += " The previous track is stale — Track again."
    status_text = _status(session, prefix)
    yield stream.read(), status_text, status_text


def clear(recording_id: str | None) -> Iterator[tuple[bytes | None, str, str]]:
    """Drop every point, every memory, and every overlay."""
    session: Session | None = _SESSIONS.get(recording_id or "")
    if session is None:
        raise gr.Error("Upload a video first.")
    rec = _rec(session)
    stream: rr.BinaryStream = rec.binary_stream()
    _clear_preview(rec, session)
    session.tracker.clear()
    stale: bool = _invalidate_track(rec, session)
    # Temporal clears only: a static Clear would shadow every later temporal log on the entity.
    for entity, frames in ((f"{VIDEO}/points", session.pointed_frames), (f"{VIDEO}/mask", session.masked_frames)):
        for frame_idx in sorted(frames):
            rec.set_time(TIMELINE, duration=_viewer_time_s(session, frame_idx))
            rec.log(entity, rr.Clear(recursive=False))
        frames.clear()
    stream.flush()
    suffix: str = " The previous track is stale — add a point and Track again." if stale else ""
    status_text: str = f"Cleared all points and memory.{suffix}"
    yield stream.read(), status_text, status_text


def track(recording_id: str | None) -> Iterator[tuple[bytes | None, str, str, Any, Any]]:
    """Propagate in both directions, stream masks, and write the downloadable RRD."""
    session: Session | None = _SESSIONS.get(recording_id or "")
    if session is None:
        raise gr.Error("Upload a video first.")
    if not session.tracker.points:
        raise gr.Error("Click at least one point on the object first.")
    rec = _rec(session)
    stream: rr.BinaryStream = rec.binary_stream()
    _clear_preview(rec, session)
    _invalidate_track(rec, session)
    stream.flush()
    status_text: str = "Tracking from the prompted frame in both directions…"
    yield stream.read(), status_text, status_text, gr.Tabs(selected="outputs"), gr.DownloadButton(visible=False)
    done: int = 0
    low_score: int = 0
    for result in session.tracker.track():
        mask_hw: UInt8[ndarray, "h w"] | None = _mask_hw(result)
        assert mask_hw is not None
        # Dense output: every frame gets its own mask, so no next-frame bounds, and
        # points were already logged (and bounded) when they were placed.
        _log_mask(rec, session, mask_hw, result.frame_idx, bound=False)
        _log_confidence(rec, session, result)
        _remember_result(session, result, mask_hw)
        done += 1
        low_score += int(result.score < 0.5)
        if done % 30 == 0:
            stream.flush()
            yield (
                stream.read(),
                f"Tracking… {done}/{session.tracker.num_frames} frames",
                f"Tracking… {done}/{session.tracker.num_frames} frames",
                gr.skip(),
                gr.skip(),
            )
    output: Path = _save_recording(session)
    stream.flush()
    yield (
        stream.read(),
        f"Done: tracked all {done} frames; {low_score} low-confidence frame(s). Add a corrective click where needed, then Track again.",
        f"Done: tracked all {done} frames; {low_score} low-confidence frame(s). Add a corrective click where needed, then Track again.",
        gr.Tabs(selected="outputs"),
        gr.DownloadButton(value=str(output), visible=True),
    )


def stop_tracking() -> tuple[str, str]:
    """Describe the consistent state left by Gradio cancelling Track at a yield."""
    status: str = "Stopped. The partial result is not downloadable; press Track to run again from scratch."
    return status, status


DESCRIPTION: str = (
    "# posekit: Click to Track\n"
    "Click an object in the Rerun viewer, refine it on any frame, and propagate the mask through the whole clip. "
    "The confidence traces help you find frames that need another click."
)

VIEWER_HEIGHT: str = "calc(100vh - 7rem)"
"""Viewer frame height: the viewport minus the header/description band, so nothing scrolls."""

# HF Spaces render the default theme in dark mode; do the same here by forcing
# Gradio's ``__theme=dark`` URL switch before the app mounts (a <head> script —
# the launch ``js`` hook runs too late to redirect).
FORCE_DARK_HEAD: str = """
<script>
(() => {
    const url = new URL(window.location);
    if (url.searchParams.get("__theme") !== "dark") {
        url.searchParams.set("__theme", "dark");
        window.location.replace(url.href);
    }
})();
</script>
"""

APP_CSS: str = """
html, body, gradio-app, .gradio-container {
    height: 100%;
    overflow: hidden;
}
.gradio-container {
    max-width: none !important;
    padding: 0.6rem 1rem !important;
}
#app-description { margin-bottom: 0.35rem; }
#app-description h1 { margin: 0 0 0.2rem; }
#app-description p { margin: 0; }
#main-row {
    height: calc(100vh - 6.4rem);
    min-height: 0;
    overflow: hidden;
}
#left-column, #viewer-column { min-height: 0; }
#source-video { height: auto !important; }
#source-video video { max-height: 225px !important; }
#examples { flex: none; }
/* Click-mode radio as a segmented pill group. */
#click-mode .wrap { display: flex; gap: 0; border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg); overflow: hidden; }
#click-mode label { flex: 1; justify-content: center; margin: 0; border-radius: 0; border: 0; background: var(--background-fill-secondary); padding: 0.45rem 0.4rem; cursor: pointer; }
#click-mode label + label { border-left: 1px solid var(--border-color-primary); }
#click-mode label.selected { background: var(--color-accent); color: white; }
#click-mode input { display: none; }
#click-mode span { font-weight: 600; }
/* Four small tiles per row with a pager, like the 4DAnyone Space. */
#examples .gallery { display: flex; flex-wrap: wrap; gap: 0.4rem; }
#examples .gallery button { flex: none; padding: 0; }
#examples video, #examples img { height: 64px !important; width: auto !important; max-width: 84px; object-fit: cover; }
/* The viewer's top edge lines up with the video card beside it. */
#viewer-column { padding-top: 0; }
#rerun-viewer { min-height: 0 !important; }
#run-status {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 4.5rem;
    overflow: visible !important;
    padding: 0.65rem 0.85rem;
    border-radius: var(--radius-lg);
    background: var(--background-fill-secondary);
}
#run-status p { font-size: 1.05rem; line-height: 1.4; margin: 0; }
footer { display: none !important; }
"""


with gr.Blocks(title="posekit: Click to Track") as demo:
    gr.Markdown(DESCRIPTION, elem_id="app-description")
    recording_state: gr.State = gr.State("")
    stamp_in: gr.Number = gr.Number(value=0.0, visible=False)
    with gr.Row(elem_id="main-row"):
        with gr.Column(scale=1, elem_id="left-column"):
            # Video and status sit above the tabs: the status must stay visible when
            # Track auto-switches to Outputs, and it belongs right under the input.
            video_in: gr.Video = gr.Video(label="Video", height=240, elem_id="source-video")
            status: gr.Markdown = gr.Markdown("Upload a video to begin.", elem_id="run-status")
            with gr.Tabs(selected="input") as tabs:
                with gr.Tab("Input", id="input"):
                    # A segmented control (see #click-mode CSS): the three everyday click modes on one line.
                    mode: gr.Radio = gr.Radio(choices=MODES, value=MODES[0], show_label=False, container=False, elem_id="click-mode")
                    with gr.Row():
                        undo_btn: gr.Button = gr.Button("Undo point", size="sm")
                        clear_btn: gr.Button = gr.Button("Clear", size="sm")
                    with gr.Accordion("Advanced", open=False):
                        resegment_box: gr.Checkbox = gr.Checkbox(
                            value=False,
                            label="Clicks replace the object (re-segment) instead of refining the propagated mask",
                        )
                with gr.Tab("Outputs", id="outputs"):
                    gr.Markdown("When tracking finishes, download a self-contained Rerun recording with the video, prompts, masks, and confidence traces.")
                    download: gr.DownloadButton = gr.DownloadButton("Download the recording (.rrd)", visible=False)
            with gr.Row():
                track_btn: gr.Button = gr.Button("Track", variant="primary")
                stop_btn: gr.Button = gr.Button("Stop", variant="stop")
            status_probe: gr.Textbox = gr.Textbox(value="Upload a video to begin.", visible="hidden", elem_id="status-probe")
            # Examples last: a secondary "pick a sample" action, below the controls it feeds.
            gr.Examples(examples=EXAMPLE_VIDEOS, inputs=[video_in], cache_examples=False, examples_per_page=4, elem_id="examples")
        with gr.Column(scale=3, elem_id="viewer-column"):
            viewer: Rerun = Rerun(
                label="Click to track",
                streaming=True,
                panel_states={"time": "expanded", "blueprint": "hidden", "selection": "hidden"},
                # A CSS height keeps the viewer's own frame inside the fold; a pixel
                # height would overflow whatever the wrapper is sized to.
                height=VIEWER_HEIGHT,
                elem_id="rerun-viewer",
            )

    video_in.change(
        fn=load_video,
        inputs=[video_in, recording_state],
        outputs=[viewer, recording_state, status, status_probe, tabs, download],
        api_visibility="private",
    )
    # always_last coalesces the flood of time updates while scrubbing/playing into "the latest one".
    viewer.time_update(
        fn=record_time,
        inputs=[recording_state, stamp_in],
        outputs=None,
        queue=False,
        trigger_mode="multiple",
        js="(rid, _stamp) => [rid, performance.now()]",
        api_visibility="private",
    )
    viewer.play(fn=on_play, inputs=[recording_state], outputs=None, queue=False, trigger_mode="multiple", api_visibility="private")
    viewer.pause(fn=on_pause, inputs=[recording_state], outputs=None, queue=False, trigger_mode="multiple", api_visibility="private")
    viewer.time_update(
        fn=on_time_update,
        inputs=[recording_state],
        outputs=[viewer, status, status_probe],
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="private",
    )
    viewer.selection_change(
        fn=on_select,
        inputs=[recording_state, mode, resegment_box],
        outputs=[viewer, status, status_probe],
        show_progress="hidden",
        api_visibility="private",
    )
    undo_btn.click(fn=undo, inputs=[recording_state], outputs=[viewer, status, status_probe], api_visibility="private")
    clear_btn.click(fn=clear, inputs=[recording_state], outputs=[viewer, status, status_probe], api_visibility="private")
    track_event = track_btn.click(fn=track, inputs=[recording_state], outputs=[viewer, status, status_probe, tabs, download])
    stop_btn.click(fn=stop_tracking, outputs=[status, status_probe], cancels=[track_event], queue=False)


# The embedded viewer stream needs a secure context: on the tailnet use
# `tailscale serve --bg --https=<port> http://127.0.0.1:<port>` (served at `/`, so root_path stays empty).
def launch(config: AppConfig) -> None:
    """Launch the click-to-track Gradio app."""
    global VARIANT
    VARIANT = config.variant  # read by load_video, which caches one predictor per variant
    demo.launch(
        server_port=config.port,
        root_path=config.root_path,
        allowed_paths=sorted({str(Path(example).parent) for example in EXAMPLE_VIDEOS} | {str(RRD_DIR)}),
        # Default theme (orange primary / red stop), forced dark — same look as the 4DAnyone Space.
        css=APP_CSS,
        head=FORCE_DARK_HEAD,
        # Surface every callback exception as a gr.Error toast with its message.
        show_error=True,
    )


if __name__ == "__main__":
    launch(tyro.cli(AppConfig))
