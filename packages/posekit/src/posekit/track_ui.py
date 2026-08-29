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
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias, get_args

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
from simplecv.video_utils import frame_at

from posekit.apis.click_tracker import ClickTracker, MaskResult, Point, PointEdit
from posekit.models.sam2_video import Sam2Variant, Sam2VideoSegmenterConfig

APP_ID: str = "posekit_click_to_track"
VIDEO: str = "video"
TIMELINE: str = "video_time"
Mode: TypeAlias = Literal["+ Include", "− Exclude", "✕ Remove"]
MODES: list[Mode] = ["+ Include", "− Exclude", "✕ Remove"]
TabName: TypeAlias = Literal["Input", "Config", "Outputs"]
RecordingPair: TypeAlias = tuple[rr.RecordingStream, rr.BinaryStream]
POSITIVE_COLOR: tuple[int, int, int] = (76, 220, 96)
NEGATIVE_COLOR: tuple[int, int, int] = (255, 87, 51)
MASK_COLOR: tuple[int, int, int] = (51, 178, 255)
MASK_SCALE: int = 4
"""Masks are logged at 1/MASK_SCALE resolution under a static scale transform: a full-res
uint8 mask per frame costs ~40x the video in viewer memory, and SAM2's logits are 128^2 anyway."""
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
DEFAULT_VARIANT: Sam2Variant = "efficienttam-s-512"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Server settings for the click-to-track app."""

    port: int = 7870
    """Port to serve on."""
    root_path: str = ""
    """Root path when mounted under a reverse-proxy subpath (empty when served at ``/``)."""
    variant: Sam2Variant = DEFAULT_VARIANT
    """SAM2-family checkpoint. -s (Kineo's pick) holds a point-seeded track through the clip where -ti drifts."""


@dataclass(slots=True)
class Session:
    """Per-browser-session state: the tracker, its recording, and the viewer's current frame."""

    tracker: ClickTracker
    """Clip-local prompts, decoder, and SAM memory state."""
    video_path: Path
    """Input video used by the live and downloadable recordings."""
    recording_id: str
    """Rerun recording ID."""
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
    tracked_frame_indices: set[int] = field(default_factory=set)
    """Frames written by the latest Track run, retained only for invalidation."""
    rrd_part_counter: int = 0
    """Monotonic callback-part number within this session."""


SAM2_VARIANTS: list[Sam2Variant] = list(get_args(Sam2Variant))
# The interactive tracker favors a longer window than Sam2VideoSegmenterConfig's streaming default of 7.
DEFAULT_MEMORY_WINDOW: int = 10
DEFAULT_REMOVE_RADIUS_PX: float = 100.0


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


def _close_session(session: Session | None) -> None:
    """Release a session's decoder thread and its downloadable recording."""
    if session is None:
        return
    session.tracker.close()
    (RRD_DIR / f"{session.recording_id}.rrd").unlink(missing_ok=True)
    shutil.rmtree(RRD_DIR / session.recording_id, ignore_errors=True)


def _open_recording(session: Session, *, write_part: bool = True) -> tuple[rr.RecordingStream, rr.BinaryStream]:
    """Open one callback recording with a browser stream and, except for previews, a footerless file part."""
    rec: rr.RecordingStream = rr.RecordingStream(application_id=APP_ID, recording_id=session.recording_id)
    stream: rr.BinaryStream = rec.binary_stream()
    if write_part:
        part_dir: Path = RRD_DIR / session.recording_id
        part_dir.mkdir(parents=True, exist_ok=True)
        part: Path = part_dir / f"{session.rrd_part_counter:04d}.rrd"
        session.rrd_part_counter += 1
        rec.set_sinks(stream, rr.FileSink(str(part), write_footer=False))
    return rec, stream


def _merge_rrd_parts(session: Session) -> Path:
    """Merge this session's footerless callback parts into one valid, compacted RRD.

    The parts are hundreds of tiny per-callback chunks; ``rrd optimize`` re-batches
    them into fewer, larger chunks and re-derives video keyframes so the download loads fast.
    """
    output: Path = RRD_DIR / f"{session.recording_id}.rrd"
    merged: Path = RRD_DIR / f"{session.recording_id}.merged.rrd"
    parts: list[Path] = sorted((RRD_DIR / session.recording_id).glob("*.rrd"))
    subprocess.run(["rerun", "rrd", "merge", "--output", str(merged), *(str(part) for part in parts)], check=True)
    # --fix-keyframe: the Mp4Reader stream logs is_keyframe=false rows, which blocks GoP rebatching of the video.
    subprocess.run(["rerun", "rrd", "optimize", "--fix-keyframe", "--output", str(output), str(merged)], check=True)
    merged.unlink()
    return output


def _viewer_time_s(session: Session, frame_idx: int) -> float:
    """The viewer's ``video_time`` for a frame — from the logged video stream, never torchcodec pts.

    A clip cut with ``-ss`` carries an edit-list offset (torchcodec frame 0 at 0.033 s
    while the stream's frame 0 is at 0.0), so overlays stamped with decoder pts
    land after the cursor and only show up later.
    """
    return float(session.frame_timestamps_ns[frame_idx]) / 1e9


def _clear_preview(rec: rr.RecordingStream, session: Session) -> None:
    """Remove the static scrub preview (no-op when none is shown).

    A native Rerun 0.36.2 pixel check confirms that this same-entity static
    ``Clear`` leaves ``video/preview``'s static scale transform in effect.
    """
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
    points: tuple[Point, ...] = session.tracker.points_on(frame_idx)
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


def _mask_hw(result: MaskResult) -> UInt8[ndarray, "h w"]:
    """Copy one result mask from the inference device, downsampled for Rerun logging.

    Strided sampling rounds each output dimension up, so a non-multiple of
    ``MASK_SCALE`` can overhang the full-resolution image by up to three pixels.
    """
    return result.mask[::MASK_SCALE, ::MASK_SCALE].cpu().numpy().astype(np.uint8)


def _log_confidence(rec: rr.RecordingStream, session: Session, result: MaskResult) -> None:
    """Log decoder IoU and object-presence confidence at one frame."""
    rec.set_time(TIMELINE, duration=_viewer_time_s(session, result.frame_idx))
    rec.log(f"{VIDEO}/confidence", rr.Scalars(result.score))
    rec.log(f"{VIDEO}/object_score", rr.Scalars(result.object_score))


def _invalidate_track(rec: rr.RecordingStream, session: Session) -> bool:
    """Clear propagated overlays and confidence after a point edit."""
    if not session.tracked_frame_indices:
        return False
    prompted: set[int] = set(session.tracker.prompted_frames())
    for frame_idx in sorted(session.masked_frames - prompted):
        rec.set_time(TIMELINE, duration=_viewer_time_s(session, frame_idx))
        rec.log(f"{VIDEO}/mask", rr.Clear(recursive=False))
    for frame_idx in sorted(session.tracked_frame_indices):
        rec.set_time(TIMELINE, duration=_viewer_time_s(session, frame_idx))
        rec.log(f"{VIDEO}/confidence", rr.Clear(recursive=False))
        rec.log(f"{VIDEO}/object_score", rr.Clear(recursive=False))
    session.masked_frames = prompted
    session.tracked_frame_indices.clear()
    return True


def _status(session: Session, prefix: str) -> str:
    frames: tuple[int, ...] = session.tracker.prompted_frames()
    return f"{prefix} {len(session.tracker.points)} point(s) on frame(s) {frames} · viewer at frame {session.current_frame}."


# ── callbacks ─────────────────────────────────────────────────────────────


def show_control_tab(selected: TabName) -> tuple[gr.Column, gr.Column, gr.Column]:
    """Show only the controls panel selected by the tab-strip radio."""
    return (
        gr.Column(visible=selected == "Input"),
        gr.Column(visible=selected == "Config"),
        gr.Column(visible=selected == "Outputs"),
    )


def load_video(
    video: str | None,
    previous_session: Session | None,
    variant: str,
    memory_window: float,
    *,
    selected_tab: TabName = "Input",
) -> Iterator[tuple[bytes | None, Session | None, str, str, TabName, gr.DownloadButton]]:
    """Start a fresh session + recording for the clip (generators only: the streaming viewer rejects plain bytes).

    ``variant`` and ``memory_window`` come from the Config panel; both shape the
    tracker's memory state, so changing them re-runs this and starts the clip over.
    """
    if video is None:
        _close_session(previous_session)
        yield (
            None,
            None,
            "Upload a video to begin.",
            "Upload a video to begin.",
            selected_tab,
            gr.DownloadButton(visible=False),
        )
        return
    recording_id: str = str(uuid.uuid4())
    video_path: Path = Path(video)
    if variant not in SAM2_VARIANTS:
        raise gr.Error(f"Unknown model {variant!r}.")
    tracker: ClickTracker = ClickTracker(video_path, _predictor(variant), memory_window_size=int(memory_window))
    _close_session(previous_session)
    session: Session = Session(
        tracker=tracker,
        video_path=video_path,
        recording_id=recording_id,
        frame_timestamps_ns=np.empty(0, dtype=np.int64),
    )
    recording_pair: RecordingPair = _open_recording(session)
    rec: rr.RecordingStream = recording_pair[0]
    stream: rr.BinaryStream = recording_pair[1]
    rec.send_blueprint(_blueprint())
    rec.log(
        VIDEO,
        rr.AnnotationContext([rr.AnnotationInfo(id=0, label="background", color=(0, 0, 0, 0)), rr.AnnotationInfo(id=1, label="object", color=MASK_COLOR)]),
        static=True,
    )
    for entity in (f"{VIDEO}/mask", f"{VIDEO}/preview"):
        rec.log(entity, rr.Transform3D(scale=float(MASK_SCALE)), static=True)
    session.frame_timestamps_ns = log_video(
        video_path,
        Path(VIDEO),
        timeline=TIMELINE,
        recording=rec,
        output_codec=rr.VideoCodec.H264,
    )
    stream.flush()
    payload: bytes | None = stream.read()
    rec.disconnect()
    yield (
        payload,
        session,
        "Click the object on frame 0, or scrub (drag / ← →) to another frame first. Add − to exclude; Remove deletes a point.",
        "Click the object on frame 0, or scrub (drag / ← →) to another frame first. Add − to exclude; Remove deletes a point.",
        selected_tab,
        gr.DownloadButton(visible=False),
    )


def _reload_video_from_config(
    video: str | None,
    previous_session: Session | None,
    variant: str,
    memory_window: float,
) -> Iterator[tuple[bytes | None, Session | None, str, str, TabName, gr.DownloadButton]]:
    """Reload tracker configuration without moving the controls away from Config."""
    yield from load_video(video, previous_session, variant, memory_window, selected_tab="Config")


def record_time(session: Session | None, stamp: float | None, evt: TimeUpdate) -> None:
    """Record the viewer's frame instantly (no queue, no outputs) so a click right after a scrub reads the right frame.

    ``stamp`` is the browser's ``performance.now()`` at dispatch (see the ``js`` hook): unqueued
    requests can complete out of order, so an older event must never overwrite a newer one.
    """
    if session is None:
        return
    stamp_value: float = float(stamp) if stamp is not None else session.last_stamp + 1.0
    if stamp_value < session.last_stamp:
        return
    session.last_stamp = stamp_value
    session.current_frame = frame_at(session.frame_timestamps_ns, float(evt.payload.time))
    session.last_time_update = time.monotonic()


def on_play(session: Session | None, evt: Play) -> None:
    """Playback emits no time_update: remember that the frame is unknown until the viewer pauses."""
    if session is not None:
        session.playing = True


def on_pause(session: Session | None, evt: Pause) -> None:
    """The viewer paused; the next time_update (or the pre-play frame) is authoritative again."""
    if session is not None:
        session.playing = False


def on_time_update(session: Session | None, evt: TimeUpdate) -> Iterator[tuple[bytes | None, str, str]]:
    """When prompts exist, preview the memory-conditioned mask on the frame under the cursor."""
    if session is None:
        yield b"", gr.skip(), gr.skip()
        return
    if session.playing:
        # Playback emits ~10 time updates a second; previewing and re-writing the
        # status on each would spam the UI and the GPU. Just drop any preview.
        if not session.preview_visible:
            yield b"", gr.skip(), gr.skip()
            return
        recording_pair: RecordingPair = _open_recording(session, write_part=False)
        rec: rr.RecordingStream = recording_pair[0]
        stream: rr.BinaryStream = recording_pair[1]
        _clear_preview(rec, session)
        stream.flush()
        payload: bytes | None = stream.read()
        rec.disconnect()
        yield payload, gr.skip(), gr.skip()
        return
    # This handler is queued (always_last) and may run late; it must never write
    # session.current_frame — only the unqueued record_time does, or a backlog of
    # previews would overwrite the frame with stale times right before a click.
    frame_idx: int = frame_at(session.frame_timestamps_ns, float(evt.payload.time))
    frame_text: str = f"Viewer at frame {frame_idx} — a click prompts this frame."
    # The preview is *static* (no timeline → no ticks, overwritten in place), so it
    # must go away the moment the cursor leaves the frame it was computed for.
    recording_pair = _open_recording(session, write_part=False)
    rec = recording_pair[0]
    stream = recording_pair[1]
    _clear_preview(rec, session)
    stamp: float = session.last_stamp
    wants_preview: bool = bool(session.tracker.points) and not session.tracker.points_on(frame_idx)
    if wants_preview:
        time.sleep(0.15)  # rest-throttle: skip frames the cursor only passed through
    if wants_preview and session.last_stamp == stamp:
        result: MaskResult | None = session.tracker.preview(frame_idx)
        if result is not None and session.last_stamp == stamp and not session.playing:  # discard late results
            preview_hw: UInt8[ndarray, "h w"] = _mask_hw(result)
            rec.log(f"{VIDEO}/preview", rr.SegmentationImage(preview_hw), static=True)
            session.preview_visible = True
    stream.flush()
    payload = stream.read()
    rec.disconnect()
    yield payload, frame_text, frame_text


def on_select(
    session: Session | None, mode: str, resegment: bool, remove_radius_px: float, evt: SelectionChange
) -> Iterator[tuple[bytes | None, str, str]]:
    """A click on the video at the current frame: add a +/− point or remove the nearest one, then show the mask."""
    # The viewer fires selection_change on recording open and on every click
    # anywhere; ignored selections must still yield a chunk for the streaming
    # viewer output (gr.skip() there crashes Gradio's end_stream).
    # A masked pixel can report both the scaled overlay and the source video.
    # Only the source entity's hit is guaranteed to use video pixel coordinates.
    position: list[float] | None = next(
        (item.position for item in evt.payload.items if item.type == "entity" and item.entity_path == f"/{VIDEO}" and item.position is not None),
        None,
    )
    if session is None or position is None:
        yield b"", gr.skip(), gr.skip()
        return
    if session.playing:
        raise gr.Error("Pause the viewer first — while it plays, the click's frame is unknown.")
    position_xy: tuple[float, float] = (float(position[0]), float(position[1]))
    x: float = position_xy[0]
    y: float = position_xy[1]
    # The viewer sends the final time_update of a scrub just before the click; wait for the stream to go quiet.
    deadline: float = time.monotonic() + 0.4
    while time.monotonic() - session.last_time_update < 0.12 and time.monotonic() < deadline:
        time.sleep(0.02)
    frame_idx: int = session.current_frame
    recording_pair: RecordingPair = _open_recording(session)
    rec: rr.RecordingStream = recording_pair[0]
    stream: rr.BinaryStream = recording_pair[1]
    _clear_preview(rec, session)
    changed: bool = True
    if mode == "✕ Remove":
        edit: PointEdit = session.tracker.remove_point_near(frame_idx, x, y, radius_px=float(remove_radius_px))
        changed = edit.point is not None
        result: MaskResult | None = edit.result
        prefix: str = "Removed a point." if changed else f"No point within {remove_radius_px:.0f}px on this frame."
    else:
        positive: bool = mode != "− Exclude"
        result = session.tracker.add_point(frame_idx, x, y, positive=positive, resegment=resegment)
        action: str = "Re-segmented +" if resegment else ("Added +" if positive else "Added −")
        prefix = f"{action} at ({x:.0f}, {y:.0f}) on frame {frame_idx}."
    stale: bool = _invalidate_track(rec, session) if changed else False
    _log_mask(rec, session, _mask_hw(result) if result is not None else None, frame_idx)
    _log_points(rec, session, frame_idx)
    stream.flush()
    payload = stream.read()
    rec.disconnect()
    if stale:
        prefix += " The previous track is stale — Track again."
    status_text: str = _status(session, prefix)
    yield payload, status_text, status_text


def undo(session: Session | None) -> Iterator[tuple[bytes | None, str, str]]:
    """Remove the most recently added point."""
    if session is None:
        raise gr.Error("Upload a video first.")
    recording_pair: RecordingPair = _open_recording(session)
    rec: rr.RecordingStream = recording_pair[0]
    stream: rr.BinaryStream = recording_pair[1]
    _clear_preview(rec, session)
    edit: PointEdit = session.tracker.undo()
    if edit.point is None:
        rec.disconnect()
        status_text: str = _status(session, "Nothing to undo.")
        yield b"", status_text, status_text
        return
    stale: bool = _invalidate_track(rec, session)
    _log_mask(rec, session, _mask_hw(edit.result) if edit.result is not None else None, edit.point.frame_idx)
    _log_points(rec, session, edit.point.frame_idx)
    stream.flush()
    payload = stream.read()
    rec.disconnect()
    prefix: str = f"Undid the point on frame {edit.point.frame_idx}."
    if stale:
        prefix += " The previous track is stale — Track again."
    status_text = _status(session, prefix)
    yield payload, status_text, status_text


def clear(session: Session | None) -> Iterator[tuple[bytes | None, str, str]]:
    """Drop every point, every memory, and every overlay."""
    if session is None:
        raise gr.Error("Upload a video first.")
    recording_pair: RecordingPair = _open_recording(session)
    rec: rr.RecordingStream = recording_pair[0]
    stream: rr.BinaryStream = recording_pair[1]
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
    payload = stream.read()
    rec.disconnect()
    suffix: str = " The previous track is stale — add a point and Track again." if stale else ""
    status_text: str = f"Cleared all points and memory.{suffix}"
    yield payload, status_text, status_text


def track(session: Session | None) -> Iterator[tuple[bytes | None, str, str, object, object]]:
    """Propagate in both directions, stream masks, and write the downloadable RRD."""
    if session is None:
        raise gr.Error("Upload a video first.")
    if not session.tracker.points:
        raise gr.Error("Click at least one point on the object first.")
    recording_pair: RecordingPair = _open_recording(session)
    rec: rr.RecordingStream = recording_pair[0]
    stream: rr.BinaryStream = recording_pair[1]
    _clear_preview(rec, session)
    _invalidate_track(rec, session)
    stream.flush()
    status_text: str = "Tracking from the prompted frame in both directions…"
    yield (
        stream.read(),
        status_text,
        status_text,
        "Outputs",
        gr.DownloadButton(visible=False),
    )
    done: int = 0
    low_score: int = 0
    for result in session.tracker.track():
        mask_hw: UInt8[ndarray, "h w"] = _mask_hw(result)
        # Dense output: every frame gets its own mask, so no next-frame bounds, and
        # points were already logged (and bounded) when they were placed.
        _log_mask(rec, session, mask_hw, result.frame_idx, bound=False)
        _log_confidence(rec, session, result)
        session.tracked_frame_indices.add(result.frame_idx)
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
    stream.flush()
    payload = stream.read()
    rec.disconnect()
    output: Path = _merge_rrd_parts(session)
    yield (
        payload,
        f"Done: tracked all {done} frames; {low_score} low-confidence frame(s). Add a corrective click where needed, then Track again.",
        f"Done: tracked all {done} frames; {low_score} low-confidence frame(s). Add a corrective click where needed, then Track again.",
        "Outputs",
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
/* Only the controls scroll on short viewports; the viewer never moves. Reserve the
   scrollbar gutter so the controls do not shift horizontally when it appears. */
#left-column { height: 100%; max-height: 100%; overflow-y: auto; scrollbar-gutter: stable; flex-wrap: nowrap; }
#left-column > * { flex-shrink: 0; }
#source-video { height: auto !important; }
#source-video video { max-height: 225px !important; }
#examples { flex: none; }
/* Radio-based tab strip: Gradio 6.13 loops in Svelte when this app renders three
   native Tab components. The panels below are ordinary Columns. */
#control-tabs { overflow: visible; min-width: 0 !important; }
#control-tabs .wrap { display: flex; gap: 0; border-bottom: 1px solid var(--border-color-primary); }
#control-tabs label { flex: 1; justify-content: center; margin: 0; border: 0; border-bottom: 2px solid transparent; border-radius: 0; background: transparent; padding: 0.55rem 0.4rem; color: var(--body-text-color); cursor: pointer; }
#control-tabs label.selected { border-bottom-color: var(--color-accent); color: var(--color-accent); }
#control-tabs input { display: none; }
#control-tabs span { font-weight: 600; }
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


def build_demo(config: AppConfig) -> gr.Blocks:
    """Build a click-to-track app with the requested initial model variant."""
    with gr.Blocks(title="posekit: Click to Track") as demo:
        gr.Markdown(DESCRIPTION, elem_id="app-description")
        recording_state: gr.State = gr.State(value=None, delete_callback=_close_session)
        stamp_in: gr.Number = gr.Number(value=0.0, visible=False)
        with gr.Row(elem_id="main-row"):
            with gr.Column(scale=1, elem_id="left-column"):
                # Video and status sit above the tabs: the status must stay visible when
                # Track auto-switches to Outputs, and it belongs right under the input.
                video_in: gr.Video = gr.Video(label="Video", height=240, elem_id="source-video")
                status: gr.Markdown = gr.Markdown("Upload a video to begin.", elem_id="run-status")
                # Three native gr.Tab components trigger a Svelte effect loop in Gradio 6.13.
                tabs_radio: gr.Radio = gr.Radio(
                    choices=["Input", "Config", "Outputs"],
                    value="Input",
                    show_label=False,
                    container=False,
                    elem_id="control-tabs",
                )
                with gr.Column() as input_panel:
                    # A segmented control (see #click-mode CSS): the three everyday click modes on one line.
                    mode: gr.Radio = gr.Radio(choices=MODES, value=MODES[0], show_label=False, container=False, elem_id="click-mode")
                    with gr.Row():
                        undo_btn: gr.Button = gr.Button("Undo point", size="sm")
                        clear_btn: gr.Button = gr.Button("Clear", size="sm")
                with gr.Column(visible=False) as config_panel:
                    model_dd: gr.Dropdown = gr.Dropdown(label="Model", choices=SAM2_VARIANTS, value=config.variant)
                    memory_window_slider: gr.Slider = gr.Slider(
                        label="Memory window (frames)", minimum=1, maximum=32, step=1, value=DEFAULT_MEMORY_WINDOW
                    )
                    remove_radius_slider: gr.Slider = gr.Slider(
                        label="Remove radius (px)", minimum=10, maximum=300, step=10, value=DEFAULT_REMOVE_RADIUS_PX
                    )
                    resegment_box: gr.Checkbox = gr.Checkbox(
                        value=False,
                        label="Clicks replace the object (re-segment) instead of refining the propagated mask",
                    )
                    gr.Markdown("Model and memory window apply when the clip loads: changing them reloads it and clears the points.")
                with gr.Column(visible=False) as outputs_panel:
                    gr.Markdown(
                        "When tracking finishes, download a self-contained Rerun recording with the video, prompts, masks, and confidence traces."
                    )
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
            inputs=[video_in, recording_state, model_dd, memory_window_slider],
            outputs=[viewer, recording_state, status, status_probe, tabs_radio, download],
            api_visibility="private",
        )
        # Model and memory window shape the tracker state, so each change reloads the clip while Config stays selected.
        for event in (model_dd.change, memory_window_slider.release):
            event(
                fn=_reload_video_from_config,
                inputs=[video_in, recording_state, model_dd, memory_window_slider],
                outputs=[viewer, recording_state, status, status_probe, tabs_radio, download],
                api_visibility="private",
            )
        tabs_radio.change(
            fn=show_control_tab,
            inputs=[tabs_radio],
            outputs=[input_panel, config_panel, outputs_panel],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        # always_last coalesces the flood of time updates while scrubbing/playing into "the latest one".
        viewer.time_update(
            fn=record_time,
            inputs=[recording_state, stamp_in],
            outputs=None,
            queue=False,
            trigger_mode="multiple",
            js="(session, _stamp) => [session, performance.now()]",
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
            inputs=[recording_state, mode, resegment_box, remove_radius_slider],
            outputs=[viewer, status, status_probe],
            show_progress="hidden",
            api_visibility="private",
        )
        undo_btn.click(fn=undo, inputs=[recording_state], outputs=[viewer, status, status_probe], api_visibility="private")
        clear_btn.click(fn=clear, inputs=[recording_state], outputs=[viewer, status, status_probe], api_visibility="private")
        track_event = track_btn.click(fn=track, inputs=[recording_state], outputs=[viewer, status, status_probe, tabs_radio, download])
        stop_btn.click(fn=stop_tracking, outputs=[status, status_probe], cancels=[track_event], queue=False)

    return demo


# The embedded viewer stream needs a secure context: on the tailnet use
# `tailscale serve --bg --https=<port> http://127.0.0.1:<port>` (served at `/`, so root_path stays empty).
def launch(config: AppConfig) -> None:
    """Launch the click-to-track Gradio app."""
    build_demo(config).launch(
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
