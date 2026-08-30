"""Owns each session's Rerun recording: the live-stream tee and downloadable RRD parts, with overlays logged beneath the scale transform."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int, UInt8
from numpy import ndarray

from posekit.apis.click_tracker import ClickTracker, MaskResult, Point

APP_ID: str = "posekit_click_to_track"
VIDEO: str = "video"
TIMELINE: str = "video_time"
RecordingPair: TypeAlias = tuple[rr.RecordingStream, rr.BinaryStream]
POSITIVE_COLOR: tuple[int, int, int] = (76, 220, 96)
NEGATIVE_COLOR: tuple[int, int, int] = (255, 87, 51)
MASK_COLOR: tuple[int, int, int] = (51, 178, 255)
MASK_SCALE: int = 4
"""Masks are logged at 1/MASK_SCALE resolution under a static scale transform: a full-res
uint8 mask per frame costs ~40x the video in viewer memory, and SAM2's logits are 128^2 anyway."""
RRD_DIR: Path = Path("/tmp/posekit-rrd")


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
