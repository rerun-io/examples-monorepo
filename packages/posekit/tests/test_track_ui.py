"""Pure helpers of the click-to-track UI."""

import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import numpy as np
import rerun as rr
import rerun.experimental as rrx

import posekit.track_ui as track_ui
from posekit.apis.click_tracker import ClickTracker
from posekit.track_ui import Session, _close_session, _invalidate_track, _merge_rrd_parts, _open_recording

CLIP: Path = Path(__file__).resolve().parents[2] / "wilor-nano" / "assets" / "video.mp4"


def _session(recording_id: str, *, prompted: list[int] | None = None) -> tuple[Session, Mock]:
    tracker_mock: Mock = Mock(spec=ClickTracker)
    tracker_mock.prompted_frames.return_value = [1] if prompted is None else prompted
    tracker_mock.num_frames = 4
    tracker_mock.points = ()
    session = Session(
        tracker=cast(ClickTracker, tracker_mock),
        video_path=CLIP,
        recording_id=recording_id,
        frame_timestamps_ns=(np.arange(4) * 33_333_333).astype(np.int64),
    )
    return session, tracker_mock


def test_invalidating_track_clears_propagated_masks_and_confidence() -> None:
    session, _ = _session("stale")
    session.masked_frames = {0, 1, 2}
    session.tracked_frame_indices = {0, 1, 2}
    rec_mock: Mock = Mock(spec=rr.RecordingStream)

    assert _invalidate_track(cast(rr.RecordingStream, rec_mock), session)
    assert session.masked_frames == {1}
    assert session.tracked_frame_indices == set()
    entities = [call.args[0] for call in rec_mock.log.call_args_list]
    assert entities.count("video/mask") == 2
    assert entities.count("video/confidence") == 3
    assert entities.count("video/object_score") == 3


def test_state_delete_callback_closes_tracker_and_removes_rrds(tmp_path: Path, monkeypatch) -> None:
    session, tracker = _session("expired")
    monkeypatch.setattr(track_ui, "RRD_DIR", tmp_path)
    part_dir = tmp_path / session.recording_id
    part_dir.mkdir()
    (part_dir / "0000.rrd").touch()
    (tmp_path / f"{session.recording_id}.rrd").touch()

    assert track_ui.recording_state.delete_callback is _close_session
    _close_session(session)

    tracker.close.assert_called_once_with()
    assert not part_dir.exists()
    assert not (tmp_path / f"{session.recording_id}.rrd").exists()


def test_download_recording_contains_video_prompts_masks_and_confidence(tmp_path: Path, monkeypatch) -> None:
    session, _ = _session("download", prompted=[0])
    monkeypatch.setattr(track_ui, "RRD_DIR", tmp_path)

    rec, _ = _open_recording(session)
    rec.log("video", rr.AssetVideo(path=CLIP), static=True)
    rec.log("video/points", rr.Points2D([[360.0, 450.0]]))
    rec.disconnect()
    rec, _ = _open_recording(session)
    rec.log("video/mask", rr.SegmentationImage(np.ones((2, 2), dtype=np.uint8)))
    rec.log("video/confidence", rr.Scalars(0.8))
    rec.log("video/object_score", rr.Scalars(0.9))
    rec.disconnect()

    output = _merge_rrd_parts(session)
    stats = subprocess.run(["rerun", "rrd", "stats", str(output)], check=True, capture_output=True, text=True)
    entities = {str(chunk.entity_path) for chunk in rrx.RrdReader(output).stream()}
    assert {"/video", "/video/points", "/video/mask", "/video/confidence", "/video/object_score"} <= entities
    assert "/video/mask" in stats.stdout
