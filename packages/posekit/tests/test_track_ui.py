"""Pure helpers of the click-to-track UI."""

import json
import subprocess
from pathlib import Path
from typing import Any, TypeAlias, cast
from unittest.mock import Mock

import gradio as gr
import numpy as np
import pytest
import rerun as rr
import rerun.experimental as rrx
import torch
from gradio_rerun import Rerun
from gradio_rerun.events import SelectionChange, TimeUpdate
from jaxtyping import Int64, UInt8
from numpy import ndarray

import posekit.track_recording as track_recording
import posekit.track_ui as track_ui
from posekit.apis.click_tracker import ClickTracker, MaskResult
from posekit.track_recording import MASK_SCALE, Session, _close_session, _invalidate_track, _mask_hw, _merge_rrd_parts, _open_recording

CLIP: Path = Path(__file__).resolve().parents[2] / "wilor-nano" / "assets" / "video.mp4"
SessionWithMock: TypeAlias = tuple[Session, Mock]


def _session(recording_id: str, *, prompted: list[int] | None = None) -> SessionWithMock:
    tracker_mock: Mock = Mock(spec=ClickTracker)
    tracker_mock.prompted_frames.return_value = (1,) if prompted is None else tuple(prompted)
    tracker_mock.num_frames = 4
    tracker_mock.points = ()
    frame_timestamps_ns: Int64[ndarray, "num_frames"] = (np.arange(4) * 33_333_333).astype(np.int64)
    session: Session = Session(
        tracker=cast(ClickTracker, tracker_mock),
        video_path=CLIP,
        recording_id=recording_id,
        frame_timestamps_ns=frame_timestamps_ns,
    )
    return session, tracker_mock


def test_invalidating_track_clears_propagated_masks_and_confidence() -> None:
    session_with_mock: SessionWithMock = _session("stale")
    session: Session = session_with_mock[0]
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


def test_state_delete_callback_closes_tracker_and_removes_rrds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_with_mock: SessionWithMock = _session("expired")
    session: Session = session_with_mock[0]
    tracker: Mock = session_with_mock[1]
    monkeypatch.setattr(track_recording, "RRD_DIR", tmp_path)
    config: track_ui.AppConfig = track_ui.AppConfig(variant="efficienttam-ti-512")
    demo: gr.Blocks = track_ui.build_demo(config)
    states: list[gr.State] = [block for block in demo.blocks.values() if isinstance(block, gr.State)]
    model_dropdowns: list[gr.Dropdown] = [block for block in demo.blocks.values() if isinstance(block, gr.Dropdown) and block.label == "Model"]
    assert len(states) == 1
    assert states[0].delete_callback is _close_session
    assert len(model_dropdowns) == 1
    assert model_dropdowns[0].value == config.variant
    part_dir: Path = tmp_path / session.recording_id
    part_dir.mkdir()
    (part_dir / "0000.rrd").touch()
    (tmp_path / f"{session.recording_id}.rrd").touch()

    _close_session(session)

    tracker.close.assert_called_once_with()
    assert not part_dir.exists()
    assert not (tmp_path / f"{session.recording_id}.rrd").exists()


def test_download_recording_contains_video_prompts_masks_and_confidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_with_mock: SessionWithMock = _session("download", prompted=[0])
    session: Session = session_with_mock[0]
    monkeypatch.setattr(track_recording, "RRD_DIR", tmp_path)

    recording_pair: tuple[rr.RecordingStream, rr.BinaryStream] = _open_recording(session)
    rec: rr.RecordingStream = recording_pair[0]
    rec.log("video", rr.AssetVideo(path=CLIP), static=True)
    rec.log("video/points", rr.Points2D([[360.0, 450.0]]))
    rec.disconnect()
    recording_pair = _open_recording(session)
    rec = recording_pair[0]
    rec.log("video/mask", rr.SegmentationImage(np.ones((2, 2), dtype=np.uint8)))
    rec.log("video/confidence", rr.Scalars(0.8))
    rec.log("video/object_score", rr.Scalars(0.9))
    rec.disconnect()

    output = _merge_rrd_parts(session)
    stats = subprocess.run(["rerun", "rrd", "stats", str(output)], check=True, capture_output=True, text=True)
    entities = {str(chunk.entity_path) for chunk in rrx.RrdReader(output).stream()}
    assert {"/video", "/video/points", "/video/mask", "/video/confidence", "/video/object_score"} <= entities
    assert "/video/mask" in stats.stdout


def test_mask_hw_downsamples_by_mask_scale() -> None:
    result: MaskResult = MaskResult(
        frame_idx=0,
        mask=torch.ones((64, 96), dtype=torch.bool),
        score=1.0,
        object_score=1.0,
    )

    mask_hw: UInt8[ndarray, "h w"] = _mask_hw(result)

    assert mask_hw.shape == (64 // MASK_SCALE, 96 // MASK_SCALE)


def test_click_on_masked_pixel_uses_the_overlay_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The viewer reports a masked click on the overlay entity (video pixels) and lists /video without a position."""
    session_with_mock: SessionWithMock = _session("selection")
    session: Session = session_with_mock[0]
    tracker: Mock = session_with_mock[1]
    tracker.points_on.return_value = ()
    tracker.add_point.return_value = MaskResult(
        frame_idx=0,
        mask=torch.ones((64, 96), dtype=torch.bool),
        score=1.0,
        object_score=1.0,
    )
    rec_mock: Mock = Mock(spec=rr.RecordingStream)
    stream_mock: Mock = Mock(spec=rr.BinaryStream)
    stream_mock.read.return_value = b"payload"
    monkeypatch.setattr(track_ui, "_open_recording", lambda _session: (rec_mock, stream_mock))
    event_json: str = json.dumps(
        {
            "type": "selection_change",
            "application_id": "test",
            "recording_id": session.recording_id,
            "items": [
                {"type": "entity", "entity_path": "/video/mask", "position": [80.0, 90.0]},
                {"type": "entity", "entity_path": "/video", "position": None},
            ],
        }
    )
    event: SelectionChange = SelectionChange(None, event_json)

    list(track_ui.on_select(session, "+ Include", False, 100.0, event))

    tracker.add_point.assert_called_once_with(0, 80.0, 90.0, positive=True, resegment=False)


def test_load_video_forwards_model_and_memory_config(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}
    predictor: object = object()

    class FakeClickTracker(ClickTracker):
        def __init__(self, video_path: Path, actual_predictor: object, *, memory_window_size: int) -> None:
            observed.update(video_path=video_path, predictor=actual_predictor, memory_window_size=memory_window_size)

    def fake_log_video(
        video_path: Path,
        video_log_path: Path,
        timeline: str,
        *,
        recording: rr.RecordingStream,
        output_codec: rr.VideoCodec | None = None,
    ) -> Int64[ndarray, "num_frames"]:
        observed.update(video_log_path=video_log_path, timeline=timeline, recording=recording, output_codec=output_codec)
        return np.arange(4, dtype=np.int64)

    rec_mock: Mock = Mock(spec=rr.RecordingStream)
    stream_mock: Mock = Mock(spec=rr.BinaryStream)
    stream_mock.read.return_value = b"payload"
    monkeypatch.setattr(track_ui, "ClickTracker", FakeClickTracker)
    monkeypatch.setattr(track_ui, "cached_predictor", lambda _variant: predictor)
    monkeypatch.setattr(track_ui, "_open_recording", lambda _session: (rec_mock, stream_mock))
    monkeypatch.setattr(track_ui, "log_video", fake_log_video)

    outputs: list[tuple[object, ...]] = list(track_ui.load_video(str(CLIP), None, "efficienttam-ti-512", 17.0))
    config_outputs: list[tuple[object, ...]] = list(track_ui._reload_video_from_config(str(CLIP), None, "efficienttam-ti-512", 17.0))

    assert observed["video_path"] == CLIP
    assert observed["predictor"] is predictor
    assert observed["memory_window_size"] == 17
    assert observed["output_codec"] == rr.VideoCodec.H264
    assert outputs[-1][4] == "Input"
    assert config_outputs[-1][4] == "Config"


def test_failed_reload_keeps_previous_session_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_with_mock: SessionWithMock = _session("previous")
    previous_session: Session = session_with_mock[0]
    previous_tracker: Mock = session_with_mock[1]
    corrupt_video: Path = tmp_path / "corrupt.mp4"
    corrupt_video.write_bytes(b"not an mp4")
    monkeypatch.setattr(track_ui, "cached_predictor", lambda _variant: object())

    with pytest.raises((RuntimeError, ValueError)):
        next(track_ui.load_video(str(corrupt_video), previous_session, "efficienttam-s-512", 10.0))

    previous_tracker.close.assert_not_called()


def test_control_tab_shows_only_selected_panel() -> None:
    updates: tuple[gr.Column, gr.Column, gr.Column] = track_ui.show_control_tab("Config")

    assert [panel.visible for panel in updates] == [False, True, False]


def test_scrub_handler_never_writes_to_the_viewer() -> None:
    """time_update → on_time_update must not output to the viewer: every push makes the viewer echo stale time_updates."""
    demo: gr.Blocks = track_ui.build_demo(track_ui.AppConfig())
    viewer_id: int = next(block._id for block in demo.blocks.values() if isinstance(block, Rerun))
    time_update_fns: list[Any] = [fn for fn in demo.fns.values() if any(t[1] == "time_update" for t in fn.targets)]
    assert len(time_update_fns) == 1  # one unqueued, stamp-guarded listener; a queued one would re-dispatch stale events
    for fn in time_update_fns:
        assert viewer_id not in {block._id for block in fn.outputs}
        # always_last re-dispatches the *event* with a stale payload, which also reaches
        # record_time: a click after moving the playhead then prompts the wrong frame.
        assert fn.trigger_mode != "always_last"


def test_late_reload_failure_keeps_previous_session_and_closes_the_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure after the candidate tracker exists (e.g. transcoding) must not close the old session nor leak the new tracker."""
    previous: Session = _session("previous")[0]
    candidate_closed: list[bool] = []

    class CandidateClickTracker(ClickTracker):
        def __init__(self, video_path: Path, actual_predictor: object, *, memory_window_size: int) -> None:
            pass

        def close(self) -> None:
            candidate_closed.append(True)

    monkeypatch.setattr(track_ui, "ClickTracker", CandidateClickTracker)
    monkeypatch.setattr(track_ui, "cached_predictor", lambda _variant: object())

    def failing_log_video(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected transcode failure")

    monkeypatch.setattr(track_ui, "log_video", failing_log_video)
    previous_closed: list[bool] = []
    monkeypatch.setattr(previous.tracker, "close", lambda: previous_closed.append(True))
    with pytest.raises(RuntimeError, match="injected"):
        list(track_ui.load_video("clip.mp4", previous, "efficienttam-s-512", 10.0))
    assert not previous_closed
    assert candidate_closed == [True]


def test_time_update_ignores_older_stamps_for_status_too() -> None:
    """Unqueued requests finish out of order: an older event must not roll back the frame or the status text."""
    session: Session = _session("stamps")[0]
    session.frame_timestamps_ns = np.arange(0, 300, dtype=np.int64) * 33_366_667

    def event(frame_idx: int) -> TimeUpdate:
        return TimeUpdate(None, json.dumps({"type": "time_update", "application_id": "t", "recording_id": session.recording_id, "time": int(session.frame_timestamps_ns[frame_idx])}))

    newer: tuple[Any, Any, Any] = track_ui.on_time_update(session, 200.0, event(30))
    older: tuple[Any, Any, Any] = track_ui.on_time_update(session, 100.0, event(27))
    assert session.current_frame == 30
    assert "frame 30" in newer[0]
    assert all(isinstance(value, type(gr.skip())) for value in older)


def test_preview_requests_are_always_fresh_values() -> None:
    """Re-requesting the same frame (e.g. after Undo) must still trigger preview_request.change."""
    session: Session = _session("seq")[0]
    first: float = track_ui._preview_request(session, 42)
    second: float = track_ui._preview_request(session, 42)
    assert first != second
    assert int(first) % track_ui.PREVIEW_SEQ_STRIDE == 42 and int(second) % track_ui.PREVIEW_SEQ_STRIDE == 42
