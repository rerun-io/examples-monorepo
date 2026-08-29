"""Pure helpers of the click-to-track UI."""

import json
import subprocess
from pathlib import Path
from typing import TypeAlias, cast
from unittest.mock import Mock

import gradio as gr
import numpy as np
import pytest
import rerun as rr
import rerun.experimental as rrx
import torch
from gradio_rerun.events import SelectionChange
from jaxtyping import Int64, UInt8
from numpy import ndarray

import posekit.track_ui as track_ui
from posekit.apis.click_tracker import ClickTracker, MaskResult
from posekit.track_ui import MASK_SCALE, Session, _close_session, _invalidate_track, _mask_hw, _merge_rrd_parts, _open_recording

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
    monkeypatch.setattr(track_ui, "RRD_DIR", tmp_path)
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
    monkeypatch.setattr(track_ui, "RRD_DIR", tmp_path)

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


def test_click_position_comes_from_video_entity_before_mask(monkeypatch: pytest.MonkeyPatch) -> None:
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
                {"type": "entity", "entity_path": "/video/mask", "position": [8.0, 9.0]},
                {"type": "entity", "entity_path": "/video", "position": [80.0, 90.0]},
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
    monkeypatch.setattr(track_ui, "_predictor", lambda _variant: predictor)
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
    monkeypatch.setattr(track_ui, "_predictor", lambda _variant: object())

    with pytest.raises((RuntimeError, ValueError)):
        next(track_ui.load_video(str(corrupt_video), previous_session, "efficienttam-s-512", 10.0))

    previous_tracker.close.assert_not_called()


def test_control_tab_shows_only_selected_panel() -> None:
    updates: tuple[gr.Column, gr.Column, gr.Column] = track_ui.show_control_tab("Config")

    assert [panel.visible for panel in updates] == [False, True, False]
