"""Pure helpers of the click-to-track UI."""

from pathlib import Path
from typing import cast
from unittest.mock import Mock

import numpy as np
import rerun as rr
import rerun.experimental as rrx
import torch
from sam2.modeling.memory import ObjectMemory
from sam2.modeling.sam2_memory import SAM2ObjectMemoryBank, _select_N_closest_conditional_memories

import posekit.track_ui as track_ui
from posekit.apis.click_tracker import ClickTracker, Point
from posekit.track_ui import MAX_SESSIONS, Session, TrackedFrame, _invalidate_track, _save_recording, _store_session, frame_at

CLIP: Path = Path(__file__).resolve().parents[2] / "wilor-nano" / "assets" / "video.mp4"


def test_frame_at_matches_viewer_latest_at_semantics() -> None:
    ts_ns = (np.arange(10) * 33_333_333).astype(np.int64)  # 30 fps
    assert frame_at(ts_ns, 0.0) == 0
    assert frame_at(ts_ns, 10_000_000.0) == 0  # 10 ms -> still frame 0
    assert frame_at(ts_ns, 30_000_000.0) == 0  # 30 ms -> still frame 0 (viewer shows pts <= t)
    assert frame_at(ts_ns, 33_333_333.0) == 1
    assert frame_at(ts_ns, 10_833_333_333.0) == 9  # past the end saturates
    assert frame_at(ts_ns, -5.0) == 0


def _memory(frame_idx: int, *, conditional: bool) -> ObjectMemory:
    return ObjectMemory(
        obj_id=0,
        frame_idx=frame_idx,
        memory_embeddings=torch.zeros(1, 1, 1, 1),
        memory_pos_embeddings=torch.zeros(1, 1, 1, 1),
        ptr=torch.zeros(1, 1),
        is_conditional=conditional,
    )


def test_conditional_memory_selection_is_bounded_and_does_not_mutate_bank() -> None:
    conditional = [_memory(frame_idx, conditional=True) for frame_idx in (0, 10, 20, 30, 40)]
    selected, unselected = _select_N_closest_conditional_memories(conditional, N=3, current_frame_idx=25)
    assert [memory.frame_idx for memory in selected] == [20, 30, 10]
    assert {memory.frame_idx for memory in unselected} == {0, 40}

    bank = SAM2ObjectMemoryBank()
    bank.known_obj_ids.add(0)
    bank.conditional_memories[0] = conditional
    bank.non_conditional_memories[0] = [_memory(24, conditional=False)]
    bank.select_memories(
        obj_ids=[0],
        current_frame_idx=25,
        max_conditional_memories=3,
        max_non_conditional_memories=2,
        max_ptr_memories=2,
    )
    assert [memory.frame_idx for memory in bank.non_conditional_memories[0]] == [24]


def _session(recording_id: str, *, prompted: list[int] | None = None) -> tuple[Session, Mock]:
    tracker_mock: Mock = Mock(spec=ClickTracker)
    tracker_mock.prompted_frames.return_value = [1] if prompted is None else prompted
    tracker_mock.num_frames = 4
    tracker_mock.points = []
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
    tracked = TrackedFrame(mask_bits=np.packbits(np.ones(4, dtype=np.uint8)), score=0.8, object_score=0.9)
    session.tracked_frames = {0: tracked, 1: tracked, 2: tracked}
    rec_mock: Mock = Mock(spec=rr.RecordingStream)

    assert _invalidate_track(cast(rr.RecordingStream, rec_mock), session)
    assert session.masked_frames == {1}
    assert session.tracked_frames == {}
    entities = [call.args[0] for call in rec_mock.log.call_args_list]
    assert entities.count("video/mask") == 2
    assert entities.count("video/confidence") == 3
    assert entities.count("video/object_score") == 3


def test_session_store_evicts_oldest_tracker() -> None:
    track_ui._SESSIONS.clear()
    sessions: list[Session] = []
    trackers: list[Mock] = []
    for index in range(MAX_SESSIONS + 1):
        session, tracker = _session(str(index))
        sessions.append(session)
        trackers.append(tracker)
        _store_session(session, None)
    assert list(track_ui._SESSIONS) == [str(index) for index in range(1, MAX_SESSIONS + 1)]
    trackers[0].close.assert_called_once_with()
    track_ui._SESSIONS.clear()


def test_download_recording_contains_video_prompts_masks_and_confidence(tmp_path: Path, monkeypatch) -> None:
    timestamps_ns = rr.AssetVideo(path=CLIP).read_frame_timestamps_nanos()
    session, tracker = _session("download", prompted=[0])
    session.frame_timestamps_ns = timestamps_ns
    tracker.num_frames = len(timestamps_ns)
    tracker.points_on.return_value = [Point(frame_idx=0, x=360.0, y=450.0, positive=True)]
    tracker.frame_hw = (720, 1280)
    tracked = TrackedFrame(mask_bits=np.packbits(np.ones(720 * 1280, dtype=np.uint8)), score=0.8, object_score=0.9)
    session.tracked_frames = {0: tracked, 1: tracked}
    monkeypatch.setattr(track_ui, "RRD_DIR", tmp_path)

    output = _save_recording(session)
    entities = {str(chunk.entity_path) for chunk in rrx.RrdReader(output).stream()}
    assert {"/video", "/video/points", "/video/mask", "/video/confidence", "/video/object_score"} <= entities
