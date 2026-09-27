"""Shared half-open action intervals and their active-label boundaries."""

from pathlib import Path

import pytest
from conftest import read_chunks

from dataforge import schema, writing
from dataforge.actions import Segment, active_labels, log_actions


def test_overlapping_segments_join_in_segment_order_and_clear_at_ends() -> None:
    frames, texts = active_labels([Segment(2, 5, "turn"), Segment(1, 3, "hold"), Segment(1, 3, "grab")])
    assert frames.tolist() == [0, 1, 2, 3, 5]
    assert texts == ["", "grab\nhold", "grab\nhold\nturn", "turn", ""]


def test_stop_drops_boundaries_at_or_after_it() -> None:
    frames, texts = active_labels([Segment(0, 3, "hold"), Segment(3, 9, "turn")], stop=3)
    assert frames.tolist() == [0]
    assert texts == ["hold"]


def test_no_segments_is_one_empty_document_at_frame_zero() -> None:
    frames, texts = active_labels([])
    assert frames.tolist() == [0]
    assert texts == [""]


@pytest.mark.parametrize("start,end", [(-1, 2), (3, 2)])
def test_invalid_interval_is_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="invalid action interval"):
        Segment(start, end, "hold")


def test_log_actions_writes_one_text_document_per_boundary(tmp_path: Path) -> None:
    frames, texts = active_labels([Segment(0, 2, "hold"), Segment(1, 3, "turn")])
    target = tmp_path / "actions.rrd"
    with writing.atomic_recording(target, recording_id="test", send_properties=False) as recording:
        log_actions(recording, "fine", times_ns=frames * 1_000_000, frame_indices=frames, texts=texts)
    chunks = [chunk for chunk in read_chunks(target) if chunk.entity_path == schema.actions_path("fine")]
    rows = [row for chunk in chunks for row in chunk.to_record_batch().to_pylist()]
    assert [row["frame_index"] for row in rows] == frames.tolist() == [0, 1, 2, 3]
    text_column = next(key for key in rows[0] if key.endswith("TextDocument:text"))
    assert [row[text_column] for row in rows] == [["hold"], ["hold\nturn"], ["turn"], [""]]
