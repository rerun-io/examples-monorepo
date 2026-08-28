"""Skip/force/failure semantics of the per-segment layer batch runner."""

from pathlib import Path

import pytest

from arkitscenes_download.ingest.layer_batch import run_layer_batch, segment_ids_from_selection


def test_run_layer_batch_skips_existing_unless_forced(tmp_path: Path) -> None:
    (tmp_path / "a.rrd").touch()
    processed: list[str] = []

    def process(video_id: str) -> str:
        processed.append(video_id)
        (tmp_path / f"{video_id}.rrd").touch()
        return "ok"

    summary = run_layer_batch(["a", "b"], lambda v: tmp_path / f"{v}.rrd", process, force=False, label="test")
    assert (summary.done, summary.skipped, summary.failed) == (["b"], ["a"], [])
    assert processed == ["b"]

    summary = run_layer_batch(["a"], lambda v: tmp_path / f"{v}.rrd", process, force=True, label="test")
    assert summary.done == ["a"]


def test_run_layer_batch_records_failures_and_continues(tmp_path: Path) -> None:
    def process(video_id: str) -> str:
        if video_id == "bad":
            raise RuntimeError("boom")
        return "ok"

    summary = run_layer_batch(["bad", "good"], lambda v: tmp_path / f"{v}.rrd", process, force=False, label="test")
    assert (summary.done, summary.failed) == (["good"], ["bad"])


def test_segment_ids_from_selection_requires_exactly_one_mode(tmp_path: Path) -> None:
    ids_file: Path = tmp_path / "ids.txt"
    ids_file.write_text("one\n\ntwo\n")
    assert segment_ids_from_selection("x", None) == ["x"]
    assert segment_ids_from_selection(None, ids_file) == ["one", "two"]
    with pytest.raises(SystemExit):
        segment_ids_from_selection(None, None)
    with pytest.raises(SystemExit):
        segment_ids_from_selection("x", ids_file)
