"""Skip/force/failure semantics of the per-segment layer batch runner."""

from pathlib import Path

import pytest

from arkitscenes_download.ingest.layer_batch import run_layer_batch, segment_ids_from_selection


def test_run_layer_batch_skips_generation_but_still_registers_existing(tmp_path: Path) -> None:
    (tmp_path / "a.rrd").touch()
    generated: list[str] = []
    registered: list[str] = []

    def generate(video_id: str, path: Path) -> str:
        generated.append(video_id)
        path.touch()
        return "ok"

    summary = run_layer_batch(["a", "b"], lambda v: tmp_path / f"{v}.rrd", generate, lambda v, _: registered.append(v), force=False, label="t")
    assert (summary.done, summary.skipped, summary.failed) == (["b"], ["a"], [])
    assert generated == ["b"]
    assert registered == ["a", "b"]  # the pre-existing file is (re-)registered, closing the written-but-unregistered gap

    summary = run_layer_batch(["a"], lambda v: tmp_path / f"{v}.rrd", generate, None, force=True, label="t")
    assert summary.done == ["a"] and generated == ["b", "a"]


def test_run_layer_batch_records_failures_and_continues(tmp_path: Path) -> None:
    def generate(video_id: str, path: Path) -> str:
        if video_id == "bad":
            raise RuntimeError("boom")
        return "ok"

    def register(video_id: str, path: Path) -> None:
        if video_id == "unregisterable":
            raise RuntimeError("catalog down")

    ids = ["bad", "good", "unregisterable"]
    summary = run_layer_batch(ids, lambda v: tmp_path / f"{v}.rrd", generate, register, force=False, label="t")
    assert (summary.done, summary.failed) == (["good"], ["bad", "unregisterable"])


def test_segment_ids_from_selection_requires_exactly_one_mode(tmp_path: Path) -> None:
    ids_file: Path = tmp_path / "ids.txt"
    ids_file.write_text("one\n\ntwo\n")
    assert segment_ids_from_selection("x", None) == ["x"]
    assert segment_ids_from_selection(None, ids_file) == ["one", "two"]
    with pytest.raises(SystemExit):
        segment_ids_from_selection(None, None)
    with pytest.raises(SystemExit):
        segment_ids_from_selection("x", ids_file)
