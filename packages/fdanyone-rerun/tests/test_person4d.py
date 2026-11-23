"""The 4D person on the example run (GPU, weights): trained on the six views, logged as Animated SOG, and stoppable."""

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import rerun as rr

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (fdanyone-rerun envs)")

pytestmark = pytest.mark.integration

if TYPE_CHECKING:
    from fdanyone_rerun.person4d import PersonConfig


def _short(steps: int) -> "PersonConfig":
    import dataclasses

    from fdanyone_rerun.person4d import PersonConfig

    return dataclasses.replace(PersonConfig().with_steps(steps), prior=False, preview_every=100, report_every=100)


def test_the_person_goes_into_the_recording_only(example_run: Path, tmp_path: Path) -> None:
    from rerun.chunk import RrdReader

    from fdanyone_rerun.person4d import train_person
    from fdanyone_rerun.rerun_log import PERSON, TRAIN_PREVIEW

    before: set[Path] = set(example_run.rglob("*"))
    recording = rr.RecordingStream("fdanyone-rerun-test")
    recording.save(tmp_path / "person.rrd")
    lines: list[str] = []
    summary = train_person(example_run, _short(300), recording, report=lines.append)
    recording.flush()
    recording.disconnect()

    rows: dict[str, int] = {}
    columns: set[str] = set()
    for chunk in RrdReader(tmp_path / "person.rrd").stream().to_chunks():
        batch = chunk.to_record_batch()
        rows[str(chunk.entity_path)] = rows.get(str(chunk.entity_path), 0) + batch.num_rows
        if str(chunk.entity_path) == f"/{PERSON}":
            columns |= set(batch.schema.names)
    assert any("AnimatedSogSplats3D:asog" in name for name in columns)
    assert any("GaussianSplats3D:centers" in name for name in columns)  # the stock viewer's one-pose preview
    assert rows[f"/{TRAIN_PREVIEW}"] == 3  # steps 100, 200, 300
    assert summary.steps == 300 and summary.asog_rows > 0 and summary.asog_bytes > 0
    assert any(line.startswith("train step 300 / 300") for line in lines)
    assert set(example_run.rglob("*")) == before  # no .asog, checkpoint or mask files next to the run


def test_stop_ends_the_training(example_run: Path, tmp_path: Path) -> None:
    from fdanyone.errors import RunCancelled
    from fdanyone_rerun.person4d import train_person

    cancel = threading.Event()

    def report(line: str) -> None:
        if line.startswith("train step 100 "):
            cancel.set()

    recording = rr.RecordingStream("fdanyone-rerun-test")
    recording.memory_recording()
    with pytest.raises(RunCancelled):
        train_person(example_run, _short(5_000), recording, report=report, cancel=cancel)
