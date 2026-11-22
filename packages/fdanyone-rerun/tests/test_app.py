"""The Gradio app builds without a GPU and refuses a run without a video."""

from pathlib import Path

import gradio as gr
import pytest

from fdanyone_rerun.gradio_ui import app
from fdanyone_rerun.gradio_ui.app import APP_PERSON, build, run_app
from fdanyone_rerun.person4d import PersonConfig


def test_app_builds() -> None:
    assert isinstance(build(examples=[]), gr.Blocks)


def test_run_without_a_video_asks_for_one() -> None:
    assert list(run_app(None, 6, 42, True, 5000)) == [(None, "Choose a video first.")]


def test_the_app_preset_is_short_and_runs_no_extra_estimator() -> None:
    # Pablo's demo-app rule: no MoGe-2 in the app; the CLI default keeps the prior and the long run.
    assert APP_PERSON.prior is False and APP_PERSON.train.steps == 5_000
    assert PersonConfig().prior is True and PersonConfig().train.steps == 25_000
    assert APP_PERSON.max_side == PersonConfig().max_side and APP_PERSON.asog == PersonConfig().asog


def test_the_pump_saves_what_the_viewer_gets_and_survives_empty_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``BinaryStream.read()`` returns None when nothing new was logged (Pablo's first run failed on that)."""
    reads: list[bytes | None] = [None, b"RRF2-a", None, b"-b"]

    class FakeStream(app.rr.BinaryStream):
        def __init__(self) -> None:  # no native stream behind it
            pass

        def read(self, *, flush: bool = True) -> bytes | None:
            return reads.pop(0) if reads else None

    class FakeRecording:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def binary_stream(self) -> FakeStream:
            return FakeStream()

        def flush(self) -> None:
            pass

    class QuickStreamer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            import threading

            self.stop = threading.Event()

        def run(self) -> None:
            import time

            time.sleep(0.2)

    monkeypatch.setattr(app.rr, "RecordingStream", FakeRecording)
    monkeypatch.setattr(app, "RunStreamer", QuickStreamer)
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "PUMP_SECONDS", 0.02)
    sent: list[bytes] = [data for data, _ in run_app(str(tmp_path / "clip.mp4"), 6, 42, True, 100) if data is not None]

    saved: list[Path] = list(tmp_path.glob("*.rrd"))
    assert len(saved) == 1 and saved[0].read_bytes() == b"".join(sent) == b"RRF2-a-b"
