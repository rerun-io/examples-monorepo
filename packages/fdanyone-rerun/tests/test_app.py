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


def test_a_run_saves_a_valid_recording_and_survives_empty_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The viewer's bytes and the saved ``<run>.rrd`` hold the same recording; the saved file reads back as one .rrd.

    ``BinaryStream.read()`` returns None when nothing new was logged (Pablo's first run failed on that).
    """
    from rerun.chunk import RrdReader

    class LoggingStreamer:
        def __init__(self, _config: object, recording: app.rr.RecordingStream, **_kwargs: object) -> None:
            import threading

            self.recording = recording
            self.stop = threading.Event()

        def run(self) -> None:
            import time

            time.sleep(0.1)  # a few empty reads first
            for k in range(3):
                self.recording.log(f"world/p{k}", app.rr.Points3D([[float(k), 0.0, 0.0]]))
                time.sleep(0.05)

    monkeypatch.setattr(app, "RunStreamer", LoggingStreamer)
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "PUMP_SECONDS", 0.02)
    sent: bytes = b"".join(data for data, _ in run_app(str(tmp_path / "clip.mp4"), 6, 42, True, 100) if data is not None)

    (saved,) = tmp_path.glob("*.rrd")
    entities: set[str] = {str(chunk.entity_path) for chunk in RrdReader(saved).stream().to_chunks()}
    assert {"/world/p0", "/world/p1", "/world/p2"} <= entities
    assert len(sent) > 0


def test_the_gpu_lock_is_held_around_a_run_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``Config.gpu_lock`` a run waits while another job holds the lock, and frees it when the run ends."""
    import fcntl
    import os

    class QuickStreamer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            import threading

            self.stop = threading.Event()

        def run(self) -> None:
            pass

    lock: Path = tmp_path / "gpu.lock"
    other: int = os.open(lock, os.O_RDWR | os.O_CREAT)
    fcntl.flock(other, fcntl.LOCK_EX)  # another job holds the GPU
    monkeypatch.setattr(app, "RunStreamer", QuickStreamer)
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "PUMP_SECONDS", 0.02)
    monkeypatch.setattr(app.SLOT, "gpu_lock", lock)
    run = run_app(str(tmp_path / "clip.mp4"), 6, 42, True, 100)
    assert next(run)[1].startswith("Waiting for the GPU")
    os.close(other)  # the other job ends
    statuses: list[str] = [status for _, status in run]
    assert statuses[-1].startswith("Done")
    mine: int = os.open(lock, os.O_RDWR)
    fcntl.flock(mine, fcntl.LOCK_EX | fcntl.LOCK_NB)  # free again once the run has ended
    os.close(mine)
