"""A stopped run leaves nothing behind: its worker subprocess is reaped, and the next run starts only after it ended."""

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fdanyone import pipeline
from fdanyone.errors import RunCancelled
from fdanyone.model.denoise import STEP_HOOKS
from fdanyone_rerun.core import RunConfig, observers
from fdanyone_rerun.gradio_ui import app

SLEEPER: str = "import os, sys, time\nopen(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
"""A worker that writes its pid to the request path and blocks (as GVHMR or the skeleton worker would)."""


@pytest.fixture
def worker_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "sleeper.py").write_text(SLEEPER)
    (tmp_path / "failer.py").write_text("raise SystemExit(3)\n")
    monkeypatch.setattr(pipeline, "_worker_environment", lambda: {**os.environ, "PYTHONPATH": str(tmp_path)})
    monkeypatch.delenv("FDANYONE_INLINE_WORKERS", raising=False)
    return tmp_path


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _cancel_once_running(pid_file: Path, cancel: threading.Event) -> None:
    """Set ``cancel`` once the worker has written its pid (it is blocked in its sleep by then)."""
    deadline: float = time.monotonic() + 20.0
    while not (pid_file.is_file() and pid_file.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    cancel.set()


def test_cancel_stops_and_reaps_a_blocked_worker(worker_module: Path) -> None:
    pid_file: Path = worker_module / "pid"
    cancel = threading.Event()
    threading.Thread(target=_cancel_once_running, args=(pid_file, cancel), daemon=True).start()
    started: float = time.perf_counter()
    with pytest.raises(RunCancelled):
        pipeline._run_worker("sleeper", pid_file, sys.executable, cancel)
    assert time.perf_counter() - started < 10.0
    assert _gone(int(pid_file.read_text()))  # terminated and reaped: not even a zombie


def test_a_failing_worker_still_raises(worker_module: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        pipeline._run_worker("failer", worker_module / "request.json", sys.executable, threading.Event())


def test_a_run_after_a_stop_waits_for_the_stopped_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fdanyone_rerun.core import _ProgressHandler

    runs: list[tuple[float, float]] = []

    class BlockingStreamer:
        """A run that holds upstream's observers until it is stopped, then takes a while to clean up."""

        def __init__(self, _config: RunConfig, _recording: object, previews: bool, person: object = None) -> None:
            self.stop: threading.Event = threading.Event()

        def on_stage(self, *_args: object) -> None:
            if self.stop.is_set():
                raise RunCancelled("stopped")

        def run(self) -> None:
            start: float = time.perf_counter()
            if runs:  # a second run: the first run's observers must be gone by now
                progress: logging.Logger = logging.getLogger("fdanyone.progress")
                assert STEP_HOOKS == [] and not any(isinstance(handler, _ProgressHandler) for handler in progress.handlers)
            else:
                with observers(self.on_stage, lambda *_: None):
                    self.stop.wait(10)
                    time.sleep(0.3)  # cleanup after the stop, longer than one slot wait: the slot must stay taken
            runs.append((start, time.perf_counter()))

    monkeypatch.setattr(app, "RunStreamer", BlockingStreamer)
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "PUMP_SECONDS", 0.05)
    video: str = str(tmp_path / "clip.mp4")
    first = app.run_app(video, 6, 42, False, 100)
    assert next(first)[1].startswith("Running")
    first.close()  # a closed page ends the request, which closes its generator
    statuses: list[str] = [status for _, status in app.run_app(video, 6, 42, False, 100)]
    assert statuses[0] == "Waiting for the previous run to stop…"
    assert statuses[-1].startswith("Done")
    (_, first_end), (second_start, _) = runs
    assert second_start >= first_end  # no overlap
    assert list(app.stop_status()) == ["Stopped."]


def test_stop_stops_a_run_whose_generator_gradio_left_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gradio's cancel does not close the run's generator: the Stop handler must set the run's stop event itself."""

    class WaitingStreamer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.stop: threading.Event = threading.Event()

        def run(self) -> None:
            if not self.stop.wait(10):
                raise AssertionError("never stopped")
            raise RunCancelled("stopped")

    monkeypatch.setattr(app, "RunStreamer", WaitingStreamer)
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "PUMP_SECONDS", 0.05)
    run = app.run_app(str(tmp_path / "clip.mp4"), 6, 42, True, 100)
    assert next(run)[1].startswith("Running")  # the generator stays suspended, as after a Gradio cancel
    started: float = time.perf_counter()
    assert list(app.stop_status())[-1] == "Stopped."
    assert time.perf_counter() - started < 5.0
    run.close()
