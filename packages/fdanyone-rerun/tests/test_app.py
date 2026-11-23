"""The Gradio app without a GPU: it builds, refuses a run without a video, saves and offers each run's recording, waits
for the GPU lock only around a run, and reports a stopped run as stopped."""

import fcntl
import os
import threading
import time
from collections.abc import Callable, Generator, Iterator
from pathlib import Path

import gradio as gr
import pytest
import rerun as rr
from rerun.chunk import RrdReader

from fdanyone_rerun.gradio_ui import app
from fdanyone_rerun.gradio_ui.app import APP_STEPS, build, run_app
from fdanyone_rerun.person4d import PersonConfig


class QuickStreamer:
    """A run that logs three points; ``hold`` keeps it running until it is stopped."""

    hold: bool = False

    def __init__(self, _config: object, recording: rr.RecordingStream, **_kwargs: object) -> None:
        self.recording: rr.RecordingStream = recording
        self.stop: threading.Event = threading.Event()
        self.messages: list[str] = ["starting"]
        self.person_summary: None = None

    def run(self) -> None:
        time.sleep(0.1)  # a few empty reads first
        for k in range(3):
            self.recording.log(f"world/p{k}", rr.Points3D([[float(k), 0.0, 0.0]]))
        if self.hold:
            self.stop.wait(10)


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The app with QuickStreamer runs saved under ``tmp_path``."""
    monkeypatch.setattr(app, "RunStreamer", QuickStreamer)
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "PUMP_SECONDS", 0.02)
    monkeypatch.setattr(app.SLOT, "recording", None)
    monkeypatch.setattr(app.SLOT, "gpu_lock", None)
    yield tmp_path


def start(runs: Path) -> Generator[app.RunYield, None, None]:
    return run_app(str(runs / "clip.mp4"), 6, 42, True, 100, False)


def status(step: app.RunYield) -> str:
    """The status box of one step (the app never skips it)."""
    assert isinstance(step[1], str)
    return step[1]


def test_app_builds() -> None:
    assert isinstance(build(examples=[]), gr.Blocks)


def test_run_without_a_video_asks_for_one() -> None:
    (step,) = list(run_app(None, 6, 42, True, APP_STEPS, True))
    assert step[:2] == (None, "Choose a video first.")


def test_the_app_starts_shorter_than_the_cli_with_the_cli_prior() -> None:
    # A demo click stays short; the prior checkbox starts at the CLI's default (on); the settings map onto the CLI's config (below).
    prior = next(block for block in build(examples=[]).blocks.values() if isinstance(block, gr.Checkbox) and "prior" in str(block.label))
    assert prior.value is True and PersonConfig().prior is True
    assert PersonConfig().train.steps > APP_STEPS


def test_a_run_saves_a_valid_recording_and_offers_it_by_name(runs: Path) -> None:
    """The viewer's bytes and the saved ``<run>.rrd`` hold the same recording; the status names the file, not its path;
    the download box gets it. ``BinaryStream.read()`` returns None while nothing new was logged."""
    assert app.latest_recording().visible is False
    pumped: list[app.RunYield] = list(start(runs))

    (saved,) = runs.glob("*.rrd")
    assert {"/world/p0", "/world/p1", "/world/p2"} <= {str(chunk.entity_path) for chunk in RrdReader(saved).stream().to_chunks()}
    assert sum(len(step[0] or b"") for step in pumped) > 0
    _, done, summary, _ = pumped[-1]
    assert isinstance(done, str) and done.startswith("Done in") and str(runs) not in done
    assert isinstance(summary, str) and f"`{saved.name}`" in summary and str(runs) not in summary
    logs: list[str] = [step[3] for step in pumped if isinstance(step[3], str)]  # a poll with no new line skips the box
    assert logs and logs[0].endswith(" s  starting")  # each line stamped with the seconds since Run
    offered = app.latest_recording()
    assert offered.visible and saved.name in str(offered.value)


def test_a_stopped_run_says_so_and_offers_nothing(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QuickStreamer, "hold", True)
    run = start(runs)
    assert status(next(run)).endswith(" s)")  # the newest progress line and the seconds so far
    assert list(app.stop_status())[-1] == "Stopped."
    assert [status(step) for step in run][-1].startswith("Stopped after")
    assert app.latest_recording().visible is False


def _hold_gpu_lock(path: Path) -> Callable[[], None]:
    """Another job holds the GPU lock; returns the call that ends that job."""
    other: int = os.open(path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(other, fcntl.LOCK_EX)
    return lambda: os.close(other)


def test_the_gpu_lock_is_held_around_a_run_only(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``Config.gpu_lock`` a run waits while another job holds the lock, and frees it when the run ends."""
    lock: Path = runs / "gpu.lock"
    release_other: Callable[[], None] = _hold_gpu_lock(lock)
    monkeypatch.setattr(app.SLOT, "gpu_lock", lock)
    run = start(runs)
    assert status(next(run)).startswith("Waiting for the GPU")
    release_other()
    assert [status(step) for step in run][-1].startswith("Done")
    mine: int = os.open(lock, os.O_RDWR)
    fcntl.flock(mine, fcntl.LOCK_EX | fcntl.LOCK_NB)  # free again once the run has ended
    os.close(mine)


def test_a_page_closed_while_waiting_for_the_gpu_leaves_nothing_behind(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock: Path = runs / "gpu.lock"
    release_other: Callable[[], None] = _hold_gpu_lock(lock)
    monkeypatch.setattr(app.SLOT, "gpu_lock", lock)
    run = start(runs)
    assert status(next(run)).startswith("Waiting for the GPU")
    run.close()  # the page goes away
    release_other()
    assert not list(runs.glob("*.rrd"))  # no recording was opened
    assert app.SLOT.lock.acquire(blocking=False)  # the run slot is free
    app.SLOT.lock.release()


@pytest.mark.parametrize(("train", "prior"), [(True, False), (True, True), (False, True)])
def test_the_settings_map_onto_the_shared_person_config(runs: Path, monkeypatch: pytest.MonkeyPatch, train: bool, prior: bool) -> None:
    """Steps and the MoGe-2 prior go into the one PersonConfig the CLI uses too (no app-only path); no training, no config."""
    seen: list[object] = []

    class Recording(QuickStreamer):
        def __init__(self, config: object, recording: rr.RecordingStream, **kwargs: object) -> None:
            super().__init__(config, recording)
            seen.append(kwargs["person"])

    monkeypatch.setattr(app, "RunStreamer", Recording)
    list(run_app(str(runs / "clip.mp4"), 6, 42, train, 1234, prior))
    expected: PersonConfig | None = PersonConfig(prior=prior).with_steps(1234) if train else None
    assert seen == [expected]


def test_a_person_config_needs_a_report_interval() -> None:
    # report_every is the progress interval and the trainer's log_every; 0 would divide by zero mid-run.
    with pytest.raises(ValueError, match="report_every"):
        PersonConfig(report_every=0)
