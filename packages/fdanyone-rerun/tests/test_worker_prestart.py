"""A worker started ahead of its request runs it when sent, exits when discarded unused, and is stopped on cancel."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fdanyone import pipeline
from fdanyone.errors import RunCancelled

WAITER: str = (
    "import os, sys, time\n"
    "assert sys.argv[1] == '--wait-for-request', sys.argv\n"
    "request = sys.stdin.readline().strip()\n"
    "if request:\n"
    "    open(request, 'w').write(f'{os.getpid()} {sys.argv[2:]}')\n"
    "    time.sleep(float(os.environ.get('WAITER_SLEEP', '0')))\n"
)


@pytest.fixture
def waiter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "waiter.py").write_text(WAITER)
    monkeypatch.setattr(pipeline, "_worker_environment", lambda: {**os.environ, "PYTHONPATH": str(tmp_path)})
    monkeypatch.delenv("FDANYONE_INLINE_WORKERS", raising=False)
    return tmp_path


def test_a_started_worker_runs_the_request_it_is_sent(waiter: Path) -> None:
    child = pipeline._start_worker("waiter", sys.executable, "root-arg")
    assert child is not None
    request: Path = waiter / "request.json"
    pipeline._run_worker("waiter", request, sys.executable, threading.Event(), child)
    pid, preload = request.read_text().split(" ", 1)
    assert int(pid) == child.pid and preload == "['root-arg']"
    assert child.returncode == 0


def test_an_unused_started_worker_exits_when_discarded(waiter: Path) -> None:
    child = pipeline._start_worker("waiter", sys.executable)
    assert child is not None
    started: float = time.perf_counter()
    pipeline._discard_worker(child)
    assert child.returncode == 0 and time.perf_counter() - started < 10.0
    pipeline._discard_worker(child)  # a second discard is a no-op


def test_cancel_stops_a_started_worker(waiter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAITER_SLEEP", "60")
    child = pipeline._start_worker("waiter", sys.executable)
    assert child is not None
    request: Path = waiter / "request.json"
    cancel = threading.Event()

    def cancel_once_running() -> None:
        deadline: float = time.monotonic() + 20.0
        while not (request.is_file() and request.read_text()) and time.monotonic() < deadline:
            time.sleep(0.05)
        cancel.set()

    threading.Thread(target=cancel_once_running, daemon=True).start()
    with pytest.raises(RunCancelled):
        pipeline._run_worker("waiter", request, sys.executable, cancel, child)
    assert child.returncode is not None


def test_inline_mode_starts_no_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FDANYONE_INLINE_WORKERS", "1")
    assert pipeline._start_worker("waiter", sys.executable) is None


def test_a_started_worker_that_fails_still_raises(waiter: Path) -> None:
    (waiter / "failer.py").write_text("raise SystemExit(3)\n")
    child = pipeline._start_worker("failer", sys.executable)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline._run_worker("failer", waiter / "request.json", sys.executable, threading.Event(), child)


def test_while_running_runs_once_the_worker_has_its_request(waiter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAITER_SLEEP", "1")
    child = pipeline._start_worker("waiter", sys.executable)
    request: Path = waiter / "request.json"
    seen: list[bool] = []

    def while_running() -> None:
        deadline: float = time.monotonic() + 20.0
        while not (request.is_file() and request.read_text()) and time.monotonic() < deadline:
            time.sleep(0.05)
        seen.append(child is not None and child.poll() is None)  # the worker has the request and is still running

    pipeline._run_worker("waiter", request, sys.executable, threading.Event(), child, while_running)
    assert seen == [True]


def test_an_error_in_while_running_stops_the_worker(waiter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAITER_SLEEP", "60")
    child = pipeline._start_worker("waiter", sys.executable)
    assert child is not None

    def while_running() -> None:
        raise RunCancelled("stopped in an observer")

    started: float = time.perf_counter()
    with pytest.raises(RunCancelled):
        pipeline._run_worker("waiter", waiter / "request.json", sys.executable, threading.Event(), child, while_running)
    assert child.returncode is not None and time.perf_counter() - started < 15.0
