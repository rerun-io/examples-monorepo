import threading
from collections.abc import Generator

import pytest

pytest.importorskip("simplecv.rerun_dataloader", reason="mvs.catalog imports the simplecv Rerun dataloader")

import mvs.log_worker  # noqa: E402
from mvs.log_worker import MAX_IN_FLIGHT, LogWorker  # noqa: E402


@pytest.fixture
def times(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, object]]:
    """Record every rr.set_time the worker makes: (thread name, timeline, value)."""
    calls: list[tuple[str, str, object]] = []

    def record(timeline: str, **value: object) -> None:
        calls.append((threading.current_thread().name, timeline, next(iter(value.values()))))

    monkeypatch.setattr(mvs.log_worker.rr, "set_time", record)
    return calls


def test_jobs_run_in_order_on_one_thread_at_their_own_time(times: list[tuple[str, str, object]]) -> None:
    ran: list[tuple[str, int]] = []

    def job(frame: int) -> None:
        ran.append((threading.current_thread().name, frame))

    with LogWorker() as worker:
        frame: int
        for frame in range(6):
            worker.submit(1_000 * frame, frame, job, frame)

    assert [frame for _, frame in ran] == list(range(6))
    assert len({name for name, _ in ran}) == 1
    assert ran[0][0].startswith("rr-log")
    assert [value for _, timeline, value in times if timeline == "frame"] == list(range(6))
    assert all(name.startswith("rr-log") for name, _, _ in times)


def test_submit_waits_when_two_jobs_are_in_flight(times: list[tuple[str, str, object]]) -> None:
    release: threading.Event = threading.Event()
    started: list[int] = []

    def blocked(frame: int) -> None:
        started.append(frame)
        release.wait(timeout=5.0)

    with LogWorker() as worker:
        frame: int
        for frame in range(MAX_IN_FLIGHT):
            worker.submit(0, frame, blocked, frame)
        third: threading.Thread = threading.Thread(target=worker.submit, args=(0, MAX_IN_FLIGHT, blocked, MAX_IN_FLIGHT))
        third.start()
        third.join(timeout=0.2)
        assert third.is_alive(), "the third submit must wait for the oldest job"
        release.set()
        third.join(timeout=5.0)
    assert started == [0, 1, 2]


def test_a_worker_error_surfaces_in_the_main_thread(times: list[tuple[str, str, object]]) -> None:
    def fail() -> None:
        raise RuntimeError("encode failed")

    ran: list[int] = []
    with pytest.raises(RuntimeError, match="encode failed"), LogWorker() as worker:
        worker.submit(0, 0, fail)
        frame: int
        for frame in range(1, 5):
            worker.submit(0, frame, ran.append, frame)
    assert len(ran) < 4, "the main loop stops at the first wait that meets the error"


def test_closing_a_generator_early_drains_and_joins_the_worker(times: list[tuple[str, str, object]]) -> None:
    ran: list[int] = []

    def frames() -> Generator[int, None, None]:
        with LogWorker() as worker:
            frame: int
            for frame in range(100):
                worker.submit(0, frame, ran.append, frame)
                yield frame

    stream: Generator[int, None, None] = frames()
    assert [next(stream) for _ in range(3)] == [0, 1, 2]
    stream.close()

    assert ran == [0, 1, 2]
    assert not any(thread.name.startswith("rr-log") and thread.is_alive() for thread in threading.enumerate())
