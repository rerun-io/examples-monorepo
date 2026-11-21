"""StreamLogger's BoundedWorker: tasks run in order on one thread, at most ``depth`` queued (CPU, no Rerun sink)."""

from __future__ import annotations

import threading

import pytest


def test_bounded_worker_runs_in_order_and_applies_backpressure() -> None:
    from mamma.viz.stream_logger import BoundedWorker

    release = threading.Event()
    done: list[int] = []
    def first() -> None:
        release.wait()
        done.append(0)

    worker = BoundedWorker("test", depth=2)
    worker.submit(first)
    worker.submit(done.append, 1)
    # A third submit waits for the oldest task, so release it from another thread.
    threading.Timer(0.05, release.set).start()
    worker.submit(done.append, 2)
    assert done[:1] == [0]
    worker.drain()
    assert done == [0, 1, 2]


@pytest.mark.integration
def test_fit_log_worker_reads_clouds_after_the_producing_stream_wrote_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cloud is written on a side stream after a long GPU sleep; the fit-log worker must read the written values
    (without the readiness event it reads the zeros the tensor held when log_tick_fit returned)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    import mamma.viz.stream_logger as stream_logger
    from mamma.viz.stream_logger import BoundedWorker, StreamLogger

    logged: dict[str, object] = {}
    monkeypatch.setattr(stream_logger.rr, "log", lambda entity, value, **_: logged.setdefault(entity, value))
    monkeypatch.setattr(stream_logger.rr, "Points3D", lambda positions, **_: positions)
    monkeypatch.setattr(stream_logger, "set_tick_time", lambda *_: None)
    logger = StreamLogger.__new__(StreamLogger)
    logger._fit_worker = BoundedWorker("test")
    logger.fps = 30.0
    target = torch.arange(30, dtype=torch.float32, device="cuda").reshape(10, 3)
    points = torch.zeros(10, 3, device="cuda")
    valid = torch.ones(10, dtype=torch.bool, device="cuda")
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(200_000_000)  # ~0.1 s of GPU time before the write
        points.copy_(target)
        logger.log_tick_fit(0, {}, {0: (points, valid)}, None)
    logger._fit_worker.drain()
    assert (logged[f"{stream_logger.WORLD_TAG}/triangulated/person_0"] == target.cpu().numpy()).all()
