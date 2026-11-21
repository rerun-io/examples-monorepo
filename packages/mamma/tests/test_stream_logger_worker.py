"""StreamLogger's BoundedWorker: tasks run in order on one thread, at most ``depth`` queued (CPU, no Rerun sink)."""

from __future__ import annotations

import threading


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
