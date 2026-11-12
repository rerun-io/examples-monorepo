"""One Rerun log thread beside the frame loop, as the original live-meshing demo runs it.

Ported from pablovela5620/simplerecon-inference ``live_meshing.py`` at d5bc1a0: a single worker thread encodes and
logs each frame's views while the main loop runs the model and the fusion; at most two jobs are in flight, so the
main loop waits (the ``rerun_log`` stage) only when the worker falls behind. One thread keeps the jobs, and so every
entity's rows, in submission order. Rerun timelines are per thread, so each job sets its own time point first.
"""

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Self

import numpy as np
import rerun as rr

from mvs.catalog import FRAME_TIMELINE, TIMELINE

MAX_IN_FLIGHT: int = 2
"""Jobs queued or running on the worker before ``submit`` waits for the oldest (the original's bound)."""


class LogWorker:
    """A single ``rr-log`` thread with back-pressure; use it as a context manager.

    Leaving the ``with`` block, normally, on an exception or when a generator that holds it is closed, waits for every
    queued job and joins the thread. A job's exception is raised in the main thread by the ``submit`` or exit that
    waits for it, so no log is lost silently.
    """

    def __init__(self) -> None:
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rr-log")
        # A maxlen deque would drop futures, and their exceptions, on overflow: submit waits instead.
        self._pending: deque[Future[None]] = deque()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            while self._pending:
                self._pending.popleft().result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def submit(self, timestamp_ns: int, frame_index: int, job: Callable[..., None], *args: object) -> None:
        """Queue ``job(*args)`` at this time point on both timelines, after waiting for the oldest job when two are
        in flight. The arguments must not change after the call: the job reads them later, on the worker."""
        if len(self._pending) >= MAX_IN_FLIGHT:
            self._pending.popleft().result()
        self._pending.append(self._executor.submit(_run_at, timestamp_ns, frame_index, job, args))


def _run_at(timestamp_ns: int, frame_index: int, job: Callable[..., None], args: tuple[object, ...]) -> None:
    """Set the worker thread's time point, then run the job."""
    rr.set_time(TIMELINE, duration=np.timedelta64(timestamp_ns, "ns"))
    rr.set_time(FRAME_TIMELINE, sequence=frame_index)
    job(*args)
