"""Bounded threaded fan-in for catalog producer pipelines."""

from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Thread
from typing import TypeVar
from warnings import warn

ItemT = TypeVar("ItemT")


@dataclass(slots=True)
class _Done:
    """Terminal message transported from one producer to the consumer."""

    error: BaseException | None
    """Original worker exception with its traceback, or None after normal exhaustion."""


def iter_threaded(  # noqa: UP047 — beartype does not support PEP 695 type parameters
    workers: Sequence[Callable[[], Iterable[ItemT]]],
    maxsize: int,
    join_timeout: float = 30.0,
) -> Generator[ItemT, None, None]:
    """Yield items from bounded worker threads and propagate their failures.

    Args:
        workers: Zero-argument factories whose iterables run in separate threads.
        maxsize: Maximum items buffered between all workers and the consumer.
        join_timeout: Seconds to wait for each worker during generator cleanup.

    Yields:
        Items from every worker in arrival order.

    Raises:
        ValueError: If the queue capacity or join timeout is not positive.
        BaseException: Any exception raised by a worker.
    """
    if maxsize <= 0:
        raise ValueError("threaded iterator maxsize must be positive")
    if join_timeout <= 0.0:
        raise ValueError("threaded iterator join_timeout must be positive")

    queue: Queue[ItemT | _Done] = Queue(maxsize=maxsize)
    stop: Event = Event()

    def put_unless_stopped(item: ItemT | _Done) -> bool:
        """Bound queue writes while allowing a closed consumer to stop workers."""
        while not stop.is_set():
            try:
                queue.put(item, timeout=0.05)
                return True
            except Full:
                continue
        return False

    def produce(worker: Callable[[], Iterable[ItemT]]) -> None:
        """Exhaust one worker iterable into the shared queue."""
        try:
            item: ItemT
            for item in worker():
                if not put_unless_stopped(item):
                    return
            put_unless_stopped(_Done(None))
        except BaseException as error:
            if put_unless_stopped(_Done(error)):
                stop.set()

    threads: list[Thread] = [
        Thread(target=produce, args=(worker,), name=f"zipdepth-producer-{index}", daemon=True)
        for index, worker in enumerate(workers)
    ]
    thread: Thread
    for thread in threads:
        thread.start()
    completed: int = 0
    try:
        while completed < len(threads):
            queued: ItemT | _Done = queue.get()
            if isinstance(queued, _Done):
                if queued.error is not None:
                    raise queued.error
                completed += 1
                continue
            yield queued
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                warn(f"catalog producer thread {thread.name!r} did not stop within {join_timeout:.1f}s", RuntimeWarning, stacklevel=2)
