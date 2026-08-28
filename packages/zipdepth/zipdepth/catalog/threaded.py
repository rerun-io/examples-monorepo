"""Bounded producer/consumer iteration shared by catalog pipelines."""

from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass
from queue import Full, Queue
from random import Random
from threading import Event, Thread
from typing import Generic, TypeVar
from warnings import warn

ItemT = TypeVar("ItemT")


class ShuffleBuffer(Generic[ItemT]):  # noqa: UP046 — beartype does not support PEP 695 generics
    """Seeded streaming shuffle with bounded memory."""

    def __init__(self, size: int, seed: int) -> None:
        """Create an empty buffer."""
        if size <= 0:
            raise ValueError("shuffle buffer size must be positive")
        self._size: int = size
        self._rng: Random = Random(seed)
        self._samples: list[ItemT] = []

    def push(self, sample: ItemT) -> ItemT | None:
        """Add one sample and return a random eviction when the buffer is full."""
        if len(self._samples) < self._size:
            self._samples.append(sample)
            return None
        index: int = self._rng.randrange(self._size)
        evicted: ItemT = self._samples[index]
        self._samples[index] = sample
        return evicted

    def flush(self) -> Generator[ItemT, None, None]:
        """Yield all remaining samples in seeded random order and empty the buffer."""
        remaining: list[ItemT] = list(self._samples)
        self._samples.clear()
        self._rng.shuffle(remaining)
        yield from remaining


@dataclass(slots=True)
class _Failure:
    """Exception transported from a producer thread to the consumer."""

    error: BaseException
    """Original worker exception with its traceback."""


_END: object = object()
"""Queue sentinel emitted once by each normally exhausted worker."""


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

    queue: Queue[ItemT | _Failure | object] = Queue(maxsize=maxsize)
    stop: Event = Event()

    def put_unless_stopped(item: ItemT | _Failure | object) -> bool:
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
            put_unless_stopped(_END)
        except BaseException as error:
            if put_unless_stopped(_Failure(error)):
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
            queued: ItemT | _Failure | object = queue.get()
            if queued is _END:
                completed += 1
                continue
            if isinstance(queued, _Failure):
                raise queued.error
            yield queued  # type: ignore[misc]  # narrowed by the private queue transports above
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                warn(f"catalog producer thread {thread.name!r} did not stop within {join_timeout:.1f}s", RuntimeWarning, stacklevel=2)
