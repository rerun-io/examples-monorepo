"""Run an iterator on one worker thread, a bounded number of items ahead of its consumer.

The frameset feed decodes in bursts (one NVDEC batch or one fetched window at a time); read inline, the tracker stops
for each burst. Ahead of it, on its own thread, decoding overlaps tracking: PyAV, NVDEC, the catalog reads and the Rust
tracker all release the GIL.
"""

import queue
import threading
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import cast

_PUT_POLL_S: float = 0.05
"""How often a worker blocked on a full queue checks for a stop; the consumer's close waits at most this long for it."""


@dataclass(slots=True, frozen=True)
class _Failure:
    """What the worker raised, carried to the consumer in the item's place."""

    error: BaseException


_END: object = object()
"""The worker's last item when its iterator ran out."""


def prefetch[T](items: Iterator[T], depth: int) -> Generator[T, None, None]:
    """Yield ``items`` in order while one worker thread advances them up to ``depth`` items ahead.

    ``items`` must be a generator not yet started (or another iterator safe to move to another thread): its body then
    runs, and is closed, on the worker alone, so a resource that is bound to one thread (a torchcodec NVDEC decoder)
    is opened, used and released on that thread. The worker holds at most ``depth`` finished items in the queue plus
    the one it is producing. Whatever the worker raises is raised here, at the position where the next item would
    have been. Closing this generator (``break``, an exception in the consumer, garbage collection) stops the worker,
    which closes ``items`` on its own thread, and returns once the worker has exited.

    Args:
        items: The iterator to advance on the worker.
        depth: Items the worker may hold finished ahead of the consumer; at least one.

    Yields:
        Each item of ``items``, in order.

    Raises:
        ValueError: If ``depth`` is below one.
    """
    if depth < 1:
        raise ValueError(f"prefetch depth must be at least 1; got {depth}")
    ready: queue.Queue[object] = queue.Queue(maxsize=depth)
    stop: threading.Event = threading.Event()

    def put(item: object) -> bool:
        while not stop.is_set():
            try:
                ready.put(item, timeout=_PUT_POLL_S)
                return True
            except queue.Full:
                pass
        return False

    def work() -> None:
        try:
            try:
                for item in items:
                    if not put(item):
                        return
            finally:
                if isinstance(items, Generator):
                    items.close()
            put(_END)
        except BaseException as error:  # every failure, beartype's included, goes to the consumer
            put(_Failure(error))

    worker: threading.Thread = threading.Thread(target=work, name="slam-rs-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item: object = ready.get()
            if item is _END:
                return
            if isinstance(item, _Failure):
                raise item.error
            yield cast(T, item)
    finally:
        stop.set()
        worker.join()
