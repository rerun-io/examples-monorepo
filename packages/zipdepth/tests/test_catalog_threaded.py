"""Behavioral tests for bounded threaded producers."""

import threading
from collections.abc import Generator, Iterator
from time import monotonic, sleep

import pytest

from zipdepth.catalog.threaded import iter_threaded


def test_iter_threaded_yields_every_worker_item() -> None:
    """Deliver every item from every producer through one bounded queue."""
    values: list[int] = list(iter_threaded([lambda: range(4), lambda: range(4, 9)], maxsize=2))

    assert sorted(values) == list(range(9))


def test_iter_threaded_propagates_worker_exceptions() -> None:
    """Raise a worker failure in the consuming thread."""

    def failing_worker() -> Iterator[int]:
        yield 1
        raise RuntimeError("threaded worker failed")

    with pytest.raises(RuntimeError, match="threaded worker failed"):
        list(iter_threaded([failing_worker], maxsize=1))


def test_early_close_leaves_no_live_producer_threads() -> None:
    """Stop producers blocked on a full queue when the consumer closes early."""

    def many_items() -> Iterator[int]:
        """Yield enough items to fill the bounded queue."""
        yield from range(10_000)

    samples: Generator[int, None, None] = iter_threaded([many_items, many_items], maxsize=1, join_timeout=0.1)
    next(samples)
    samples.close()

    deadline: float = monotonic() + 2.0
    live_producers: list[threading.Thread] = [
        thread for thread in threading.enumerate() if thread.name.startswith("zipdepth-producer-") and thread.is_alive()
    ]
    while live_producers and monotonic() < deadline:
        sleep(0.01)
        live_producers = [
            thread for thread in threading.enumerate() if thread.name.startswith("zipdepth-producer-") and thread.is_alive()
        ]

    assert live_producers == []


def test_failure_is_delivered_after_the_items_queued_before_it() -> None:
    """Preserve FIFO order between worker items and its terminal failure."""

    def failing_worker() -> Iterator[int]:
        """Queue two items before failing."""
        yield 1
        yield 2
        raise RuntimeError("failure after queued items")

    samples: Generator[int, None, None] = iter_threaded([failing_worker], maxsize=3)

    assert next(samples) == 1
    assert next(samples) == 2
    with pytest.raises(RuntimeError, match="failure after queued items"):
        next(samples)


def test_zero_workers_terminates() -> None:
    """Return immediately when no worker factories are supplied."""
    assert list(iter_threaded([], maxsize=1)) == []
