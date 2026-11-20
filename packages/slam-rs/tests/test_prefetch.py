"""The frameset prefetch: order, a bounded queue, a worker error reaching the consumer, and a clean stop."""

import threading
import time
from collections.abc import Callable, Generator

import pytest

from slam_rs.prefetch import prefetch


def _counting(produced: list[int]) -> Generator[int, None, None]:
    index: int = 0
    while True:
        produced.append(index)
        yield index
        index += 1


def _wait_for(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline: float = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def test_items_arrive_in_order_and_complete() -> None:
    assert list(prefetch((index for index in range(1000)), depth=3)) == list(range(1000))


def test_the_worker_holds_at_most_depth_items_ahead() -> None:
    depth: int = 4
    produced: list[int] = []
    items: Generator[int, None, None] = prefetch(_counting(produced), depth)
    assert next(items) == 0
    # The queue holds items 1..depth and the worker waits to put item depth + 1; it produces nothing more.
    _wait_for(lambda: len(produced) == depth + 2)
    time.sleep(0.2)
    assert len(produced) == depth + 2
    assert next(items) == 1
    _wait_for(lambda: len(produced) == depth + 3)
    items.close()


def test_a_worker_error_reaches_the_consumer_after_the_items_before_it() -> None:
    def failing() -> Generator[int, None, None]:
        yield 0
        yield 1
        raise ValueError("cam_01 ran out of frames")

    items: Generator[int, None, None] = prefetch(failing(), depth=8)
    assert next(items) == 0
    assert next(items) == 1
    with pytest.raises(ValueError, match="ran out of frames"):
        next(items)


def test_closing_stops_the_worker_and_closes_the_source_on_the_worker_thread() -> None:
    closed_on: list[str] = []

    def endless() -> Generator[int, None, None]:
        try:
            index: int = 0
            while True:
                yield index
                index += 1
        finally:
            closed_on.append(threading.current_thread().name)

    items: Generator[int, None, None] = prefetch(endless(), depth=2)
    assert [next(items) for _ in range(3)] == [0, 1, 2]
    items.close()
    assert closed_on == ["prefetch"]
    assert not any(thread.name == "prefetch" for thread in threading.enumerate())


def test_a_depth_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        next(prefetch((index for index in [1]), depth=0))
