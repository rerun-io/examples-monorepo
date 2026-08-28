"""Behavioral tests for the bounded shuffle buffer and threaded producers."""

from collections.abc import Iterator

import pytest

from zipdepth.catalog.threaded import ShuffleBuffer, iter_threaded


def test_shuffle_buffer_is_seeded_and_preserves_every_sample() -> None:
    """Produce one deterministic streaming permutation without drops or duplicates."""

    def shuffled(seed: int) -> list[int]:
        buffer: ShuffleBuffer[int] = ShuffleBuffer(size=3, seed=seed)
        output: list[int] = []
        value: int
        for value in range(10):
            evicted: int | None = buffer.push(value)
            if evicted is not None:
                output.append(evicted)
        output.extend(buffer.flush())
        return output

    first: list[int] = shuffled(17)
    second: list[int] = shuffled(17)

    assert first == second
    assert sorted(first) == list(range(10))


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
