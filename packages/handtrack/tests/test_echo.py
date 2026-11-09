from collections import Counter
from collections.abc import Iterator

import pytest
import torch
from torch import Tensor

from handtrack.data.batches import DetNetBatch
from handtrack.data.echo import EchoStream
from handtrack.data.stream import DetNetSamples

CHUNK: int = 32


def _chunks(dataset: int, images: int) -> Iterator[DetNetSamples]:
    """Fake decoded images; each carries a unique id (dataset * 10000 + n) in two pooled pixels."""
    for start in range(0, images, CHUNK):
        n: int = min(CHUNK, images - start)
        ids: Tensor = torch.arange(start, start + n) + dataset * 10000
        pooled: Tensor = torch.zeros((n, 120, 160), dtype=torch.uint8)
        pooled[:, 0, 0] = (ids // 256).to(torch.uint8)
        pooled[:, 0, 1] = (ids % 256).to(torch.uint8)
        yield DetNetSamples(
            pooled=pooled, circle=torch.zeros((n, 2, 3)), presence=torch.zeros((n, 2)), circle_mask=torch.zeros((n, 2), dtype=torch.bool),
            presence_mask=torch.ones((n, 2), dtype=torch.bool), dataset=torch.full((n,), dataset), points=torch.zeros((n, 2, 21, 2)),
            in_front=torch.zeros((n, 2, 21), dtype=torch.bool), camera=torch.zeros(n, dtype=torch.int64),
        )


def _ids(batch: DetNetBatch) -> list[int]:
    pixels: Tensor = (batch.pooled[:, 0, 0, :2] * 255.0).round().long()
    return (pixels[:, 0] * 256 + pixels[:, 1]).tolist()


def _stream(images: int, *, max_echo: int, batch_size: int = 64) -> EchoStream:
    producers = [lambda epoch, d=d: _chunks(d, images) for d in range(4)]
    return EchoStream(producers, batch_size=batch_size, pool_capacity=512, min_fill=64, max_echo=max_echo, device=torch.device("cpu"), seed=0)


def _drain(stream: EchoStream) -> list[DetNetBatch]:
    stream.start_epoch(0)
    batches: list[DetNetBatch] = []
    while (batch := stream.next_detnet_batch()) is not None:
        batches.append(batch)
    stream.close()
    return batches


def test_every_batch_is_full_and_split_equally_across_the_datasets() -> None:
    batches = _drain(_stream(2000, max_echo=1))
    assert batches
    for batch in batches:
        assert batch.pooled.shape == (64, 1, 120, 160)
        assert Counter(batch.dataset.tolist()) == {0: 16, 1: 16, 2: 16, 3: 16}


def test_each_decoded_image_is_reused_up_to_the_echo_limit_and_never_twice_in_a_batch() -> None:
    batches = _drain(_stream(256, max_echo=3))
    uses: Counter[int] = Counter(i for batch in batches for i in _ids(batch))
    assert max(uses.values()) == 3
    assert len(uses) == 4 * 256
    for batch in batches:
        assert len(set(_ids(batch))) == len(_ids(batch))


def test_a_producer_failure_reaches_the_trainer() -> None:
    def broken(epoch: int) -> Iterator[DetNetSamples]:
        yield from _chunks(0, 64)
        raise RuntimeError("decode failed")

    producers = [broken, *[lambda epoch, d=d: _chunks(d, 4000) for d in range(1, 4)]]
    stream = EchoStream(producers, batch_size=64, pool_capacity=512, min_fill=64, max_echo=1, device=torch.device("cpu"), seed=0)
    stream.start_epoch(0)
    with pytest.raises(RuntimeError, match="decode failed"):
        while stream.next_detnet_batch() is not None:
            pass
    stream.close()


def test_a_step_budget_epoch_ends_on_time_and_the_next_epoch_keeps_the_running_producers() -> None:
    starts: Counter[int] = Counter()

    def endless(dataset: int):
        def produce(epoch: int) -> Iterator[DetNetSamples]:
            starts[dataset] += 1
            while True:
                yield from _chunks(dataset, 512)
        return produce

    stream = EchoStream([endless(d) for d in range(4)], batch_size=64, pool_capacity=512, min_fill=64, max_echo=2,
                        device=torch.device("cpu"), seed=0, epoch_steps=5)
    counts: list[int] = []
    for epoch in range(3):
        stream.start_epoch(epoch)
        n: int = 0
        while stream.next_detnet_batch() is not None:
            n += 1
        counts.append(n)
    stream.close()
    assert counts == [5, 5, 5]
    assert starts == {0: 1, 1: 1, 2: 1, 3: 1}


def test_producers_only_replace_images_drawn_at_least_min_echo_times() -> None:
    def endless(dataset: int):
        def produce(epoch: int) -> Iterator[DetNetSamples]:
            yield from _chunks(dataset, 9000)
        return produce

    stream = EchoStream([endless(d) for d in range(4)], batch_size=64, pool_capacity=128, min_fill=64, max_echo=8, min_echo=3,
                        device=torch.device("cpu"), seed=0, epoch_steps=200)
    batches = _drain(stream)
    uses: Counter[int] = Counter(i for batch in batches for i in _ids(batch))
    first_seen: list[int] = list(dict.fromkeys(i for batch in batches for i in _ids(batch)))
    settled: list[int] = first_seen[: len(first_seen) // 2]
    assert settled and min(uses[i] for i in settled) >= 3


@pytest.mark.parametrize("prefetch", [0, 2])
def test_prefetching_keeps_full_equal_share_batches(prefetch: int) -> None:
    producers = [lambda epoch, d=d: _chunks(d, 2000) for d in range(4)]
    stream = EchoStream(producers, batch_size=64, pool_capacity=512, min_fill=64, max_echo=1, device=torch.device("cpu"), seed=0, prefetch=prefetch)
    batches = _drain(stream)
    assert batches
    assert all(Counter(batch.dataset.tolist()) == {0: 16, 1: 16, 2: 16, 3: 16} for batch in batches)
