"""FlexGlue's descriptor prefetch: a batch is the images the pair loop needs next, not the ones after an image's first use."""

from __future__ import annotations

from collections import OrderedDict

import pytest

pytest.importorskip("trtkit", reason="colsfm.flexglue_trt imports trtkit's runtime")

from colsfm.flexglue_trt import DESCRIPTOR_BATCH, descriptor_batch, pair_schedule  # noqa: E402


class _FakeMatcher:
    """The production pair loop's CPU stand-in: an LRU feature cache, a descriptor call that fills it, an enqueue that
    refreshes both endpoints (as `FlexGlueRunner.enqueue` does) and needs both cached."""

    def __init__(self, capacity: int) -> None:
        self.images: OrderedDict[int, None] = OrderedDict()
        self.capacity: int = capacity
        self.calls: int = 0
        self.encoded: int = 0
        self.enqueued: list[tuple[int, int]] = []

    def encode(self, batch: list[int]) -> None:
        assert len(batch) == len(set(batch)) <= DESCRIPTOR_BATCH
        assert not any(image in self.images for image in batch), "a batch only holds uncached images"
        self.calls += 1
        self.encoded += len(batch)
        for image in batch:
            self.images[image] = None
        while len(self.images) > self.capacity:
            self.images.popitem(last=False)

    def enqueue(self, pair: tuple[int, int]) -> None:
        assert pair[0] in self.images and pair[1] in self.images, f"{pair} enqueued without both images cached"
        self.images.move_to_end(pair[0])
        self.images.move_to_end(pair[1])
        self.enqueued.append(pair)


def _run(pairs: list[tuple[int, int]], capacity: int) -> _FakeMatcher:
    matcher = _FakeMatcher(capacity)
    for index in pair_schedule(pairs, matcher.images, matcher.encode):
        matcher.enqueue(pairs[index])
    assert matcher.enqueued == pairs, "every pair, once, in order"
    return matcher


def test_a_batch_is_the_next_uncached_images_from_the_current_endpoint() -> None:
    pairs: list[tuple[int, int]] = [(1, 2), (1, 3), (2, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    assert descriptor_batch(pairs, 0, 0, {}) == [1, 2, 3, 4, 5, 6, 7, 8]
    # A hit on the first image, then a miss on the second: the batch starts at the second endpoint.
    assert descriptor_batch(pairs, 1, 1, {1: None, 2: None}) == [3, 4, 5, 6, 7, 8, 9, 10]


def test_an_image_needed_again_after_eviction_prefetches_what_comes_next_not_what_followed_its_first_use() -> None:
    # Image 0 is needed first, then evicted by a long run of other images, then needed again with 900.. next.
    pairs: list[tuple[int, int]] = [(0, 1)] + [(i, i + 1) for i in range(2, 600)] + [(0, 900 + k) for k in range(8)]
    assert descriptor_batch(pairs, 599, 0, {}) == [0, 900, 901, 902, 903, 904, 905, 906]


def test_the_pair_loop_batches_its_misses() -> None:
    """The production loop, not a copy: a sequential pass encodes each image once in full batches."""
    pairs: list[tuple[int, int]] = [(i, i + 1) for i in range(400)]
    matcher = _run(pairs, 64)
    assert matcher.encoded == 401
    assert matcher.calls <= -(-401 // DESCRIPTOR_BATCH) + 1, f"{matcher.calls} descriptor calls: look-ahead is not batching"


def test_revisit_pairs_encode_each_image_at_most_twice_with_a_small_cache() -> None:
    sequence: list[tuple[int, int]] = [(i, i + 1) for i in range(2000)]
    revisits: list[tuple[int, int]] = sorted((i, 1000 + (7 * i) % 1000) for i in range(0, 1000, 2))
    pairs: list[tuple[int, int]] = sequence + revisits
    unique: int = len({image for pair in pairs for image in pair})
    matcher = _run(pairs, 64)
    assert matcher.encoded <= 2 * unique
    assert matcher.calls <= matcher.encoded // 4, f"{matcher.calls} descriptor calls for {matcher.encoded} images: batches too small"
