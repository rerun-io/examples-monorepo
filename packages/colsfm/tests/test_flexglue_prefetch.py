"""FlexGlue's descriptor prefetch: a batch is the images the pair loop needs next, not the ones after an image's first use."""

from __future__ import annotations

from collections import OrderedDict

import pytest

pytest.importorskip("trtkit", reason="colsfm.flexglue_trt imports trtkit's runtime")

from colsfm.flexglue_trt import DESCRIPTOR_BATCH, descriptor_batch  # noqa: E402


def _encodes(pairs: list[tuple[int, int]], capacity: int) -> int:
    """Images encoded walking `pairs` with an LRU of `capacity` and `descriptor_batch` choosing each batch."""
    cached: OrderedDict[int, None] = OrderedDict()
    need: list[int] = [image for pair in pairs for image in pair]
    encoded: int = 0
    for index, image in enumerate(need):
        if image in cached:
            cached.move_to_end(image)
            continue
        batch: list[int] = descriptor_batch(need, index, cached)
        assert batch[0] == image and len(batch) == len(set(batch)) <= DESCRIPTOR_BATCH
        assert not any(candidate in cached for candidate in batch)
        for candidate in batch:
            cached[candidate] = None
        encoded += len(batch)
        while len(cached) > capacity:
            cached.popitem(last=False)
    return encoded


def test_a_batch_is_the_next_uncached_images_the_loop_needs() -> None:
    need: list[int] = [1, 2, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11]
    assert descriptor_batch(need, 0, {}) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert descriptor_batch(need, 3, {1: None, 2: None}) == [3, 4, 5, 6, 7, 8, 9, 10]


def test_an_image_needed_again_after_eviction_prefetches_what_comes_next_not_what_followed_its_first_use() -> None:
    # Image 0 is needed first, then evicted by a long run of other images, then needed again with 900.. next.
    pairs: list[tuple[int, int]] = [(0, 1)] + [(i, i + 1) for i in range(2, 600)] + [(0, 900 + k) for k in range(8)]
    need: list[int] = [image for pair in pairs for image in pair]
    batch: list[int] = descriptor_batch(need, need.index(0, 2), {})
    assert batch == [0, 900, 901, 902, 903, 904, 905, 906]


def test_revisit_pairs_encode_each_image_at_most_twice_with_a_small_cache() -> None:
    sequence: list[tuple[int, int]] = [(i, i + 1) for i in range(2000)]
    revisits: list[tuple[int, int]] = sorted((i, 1000 + (7 * i) % 1000) for i in range(0, 1000, 2))
    pairs: list[tuple[int, int]] = sequence + revisits
    unique: int = len({image for pair in pairs for image in pair})
    assert _encodes(pairs, 64) <= 2 * unique
