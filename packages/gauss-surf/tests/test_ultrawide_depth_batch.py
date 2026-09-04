"""Behavior checks for the ultrawide depth batch queue split."""

import numpy as np
import pytest

from gauss_surf.apis.ultrawide_depth_batch import shard_video_ids

CORPUS_VIDEO_IDS: list[str] = [f"4244{index:04d}" for index in range(37)]


def test_shards_partition_the_queue_without_overlap() -> None:
    """Four shards together cover every queued segment exactly once."""
    shards: list[list[str]] = [shard_video_ids(CORPUS_VIDEO_IDS, index, 4) for index in range(4)]
    combined: list[str] = sorted(video_id for shard in shards for video_id in shard)
    assert combined == sorted(CORPUS_VIDEO_IDS)
    assert len(combined) == len(set(combined))
    sizes: list[int] = [len(shard) for shard in shards]
    assert max(sizes) - min(sizes) <= 1


def test_shard_is_independent_of_queue_order() -> None:
    """Shuffling the catalog row order leaves each shard's contents unchanged."""
    shuffled: list[str] = list(CORPUS_VIDEO_IDS)
    np.random.default_rng(0).shuffle(shuffled)
    assert shard_video_ids(shuffled, 2, 4) == shard_video_ids(CORPUS_VIDEO_IDS, 2, 4)


def test_single_shard_returns_the_sorted_queue() -> None:
    """One shard owns the whole queue in sorted order."""
    assert shard_video_ids(CORPUS_VIDEO_IDS, 0, 1) == sorted(CORPUS_VIDEO_IDS)


@pytest.mark.parametrize(("shard_index", "shard_count"), [(0, 0), (-1, 4), (4, 4)])
def test_invalid_shard_arguments_raise(shard_index: int, shard_count: int) -> None:
    """Out-of-range shard arguments fail loudly instead of silently dropping work."""
    with pytest.raises(ValueError):
        shard_video_ids(CORPUS_VIDEO_IDS, shard_index, shard_count)
