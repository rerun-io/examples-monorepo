"""The cube and panorama lens decoder's plan: runs a decoder reads front to back, each sample decoded once."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from jaxtyping import Int64
from numpy import ndarray


def decoded_samples(indices: Int64[ndarray, " k"], gop_starts: Int64[ndarray, " g"]) -> int:
    from dataforge.datasets.insta360 import decode_runs

    return sum(run.stop - run.start for run in decode_runs(indices, gop_starts))


def x5_gops(frames: int = 2498, gop: int = 60) -> Int64[ndarray, " g"]:
    return np.arange(0, frames, gop, dtype=np.int64)


def test_every_frame_of_the_x5_clip_is_decoded_once_in_runs_of_eight_gops() -> None:
    """Stride 1 on the X5 (GOP 60): 2,498 samples decoded, where a decoder per batch of 4 decoded 19,898."""
    from dataforge.datasets.insta360 import decode_runs

    indices = np.arange(2498, dtype=np.int64)
    runs = decode_runs(indices, x5_gops())
    assert sum(run.stop - run.start for run in runs) == 2498
    assert [run.start for run in runs] == [0, 480, 960, 1440, 1920, 2400]
    assert np.array_equal(np.concatenate([run.wanted for run in runs]), indices)


def test_a_sparse_stride_decodes_each_gop_up_to_its_last_wanted_picture() -> None:
    """Stride 10 at GOP 60: the last wanted picture of each GOP is its sample 50, so 51 of every 60 samples are decoded."""
    indices = np.arange(0, 2498, 10, dtype=np.int64)
    assert decoded_samples(indices, x5_gops()) == 41 * 51 + (2490 - 2460 + 1)


def test_a_stride_longer_than_a_gop_starts_a_run_at_each_picture_s_keyframe() -> None:
    from dataforge.datasets.insta360 import decode_runs

    runs = decode_runs(np.array([5, 130, 250], dtype=np.int64), x5_gops())
    assert [(run.start, run.stop) for run in runs] == [(0, 6), (120, 131), (240, 251)]


def test_the_plan_refuses_unsorted_pictures_and_gops_that_skip_sample_0() -> None:
    from dataforge.datasets.insta360 import decode_runs

    with pytest.raises(ValueError, match="increasing"):
        decode_runs(np.array([3, 3], dtype=np.int64), x5_gops())
    with pytest.raises(ValueError, match="sample 0"):
        decode_runs(np.array([3], dtype=np.int64), np.array([2, 60], dtype=np.int64))


@given(
    gop_lengths=st.lists(st.integers(1, 20), min_size=1, max_size=12),
    picks=st.lists(st.booleans(), min_size=1, max_size=240),
    max_gops=st.integers(1, 4),
)
def test_runs_decode_exactly_the_samples_from_each_picture_s_keyframe_once(gop_lengths: list[int], picks: list[bool], max_gops: int) -> None:
    """Whatever the GOPs and the pictures: runs start on keyframes, do not overlap, and decode the union of
    ``keyframe(p) .. p`` over the wanted pictures ``p``, no sample twice."""
    from dataforge.datasets.insta360 import decode_runs

    gop_starts = np.cumsum([0, *gop_lengths[:-1]]).astype(np.int64)
    frames = int(sum(gop_lengths))
    indices = np.flatnonzero(np.resize(np.asarray(picks), frames)).astype(np.int64)
    runs = decode_runs(indices, gop_starts, max_gops=max_gops)
    assert np.array_equal(np.concatenate([run.wanted for run in runs]) if runs else indices, indices)
    assert all(run.start in set(gop_starts.tolist()) and run.start <= run.wanted[0] and run.stop == run.wanted[-1] + 1 for run in runs)
    assert all(a.stop <= b.start for a, b in zip(runs[:-1], runs[1:], strict=True))
    needed: set[int] = set()
    for p in indices.tolist():
        needed.update(range(int(gop_starts[np.searchsorted(gop_starts, p, side="right") - 1]), p + 1))
    assert sum(run.stop - run.start for run in runs) == len(needed)


@given(sizes=st.lists(st.integers(1, 7), min_size=0, max_size=12), batch=st.integers(1, 9))
def test_rebatching_keeps_every_picture_in_order_in_full_batches(sizes: list[int], batch: int) -> None:
    import torch

    from dataforge.datasets.insta360 import rebatch

    chunks = []
    start = 0
    for size in sizes:
        chunks.append(torch.arange(start, start + size, dtype=torch.uint8).reshape(size, 1, 1, 1).expand(size, 3, 1, 1).contiguous())
        start += size
    batches = list(rebatch(iter(chunks), batch))
    assert [b.shape[0] for b in batches[:-1]] == [batch] * (len(batches) - 1) and (not batches or 0 < batches[-1].shape[0] <= batch)
    assert torch.cat(batches)[:, 0, 0, 0].tolist() == list(range(start)) if batches else start == 0
