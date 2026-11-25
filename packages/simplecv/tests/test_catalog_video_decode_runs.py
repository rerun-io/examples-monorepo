"""The decode-run planner: runs a decoder reads front to back from a keyframe, each sample decoded once."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from jaxtyping import Int64
from numpy import ndarray


def gops_of_60(frames: int = 2498) -> Int64[ndarray, " g"]:
    return np.arange(0, frames, 60, dtype=np.int64)


def test_a_sparse_stride_decodes_each_gop_up_to_its_last_wanted_picture() -> None:
    """Stride 10 at GOP 60: the last wanted picture of each GOP is its sample 50, so 51 of every 60 samples are decoded."""
    from simplecv.catalog_video_decode import decode_runs

    runs = decode_runs(np.arange(0, 2498, 10, dtype=np.int64), gops_of_60(), max_gops=8)
    assert sum(run.stop - run.start for run in runs) == 41 * 51 + (2490 - 2460 + 1)


def test_a_stride_longer_than_a_gop_starts_a_run_at_each_picture_s_keyframe() -> None:
    from simplecv.catalog_video_decode import decode_runs

    runs = decode_runs(np.array([5, 130, 250], dtype=np.int64), gops_of_60())
    assert [(run.start, run.stop) for run in runs] == [(0, 6), (120, 131), (240, 251)]


def test_adjacent_gops_share_a_run_without_a_cap_and_split_at_one_gop_per_run() -> None:
    from simplecv.catalog_video_decode import decode_runs

    keyframes: Int64[ndarray, " g"] = np.array([0, 4, 8, 12], dtype=np.int64)
    wanted: Int64[ndarray, " k"] = np.array([1, 2, 3, 4, 9, 15], dtype=np.int64)
    assert [(run.start, run.stop) for run in decode_runs(wanted, keyframes)] == [(0, 5), (8, 10), (12, 16)]
    one_gop = decode_runs(wanted, keyframes, max_gops=1)
    assert [(run.start, run.wanted.tolist()) for run in one_gop] == [(0, [1, 2, 3]), (4, [4]), (8, [9]), (12, [15])]


def test_the_plan_refuses_unsorted_pictures_and_pictures_before_the_first_keyframe() -> None:
    from simplecv.catalog_video_decode import decode_runs

    with pytest.raises(ValueError, match="increasing"):
        decode_runs(np.array([3, 3], dtype=np.int64), gops_of_60())
    with pytest.raises(ValueError, match="no keyframe at or before picture 1"):
        decode_runs(np.array([1, 3], dtype=np.int64), np.array([2, 60], dtype=np.int64))
    assert [run.start for run in decode_runs(np.array([3], dtype=np.int64), np.array([2, 60], dtype=np.int64))] == [2]


@given(
    gop_lengths=st.lists(st.integers(1, 20), min_size=1, max_size=12),
    picks=st.lists(st.booleans(), min_size=1, max_size=240),
    max_gops=st.one_of(st.none(), st.integers(1, 4)),
)
def test_runs_decode_exactly_the_samples_from_each_picture_s_keyframe_once(gop_lengths: list[int], picks: list[bool], max_gops: int | None) -> None:
    """Whatever the GOPs and the pictures: runs start on keyframes, do not overlap, hold at most ``max_gops`` GOPs, and
    decode the union of ``keyframe(p) .. p`` over the wanted pictures ``p``, no sample twice."""
    from simplecv.catalog_video_decode import decode_runs

    gop_starts = np.cumsum([0, *gop_lengths[:-1]]).astype(np.int64)
    frames = int(sum(gop_lengths))
    indices = np.flatnonzero(np.resize(np.asarray(picks), frames)).astype(np.int64)
    runs = decode_runs(indices, gop_starts, max_gops=max_gops)
    assert np.array_equal(np.concatenate([run.wanted for run in runs]) if runs else indices, indices)
    assert all(run.start in set(gop_starts.tolist()) and run.start <= run.wanted[0] and run.stop == run.wanted[-1] + 1 for run in runs)
    assert all(a.stop <= b.start for a, b in zip(runs[:-1], runs[1:], strict=True))
    if max_gops is not None:
        assert all(np.searchsorted(gop_starts, run.stop - 1, side="right") - np.searchsorted(gop_starts, run.start, side="right") < max_gops for run in runs)
    needed: set[int] = set()
    for p in indices.tolist():
        needed.update(range(int(gop_starts[np.searchsorted(gop_starts, p, side="right") - 1]), p + 1))
    assert sum(run.stop - run.start for run in runs) == len(needed)
