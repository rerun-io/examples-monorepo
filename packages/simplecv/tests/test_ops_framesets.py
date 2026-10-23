"""Frameset grouping (``simplecv.ops.framesets``): a worked example plus hypothesis properties (unit lane)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Int64
from numpy import ndarray

from simplecv.ops.framesets import FramesetIndex, clear_times, group_framesets


@st.composite
def camera_times(draw: st.DrawFn) -> list[Int64[ndarray, " n"]]:
    """1-6 cameras, each with strictly increasing frame times (ns) on a common jittered 33 ms grid."""
    cameras: int = draw(st.integers(1, 6))
    times: list[Int64[ndarray, " n"]] = []
    for _ in range(cameras):
        steps: list[int] = draw(st.lists(st.integers(1, 40_000_000), min_size=0, max_size=30))
        start: int = draw(st.integers(0, 10_000_000))
        times.append(start + np.cumsum(np.asarray(steps, dtype=np.int64)))
    return times


def test_a_frameset_opens_at_the_earliest_frame_and_takes_each_cameras_next_frame_within_the_tolerance() -> None:
    cameras = [
        np.array([0, 33_000_000, 66_000_000], dtype=np.int64),
        np.array([1_000_000, 34_500_000, 67_000_000], dtype=np.int64),
        np.array([2_500_000, 40_000_000], dtype=np.int64),  # starts late and drifts out of the tolerance on its second frame
    ]
    index: FramesetIndex = group_framesets(cameras, tolerance_ns=3_000_000)
    assert index.t_ns.tolist() == [0, 33_000_000, 40_000_000, 66_000_000]
    assert index.frame.tolist() == [[0, 0, 0], [1, 1, -1], [-1, -1, 1], [2, 2, -1]]
    assert index.cam_t_ns[1].tolist() == [33_000_000, 34_500_000, 0]


def test_grouping_refuses_frame_times_that_do_not_increase_and_a_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="camera 1"):
        group_framesets([np.array([0, 1], dtype=np.int64), np.array([5, 5], dtype=np.int64)], tolerance_ns=3)
    with pytest.raises(ValueError, match="negative"):
        group_framesets([np.array([0, 1], dtype=np.int64)], tolerance_ns=-1)


@settings(max_examples=200, deadline=None)
@given(times=camera_times(), tolerance_ns=st.integers(0, 20_000_000))
def test_every_frame_lands_in_exactly_one_frameset_in_order(times: list[Int64[ndarray, " n"]], tolerance_ns: int) -> None:
    index: FramesetIndex = group_framesets(times, tolerance_ns)
    for camera, stamps in enumerate(times):
        taken: Int64[ndarray, " k"] = index.frame[:, camera][index.frame[:, camera] >= 0]
        np.testing.assert_array_equal(taken, np.arange(stamps.size))  # each frame once, in decode order
        np.testing.assert_array_equal(index.cam_t_ns[index.frame[:, camera] >= 0, camera], stamps)


@settings(max_examples=200, deadline=None)
@given(times=camera_times(), tolerance_ns=st.integers(0, 20_000_000))
def test_members_sit_within_the_tolerance_of_the_opening_frame_and_openings_strictly_increase(
    times: list[Int64[ndarray, " n"]], tolerance_ns: int
) -> None:
    index: FramesetIndex = group_framesets(times, tolerance_ns)
    present = index.frame >= 0
    assert np.all(present.any(axis=1))  # no empty frameset
    offsets: Int64[ndarray, "f c"] = np.where(present, index.cam_t_ns - index.t_ns[:, None], 0)
    assert np.all((offsets >= 0) & (offsets <= tolerance_ns))
    assert np.all(index.t_ns == np.where(present, index.cam_t_ns, np.iinfo(np.int64).max).min(axis=1))  # opens at its earliest member
    assert np.all(np.diff(index.t_ns) > 0)


@settings(max_examples=100, deadline=None)
@given(cameras=st.integers(1, 6), stamps=st.lists(st.integers(0, 10**12), min_size=1, max_size=40, unique=True))
def test_tolerance_zero_with_equal_stamps_gives_full_framesets(cameras: int, stamps: list[int]) -> None:
    shared: Int64[ndarray, " n"] = np.sort(np.asarray(stamps, dtype=np.int64))
    index: FramesetIndex = group_framesets([shared.copy() for _ in range(cameras)], tolerance_ns=0)
    np.testing.assert_array_equal(index.t_ns, shared)
    np.testing.assert_array_equal(index.frame, np.repeat(np.arange(shared.size)[:, None], cameras, axis=1))


def test_a_drawing_clears_at_its_camera_s_next_frame_unless_that_frame_draws_again() -> None:
    """Frames 2 and 3 draw (3 replaces 2, so 2 needs no clear), frame 5 draws and 6 does not (it clears at 6), and the last
    frame has no next frame. The clock starts below zero: frame 4 is at -1 ns, a time like any other."""
    times: Int64[ndarray, " 10"] = 1_000 * np.arange(10, dtype=np.int64) - 4_001
    assert clear_times(times, np.array([2, 3, 5, 9], dtype=np.int64)) == [None, -1, 1_999, None]


@settings(max_examples=200, deadline=None)
@given(offset=st.integers(-(10**15), 10**15), drawn=st.sets(st.integers(0, 9), min_size=1))
def test_moving_the_clock_origin_moves_every_clear_with_it(offset: int, drawn: set[int]) -> None:
    times: Int64[ndarray, " 10"] = 1_000 * np.arange(10, dtype=np.int64)
    frames: Int64[ndarray, " k"] = np.array(sorted(drawn), dtype=np.int64)
    assert clear_times(times + offset, frames) == [None if t is None else t + offset for t in clear_times(times, frames)]
