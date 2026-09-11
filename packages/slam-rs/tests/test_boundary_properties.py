"""Property tests of the estimator's inertial boundary, driven through ``slam_rs._core`` (D23).

Hypothesis drives the PyO3 boundary directly: no CLI oracle. Examples are capped
so the whole file stays inside the default, seconds-long suite. The image rules
both entry points share — rank, dtype, layout, the frameset's width and the
calibrated frame size — are parametrized over ``Vio.track`` and
``OpticalFlow.process`` in ``test_frontend_boundary.py`` rather than written once
per entry point; what is left here is the IMU clock and the determinism of a run,
which only the estimator has.
"""

from typing import cast

import numpy as np
import pytest
from fixture_types import FRAME_PERIOD_NS, IMU_PERIOD_NS, PipelineFactory, TextureFactory, gravity_batch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from numpy.typing import NDArray

from slam_rs import _core

MAX_EXAMPLES: int = 50

gaps = st.lists(st.integers(min_value=1, max_value=10**6), min_size=1, max_size=20)
starts = st.integers(min_value=-(10**9), max_value=10**9)
positions = st.integers(min_value=0, max_value=1000)

def increasing(start: int, steps: list[int]) -> NDArray[np.int64]:
    """Timestamps that strictly increase from ``start`` by the given positive steps."""
    return np.array(np.cumsum([start, *steps]), dtype=np.int64)


def zeros(count: int) -> NDArray[np.float64]:
    """A C-contiguous ``(count, 3)`` block of zeros."""
    return np.zeros((count, 3), dtype=np.float64)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(start=starts, steps=gaps)
def test_strictly_increasing_timestamps_are_accepted(pipeline: PipelineFactory, start: int, steps: list[int]) -> None:
    vio: _core.Vio = pipeline(2)
    for t_ns in increasing(start, steps):
        vio.push_imu(int(t_ns), [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(start=starts, steps=gaps, back=st.integers(min_value=0, max_value=10**6))
def test_any_non_increasing_timestamp_is_rejected(pipeline: PipelineFactory, start: int, steps: list[int], back: int) -> None:
    vio: _core.Vio = pipeline(2)
    times: NDArray[np.int64] = increasing(start, steps)
    for t_ns in times:
        vio.push_imu(int(t_ns), [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
    with pytest.raises(ValueError, match="does not follow"):
        vio.push_imu(int(times[-1]) - back, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(start=starts, steps=gaps)
def test_an_increasing_batch_is_accepted(pipeline: PipelineFactory, start: int, steps: list[int]) -> None:
    vio: _core.Vio = pipeline(2)
    times: NDArray[np.int64] = increasing(start, steps)
    vio.push_imu_batch(times, zeros(len(times)), zeros(len(times)))


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(start=starts, steps=gaps, index=positions)
def test_a_batch_that_repeats_one_timestamp_keeps_none_of_it(pipeline: PipelineFactory, start: int, steps: list[int], index: int) -> None:
    """A rejected batch is rejected whole, wherever the bad sample sits in it.

    The batch used to be pushed sample by sample and to return at the first
    refusal, so the samples before it stayed and the frontier moved past them:
    the caller could then neither retry the batch nor correct it, because its own
    first samples had become too old. The state that says so is the frontier, and
    what reads it is the same batch, corrected: it is accepted here from its very
    first sample, which it could not be if anything of the rejected one had been
    kept.
    """
    times: NDArray[np.int64] = increasing(start, steps)  # at least two samples
    position: int = 1 + index % (len(times) - 1)
    broken: NDArray[np.int64] = times.copy()
    broken[position] = broken[position - 1]
    vio: _core.Vio = pipeline(2)
    with pytest.raises(ValueError, match="does not follow"):
        vio.push_imu_batch(broken, zeros(len(broken)), zeros(len(broken)))
    vio.push_imu_batch(times, zeros(len(times)), zeros(len(times)))


def test_a_rejected_batch_leaves_the_frontier_where_it_was(pipeline: PipelineFactory) -> None:
    """Codex's case verbatim: ``[10, 20, 20, 30]`` raised at the duplicate and kept 10 and 20.

    Retrying either of them then failed as older than a frontier the caller never
    asked for, which is how the partial commit was found.
    """
    vio: _core.Vio = pipeline(2)
    with pytest.raises(ValueError, match="does not follow"):
        vio.push_imu_batch(np.array([10, 20, 20, 30], dtype=np.int64), zeros(4), zeros(4))
    vio.push_imu(10, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
    vio.push_imu(20, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(start=starts, steps=gaps, index=positions, component=st.sampled_from([np.nan, np.inf, -np.inf]))
def test_a_batch_carrying_a_value_that_is_not_a_number_keeps_none_of_it(
    pipeline: PipelineFactory, start: int, steps: list[int], index: int, component: float
) -> None:
    """One non-finite component refuses the batch, and refuses it whole.

    A NaN reaches both preintegrators and poisons every state after it with
    nothing to undo it, so it is bad input rather than a value the estimator is
    asked to survive (D32) — and being bad input, it has to be found before the
    batch is pushed, or it takes the samples before it down with it.
    """
    times: NDArray[np.int64] = increasing(start, steps)
    position: int = index % len(times)
    gyro: NDArray[np.float64] = zeros(len(times))
    gyro[position, index % 3] = component
    vio: _core.Vio = pipeline(2)
    with pytest.raises(ValueError, match="non-finite gyro"):
        vio.push_imu_batch(times, gyro, zeros(len(times)))
    vio.push_imu_batch(times, zeros(len(times)), zeros(len(times)))


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(times=st.integers(min_value=1, max_value=8), rows=st.integers(min_value=1, max_value=8))
def test_mismatched_batch_lengths_are_rejected(pipeline: PipelineFactory, times: int, rows: int) -> None:
    assume(times != rows)
    vio: _core.Vio = pipeline(2)
    with pytest.raises(ValueError, match="same length"):
        vio.push_imu_batch(np.arange(times, dtype=np.int64), zeros(rows), zeros(rows))


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(dtype=st.sampled_from([np.int32, np.float64, np.uint64]), count=st.integers(min_value=1, max_value=8))
def test_batch_timestamps_of_the_wrong_dtype_are_rejected(pipeline: PipelineFactory, dtype: type, count: int) -> None:
    vio: _core.Vio = pipeline(2)
    wrong: NDArray[np.int64] = cast("NDArray[np.int64]", np.arange(count, dtype=dtype))
    with pytest.raises(ValueError, match="1-D int64"):
        vio.push_imu_batch(wrong, zeros(count), zeros(count))


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(count=st.integers(min_value=2, max_value=4), t_ns=starts)
def test_a_frameset_of_the_calibrated_size_is_accepted_at_any_timestamp(
    pipeline: PipelineFactory, texture: TextureFactory, count: int, t_ns: int
) -> None:
    """Any ``int64`` is a frameset timestamp, negative ones included.

    Without inertial samples covering it the frameset cannot measure, which is
    what :attr:`slam_rs._core.VioStatus.NeedMoreImu` says; the point here is that
    it is an answer rather than a refusal.
    """
    vio: _core.Vio = pipeline(count)
    result: _core.VioResult = vio.track(t_ns, [texture(0, 0)] * count)
    assert result.status == _core.VioStatus.NeedMoreImu
    assert result.t_ns == t_ns


@settings(max_examples=3, deadline=None)
@given(shifts=st.lists(st.integers(min_value=0, max_value=4), min_size=6, max_size=6))
def test_two_runs_over_the_same_input_agree_exactly(pipeline: PipelineFactory, texture: TextureFactory, shifts: list[int]) -> None:
    """Offline mode has no queues and no threads, so a repeat run is bit-identical (D17).

    Two estimators are driven over one sequence of framesets and inertial
    batches, and every pose is compared exactly rather than to a tolerance: this
    is the property that makes a reference run reproducible, and a scheduling
    dependency would break it by an ulp long before it broke an ATE gate.

    Catalog replay accuracy is covered by the slow ground-truth gate.
    """
    frames: list[list[UInt8[ndarray, "h w"]]] = [[texture(shift, 0), texture(shift + 1, 0)] for shift in shifts]
    runs: list[list[Float64[ndarray, " 7"]]] = []
    for _ in range(2):
        vio: _core.Vio = pipeline(2)
        poses: list[Float64[ndarray, " 7"]] = []
        for index, images in enumerate(frames):
            t_ns: int = index * FRAME_PERIOD_NS
            samples: Int64[ndarray, " n_samples"] = np.arange(t_ns, t_ns + FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
            vio.push_imu_batch(samples, *gravity_batch(samples))
            poses.append(vio.track(t_ns, images).world_from_rig)
        runs.append(poses)
    assert len(runs[0]) == len(frames)
    for index, (first, second) in enumerate(zip(runs[0], runs[1], strict=True)):
        assert np.array_equal(first, second), f"frameset {index} differed between two runs of the same input"
