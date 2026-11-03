"""Gap-gated interpolation on measured timestamps."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataforge.interpolation import bracket, lerp, slerp


@pytest.mark.parametrize(
    ("times", "gap", "expected_alpha", "expected_valid"),
    [
        ([0.025], 0.1, [0.25], [True]),
        ([0.025], 0.05, [0.25], [False]),
        ([0.0, 0.1], 0.1, [0.0, 1.0], [True, True]),
        ([-0.025, 0.125], 0.1, [-0.25, 1.25], [False, False]),
    ],
)
def test_bracket_keeps_exact_hits_and_rejects_large_gaps(times, gap, expected_alpha, expected_valid) -> None:
    result = bracket(np.array([0.0, 0.1]), np.array(times), gap)
    np.testing.assert_array_equal(result.left, np.zeros(len(times), dtype=np.int64))
    np.testing.assert_array_equal(result.right, np.ones(len(times), dtype=np.int64))
    np.testing.assert_allclose(result.alpha, expected_alpha, atol=1e-15)
    np.testing.assert_array_equal(result.valid, expected_valid)


@pytest.mark.parametrize("inclusive", [False, True])
def test_the_gap_boundary_policy_is_explicit(inclusive: bool) -> None:
    result = bracket(np.array([0.0, 0.04]), np.array([0.02]), 0.04, inclusive=inclusive)
    assert bool(result.valid[0]) == inclusive


def test_zero_length_gaps_are_invalid_and_do_not_divide_by_zero() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        result = bracket(np.array([0.0, 0.0]), np.array([0.0, 0.01]), 0.04)
    assert not result.valid.any()
    assert np.isfinite(result.alpha).all()


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_samples_have_no_bracket(count: int) -> None:
    result = bracket(np.zeros(count), np.array([0.0, 0.01]), 0.1)
    assert not result.valid.any()


def test_lerp_preserves_the_value_axes() -> None:
    samples = bracket(np.array([0.0, 0.1]), np.array([0.025, 0.075]), 0.1)
    values = np.array([[[0.0, 2.0], [4.0, 6.0]], [[8.0, 10.0], [12.0, 14.0]]])
    np.testing.assert_allclose(lerp(values, samples), [[[2.0, 4.0], [6.0, 8.0]], [[6.0, 8.0], [10.0, 12.0]]])


def test_slerp_interpolates_rotation_and_clamps_the_track_ends() -> None:
    rotations = Rotation.from_rotvec([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]])
    result = slerp(np.array([0.0, 0.1]), rotations, np.array([-0.1, 0.05, 0.2]))
    expected = np.array([np.eye(3), [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_slerp_keeps_the_first_sample_at_a_repeated_stamp() -> None:
    rotations = Rotation.from_rotvec([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = slerp(np.array([0.0, 0.0]), rotations, np.array([0.0]))
    np.testing.assert_allclose(result, np.eye(3)[None], atol=1e-12)
