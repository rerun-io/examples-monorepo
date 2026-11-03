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


def test_the_gap_boundary_is_inclusive() -> None:
    result = bracket(np.array([0.0, 0.04]), np.array([0.02]), 0.04)
    assert result.valid[0]


def test_zero_length_gaps_use_the_left_sample_without_dividing_by_zero() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        result = bracket(np.array([0.0, 0.0]), np.array([0.0, 0.01]), 0.04)
    np.testing.assert_array_equal(result.valid, [True, False])
    np.testing.assert_array_equal(result.alpha, [0.0, 0.0])
    np.testing.assert_array_equal(lerp(np.array([1.0, 99.0]), result), [1.0, 1.0])


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_samples_have_no_bracket(count: int) -> None:
    result = bracket(np.zeros(count), np.array([0.0, 0.01]), 0.1)
    assert not result.valid.any()


def test_lerp_preserves_the_value_axes() -> None:
    samples = bracket(np.array([0.0, 0.1]), np.array([0.025, 0.075]), 0.1)
    values = np.array([[[0.0, 2.0], [4.0, 6.0]], [[8.0, 10.0], [12.0, 14.0]]])
    np.testing.assert_allclose(lerp(values, samples), [[[2.0, 4.0], [6.0, 8.0]], [[6.0, 8.0], [10.0, 12.0]]])


def test_slerp_interpolates_rotation_without_extrapolating_track_ends() -> None:
    rotations = Rotation.from_rotvec([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]])
    result = slerp(np.array([0.0, 0.1]), rotations, np.array([-0.1, 0.05, 0.2]))
    expected = np.array([np.eye(3), [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]])
    assert np.isnan(result[[0, 2]]).all()
    np.testing.assert_allclose(result[1], expected[1], atol=1e-12)


def test_slerp_keeps_the_first_sample_at_a_repeated_stamp() -> None:
    rotations = Rotation.from_rotvec([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = slerp(np.array([0.0, 0.0]), rotations, np.array([0.0]))
    np.testing.assert_allclose(result, np.eye(3)[None], atol=1e-12)


def test_position_interpolation_uses_valid_source_neighbours_and_never_clamps() -> None:
    from dataforge.interpolation import interpolate_positions

    stamps = np.array([0.0, 0.04, 0.08])
    values = np.array([[0.0, 1.0, 2.0], [4.0, 1.0, 2.0], [8.0, 1.0, 2.0]])
    times = np.array([-0.01, 0.02, 0.06, 0.09])
    result = interpolate_positions(stamps, values, times, 0.04, valid=np.array([True, True, False]))
    np.testing.assert_array_equal(result[1], [2.0, 1.0, 2.0])
    assert np.isnan(result[[0, 2, 3]]).all()


def test_pose_interpolation_uses_one_policy_for_positions_rotations_and_validity() -> None:
    from dataforge.interpolation import interpolate_poses

    result = interpolate_poses(np.array([0.0, 0.04]), np.array([[0.0, 0.0, 0.0], [0.004, 0.0, 0.0]]),
                               Rotation.from_euler("z", [[0.0], [0.2]], degrees=True), np.array([-0.01, 0.02, 0.05]), 0.04)
    np.testing.assert_allclose(result.positions[1], [0.002, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(Rotation.from_matrix(result.rotations[1]).as_euler("xyz", degrees=True), [0.0, 0.0, 0.1], atol=1e-12)
    np.testing.assert_array_equal(result.valid, [False, True, False])
    assert np.isnan(result.transforms[[0, 2]]).all()
