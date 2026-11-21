"""The logger's reported poses: growing arrays, so the whole estimate is a slice, never a rebuild from lists."""

import numpy as np
import pytest

from slam_rs.trajectory import PoseLog


def test_appended_poses_come_back_in_order_past_the_initial_capacity() -> None:
    log = PoseLog(capacity=4)
    poses = [(i * 10, np.array([i, 2.0 * i, 3.0 * i]), np.array([1.0, 0.0, 0.0, float(i)])) for i in range(11)]
    for t_ns, position, quaternion in poses:
        log.append(t_ns, position, quaternion)
    estimate = log.trajectory()
    assert len(log) == 11 and log.last_t_ns() == 100
    np.testing.assert_array_equal(estimate.t_ns, np.arange(11, dtype=np.int64) * 10)
    np.testing.assert_array_equal(estimate.position_m, np.stack([p for _, p, _ in poses]))
    np.testing.assert_array_equal(estimate.quaternion_wxyz, np.stack([q for _, _, q in poses]))


def test_a_trajectory_taken_earlier_is_not_changed_by_later_appends() -> None:
    log = PoseLog(capacity=2)
    log.append(0, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    early = log.trajectory()
    for t_ns in range(1, 9):
        log.append(t_ns, np.full(3, float(t_ns)), np.array([1.0, 0.0, 0.0, 0.0]))
    assert len(early) == 1 and early.t_ns.tolist() == [0] and early.position_m.tolist() == [[0.0, 0.0, 0.0]]


def test_an_empty_log_has_no_trajectory_and_no_last_time() -> None:
    log = PoseLog()
    assert len(log) == 0 and len(log.trajectory()) == 0 and log.last_t_ns() is None


def test_an_estimate_is_read_only_and_the_log_keeps_its_bytes_as_it_grows() -> None:
    log = PoseLog(capacity=2)
    log.append(0, np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    early = log.trajectory()
    with pytest.raises(ValueError, match="read-only"):
        early.position_m[0, 0] = 123.0
    with pytest.raises(ValueError, match="read-only"):
        np.subtract(early.position_m, early.position_m.mean(axis=0), out=early.position_m)
    for t_ns in range(1, 9):
        log.append(t_ns, np.full(3, float(t_ns)), np.array([1.0, 0.0, 0.0, 0.0]))
    assert log.trajectory().position_m[0].tolist() == [1.0, 2.0, 3.0]
