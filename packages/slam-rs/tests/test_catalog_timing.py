"""Time interval selection preserves boundaries and independent sample storage."""

import numpy as np
import pytest

from slam_rs.catalog_timing import ImuStream


@pytest.mark.parametrize(
    ("first", "last", "expected"),
    [(10, 30, [20, 30]), (10, 10, []), (30, 10, []), (-10, 9, []), (31, 40, []), (-10, 40, [10, 20, 30])],
)
def test_imu_interval_is_open_left_closed_right_and_owned(first: int, last: int, expected: list[int]) -> None:
    stream: ImuStream = ImuStream(
        t_ns=np.array([10, 20, 30], dtype=np.int64),
        gyro_rad_s=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        accel_m_s2=np.zeros((3, 3)),
    )
    selected: ImuStream = stream.between(first, last)
    np.testing.assert_array_equal(selected.t_ns, expected)
    for stamp, gyro in zip(selected.t_ns, selected.gyro_rad_s, strict=True):
        np.testing.assert_array_equal(gyro, stream.gyro_rad_s[{10: 0, 20: 1, 30: 2}[int(stamp)]])
    selected.t_ns[:] = 0
    selected.gyro_rad_s[:] = 0.0
    selected.accel_m_s2[:] = 1.0
    np.testing.assert_array_equal(stream.t_ns, [10, 20, 30])
    assert stream.gyro_rad_s.sum() == 45.0
    assert stream.accel_m_s2.sum() == 0.0


def test_empty_imu_interval() -> None:
    stream: ImuStream = ImuStream(np.empty(0, dtype=np.int64), np.empty((0, 3)), np.empty((0, 3)))
    selected: ImuStream = stream.between(0, 10)
    assert len(selected) == 0
    assert selected.gyro_rad_s.shape == (0, 3)
