"""The EuRoC-style csv streams: the clock, the value columns, and the quaternion order.

The literals below are the header rows MSD actually ships — including the gt
file's spaces after each comma and its scalar-first quaternion — so what is
asserted is the format rather than a fixture's convenience. What a parsed
trajectory *means* is MSD's business: ``test_msd`` owns the gravity check.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataforge import euroc
from dataforge.euroc import (
    CameraRow,
    GtTrajectory,
    TimestampedSamples,
    gt_trajectory,
    read_camera_index,
    read_numeric_csv,
)

CAMERA_CSV: bytes = b"#timestamp [ns],filename\n13000000000000,13000000000000.png\n13000018518000,13000018518000.png\n"
IMU_CSV: bytes = (
    b"#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    b"a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
    b"12999000000000,0.01,-0.02,0.03,0.1,-0.2,9.8\n"
    b"12999001000000,0.011,-0.021,0.031,0.11,-0.21,9.81\n"
)
def test_camera_index_rows_carry_a_typed_stamp_and_filename() -> None:
    rows: list[CameraRow] = read_camera_index(CAMERA_CSV)
    assert [row.timestamp_ns for row in rows] == [13000000000000, 13000018518000]
    assert rows[1].filename == "13000018518000.png"


def test_numeric_csv_splits_the_clock_from_the_values() -> None:
    samples: TimestampedSamples = read_numeric_csv(IMU_CSV, num_values=6)
    assert samples.times_ns.dtype == np.int64
    np.testing.assert_array_equal(samples.times_ns, [12999000000000, 12999001000000])
    np.testing.assert_allclose(samples.values[0], [0.01, -0.02, 0.03, 0.1, -0.2, 9.8])
    assert samples.values.shape == (2, 6)


def test_a_single_row_numeric_csv_still_reads_as_a_table() -> None:
    one_row: bytes = b"\n".join(IMU_CSV.splitlines()[:2]) + b"\n"
    samples: TimestampedSamples = read_numeric_csv(one_row, num_values=6)
    assert samples.times_ns.shape == (1,)
    assert samples.values.shape == (1, 6)


GT_WXYZ_CSV: bytes = (
    b"#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n"
    b"12998000000000,0.0,0.1,0.2,0.5,0.5,-0.5,0.5\n"
    b"12998001000000,0.01,0.11,0.21,0.0,0.0,0.0,0.0\n"
)
"""A gt csv with a real rotation and one degenerate row, as a tracking dropout is written."""


def test_gt_quaternions_are_reordered_to_xyzw_and_dropouts_become_identity() -> None:
    trajectory: GtTrajectory = gt_trajectory(read_numeric_csv(GT_WXYZ_CSV, num_values=euroc.GT_VALUE_COLUMNS))
    np.testing.assert_array_equal(trajectory.times_ns, [12998000000000, 12998001000000])
    np.testing.assert_allclose(trajectory.translations_xyz[1], [0.01, 0.11, 0.21])
    # The file writes the scalar first; rr.Transform3D wants it last.
    np.testing.assert_allclose(trajectory.quaternions_xyzw[0], [0.5, -0.5, 0.5, 0.5])
    # A zero quaternion would break the rotation chain for every later frame.
    np.testing.assert_allclose(trajectory.quaternions_xyzw[1], [0.0, 0.0, 0.0, 1.0])
    assert trajectory.num_sanitized == 1


def test_an_empty_gt_is_an_error_not_a_silent_zero() -> None:
    # gt's first stamp is the whole sequence's clock origin; there is no default for it.
    header_only: bytes = GT_WXYZ_CSV.splitlines()[0] + b"\n"
    with pytest.raises(ValueError, match="no data rows"):
        gt_trajectory(read_numeric_csv(header_only, num_values=euroc.GT_VALUE_COLUMNS))
