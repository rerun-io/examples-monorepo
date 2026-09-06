"""The EuRoC-style csv streams: the clock, the value columns, and the gravity check.

The literals below are the header rows MSD actually ships — including the gt
file's spaces after each comma and its scalar-first quaternion — so what is
asserted is the format rather than a fixture's convenience.
"""

from __future__ import annotations

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import euroc
from dataforge.euroc import (
    CameraRow,
    GtTrajectory,
    MeasuredUp,
    TimestampedSamples,
    gt_trajectory,
    measured_world_up,
    read_camera_index,
    read_numeric_csv,
)
from dataforge.logging_toolkit import ImuChannel

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


def constant_pose_gt(times_ns: Int64[ndarray, "n_poses"], quaternion_xyzw: Float64[ndarray, "4"]) -> TimestampedSamples:
    """A gt table holding one fixed orientation at the origin, in the file's wxyz order."""
    return TimestampedSamples(
        times_ns=times_ns,
        values=np.column_stack([np.zeros((times_ns.size, 3)), np.tile(quaternion_xyzw[[3, 0, 1, 2]], (times_ns.size, 1))]),
    )


def test_the_world_up_axis_is_measured_by_rotating_the_accelerometer_into_the_world() -> None:
    """An accelerometer at rest reads +g pointing *up*, so ``world_R_rig @ a_rig`` averages to the up axis."""
    # -90 deg about x maps the rig's +z onto the world's +y, so a headset held level
    # in a Y-up world reads gravity along its own +z.
    world_R_rig: Rotation = Rotation.from_euler("x", -90.0, degrees=True)
    times_ns: Int64[ndarray, "n_poses"] = np.arange(4_000, dtype=np.int64) * 1_000_000
    rig_accel_xyz: Float64[ndarray, "n_samples 3"] = np.tile([0.1, -0.2, 9.81], (times_ns.size, 1))
    # The second half of the capture points the other way; the 2 s window must ignore it.
    rig_accel_xyz[times_ns >= euroc.MEASURED_UP_WINDOW_NS] = [0.1, -0.2, -9.81]
    gt: GtTrajectory = gt_trajectory(constant_pose_gt(times_ns, np.asarray(world_R_rig.as_quat(), dtype=np.float64)))

    measured: MeasuredUp = measured_world_up(gt, ImuChannel(times_ns=times_ns, values_xyz=rig_accel_xyz))

    assert measured.axis == "+y"
    # At rest the whole of gravity lands on that one axis.
    assert measured.fraction == pytest.approx(1.0, abs=0.01)


def test_an_empty_gt_is_an_error_not_a_silent_zero() -> None:
    # gt's first stamp is the whole sequence's clock origin; there is no default for it.
    header_only: bytes = GT_WXYZ_CSV.splitlines()[0] + b"\n"
    with pytest.raises(ValueError, match="no data rows"):
        gt_trajectory(read_numeric_csv(header_only, num_values=euroc.GT_VALUE_COLUMNS))
