"""The EuRoC-style csv streams a Monado SLAM archive ships, and the gravity check on them.

Every stream inside a sequence is one csv whose first column is nanoseconds on a
monotonic device clock: ``cam<N>/data.csv`` pairs a stamp with a frame filename,
``imu0``/``mag0``/``gt`` are otherwise all numeric. The layout is EuRoC's (the
quaternion is written **wxyz**, scalar first) and nothing here is specific to one
headset, so a second dataset in the same format reuses it as it stands.

The one piece that is not parsing is ``measured_world_up``: a tracking world with
no documented axes gives itself away through gravity, and gravity is only legible
once the ground truth and the accelerometer are both parsed and on one clock —
which is exactly here.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import serde
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge.logging_toolkit import ImuChannel

CAMERA_TIMESTAMP_FIELD: str = "#timestamp [ns]"
"""Header cell every one of these csvs starts with; the gt file pads the later cells with spaces."""
IMU_VALUE_COLUMNS: int = 6
"""``imu0/data.csv`` value columns: gyro xyz (rad/s) then accel xyz (m/s^2)."""
MAG_VALUE_COLUMNS: int = 3
"""``mag0/data.csv`` value columns: the field xyz, in the sensor's own units."""
GT_VALUE_COLUMNS: int = 7
"""``gt/data.csv`` value columns: position xyz (m) then the quaternion **wxyz**, scalar first."""
IDENTITY_QUATERNION_XYZW: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
"""What a dropout row's rotation becomes, so its translation still applies."""
QUATERNION_NORM_TOLERANCE: float = 1e-3
"""How far a gt quaternion's norm may sit from 1 before the row counts as a dropout."""
WorldUpAxis: TypeAlias = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
"""Signed axis of a tracking world that gravity points *away* from."""
POSITIVE_WORLD_AXES: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = ("+x", "+y", "+z")
"""Axis names by column index, for a positive mean; the negative row is below."""
NEGATIVE_WORLD_AXES: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = ("-x", "-y", "-z")
"""Axis names by column index, for a negative mean."""
STANDARD_GRAVITY_MS2: float = 9.80665
"""Standard gravity; ``measured_world_up`` reports its result as a fraction of this."""
MEASURED_UP_WINDOW_NS: int = 2_000_000_000
"""How much of a sequence's start ``measured_world_up`` averages over."""


@serde.serde(type_check=serde.coerce)
@dataclass(frozen=True, slots=True)
class CameraRow:
    """One row of a ``cam<N>/data.csv``: when a frame was captured, and where it is."""

    timestamp_ns: int = serde.field(rename=CAMERA_TIMESTAMP_FIELD)
    """Capture time on the device clock, nanoseconds."""
    filename: str = serde.field(rename="filename")
    """PNG file name inside the camera's ``data/`` directory."""


@dataclass(frozen=True, slots=True)
class TimestampedSamples:
    """One purely numeric MSD csv, split into its clock and its value columns."""

    times_ns: Int64[ndarray, "n_samples"]
    """Sample times on the device clock, nanoseconds."""
    values: Float64[ndarray, "n_samples n_values"]
    """Every column after the timestamp, in file order."""


@dataclass(frozen=True, slots=True)
class GtTrajectory:
    """One ``gt/data.csv`` as Rerun wants it: ``world_T_rig`` per sample, quaternion xyzw."""

    times_ns: Int64[ndarray, "n_poses"]
    """Pose times on the device clock, nanoseconds — already the IMU's clock."""
    translations_xyz: Float64[ndarray, "n_poses 3"]
    """``world_t_rig``, metres."""
    quaternions_xyzw: Float64[ndarray, "n_poses 4"]
    """``world_R_rig``, scalar **last**; a dropout row holds identity."""
    num_sanitized: int
    """Rows whose quaternion was not unit-norm and was replaced with identity."""


def gt_trajectory(samples: TimestampedSamples) -> GtTrajectory:
    """Split a parsed ``gt/data.csv`` into the columns ``rr.Transform3D`` takes.

    Two file-format facts live here and nowhere else. The csv writes the
    quaternion **wxyz** (EuRoC's scalar-first layout) while ``Transform3D``
    wants xyzw; and a tracking dropout is written as a degenerate quaternion.
    A non-unit quaternion breaks the rotation chain from that row on, so every
    child frustum of the rig stops tracking — slam-evals hit exactly this on
    ROVER-T265's ``0 0 0 0`` rows (``slam_evals/ingest/columns.py``) and settled
    on the same per-row repair: identity rotation, translation untouched, and
    the count reported rather than hidden.

    Args:
        samples: ``read_numeric_csv(..., num_values=GT_VALUE_COLUMNS)`` output.

    Returns:
        The pose columns, plus how many rows had to be repaired.

    Raises:
        ValueError: The file holds a header and nothing else. gt's first stamp is
            also the sequence's clock origin, so an empty one has no silent default.
    """
    if samples.times_ns.size == 0:
        raise ValueError("gt/data.csv has no data rows, so the sequence has neither poses nor a clock origin")
    translations_xyz: Float64[ndarray, "n_poses 3"] = np.ascontiguousarray(samples.values[:, 0:3])
    quaternions_xyzw: Float64[ndarray, "n_poses 4"] = np.ascontiguousarray(samples.values[:, [4, 5, 6, 3]])
    norms: Float64[ndarray, "n_poses"] = np.linalg.norm(quaternions_xyzw, axis=1)
    dropped: Bool[ndarray, "n_poses"] = np.abs(norms - 1.0) > QUATERNION_NORM_TOLERANCE
    quaternions_xyzw[dropped] = IDENTITY_QUATERNION_XYZW
    return GtTrajectory(
        times_ns=samples.times_ns,
        translations_xyz=translations_xyz,
        quaternions_xyzw=quaternions_xyzw,
        num_sanitized=int(dropped.sum()),
    )


@dataclass(frozen=True, slots=True)
class MeasuredUp:
    """What one sequence's own gravity measurement found; the gt layer records both."""

    axis: WorldUpAxis
    """The dominant signed world axis the mean acceleration points along."""
    fraction: float
    """That component as a fraction of standard gravity; near 1 is a clean measurement."""


def measured_world_up(gt: GtTrajectory, accel: ImuChannel, *, window_ns: int = MEASURED_UP_WINDOW_NS) -> MeasuredUp:
    """Measure which world axis is up, from gravity as the accelerometer sees it.

    An accelerometer at rest measures the *reaction* to gravity, so its reading
    points **up**; rotating each sample into the world with the ground truth's
    own orientation (``world_R_rig @ a_rig``) and averaging therefore yields a
    vector along the world's up axis. Only the first couple of seconds are used:
    a headset is typically still on the floor or on a head that has not started
    moving, so the mean is nearly pure gravity there and gets noisier the longer
    the window. Why this is measured at all, and what the three devices answer,
    is on ``MSD_DEVICES``.

    Args:
        gt: The sequence's ground truth, already in xyzw order and sanitized.
        accel: Accelerometer samples in m/s^2, on the same clock as ``gt``.
        window_ns: Length of the averaging window, from the first sample both
            streams cover.

    Returns:
        The axis and how much of gravity it carried — a health check, not a
        calibration: a much smaller fraction means the mean is not gravity.

    Raises:
        ValueError: Either stream is empty, or they do not overlap inside the window.
    """
    if gt.times_ns.size == 0 or accel.times_ns.size == 0:
        raise ValueError("measuring the world up axis needs both a gt pose and an accelerometer sample")
    start_ns: int = max(int(gt.times_ns[0]), int(accel.times_ns[0]))
    inside: Bool[ndarray, "n_samples"] = (accel.times_ns >= start_ns) & (accel.times_ns < start_ns + window_ns)
    if not inside.any():
        raise ValueError(f"no accelerometer sample within {window_ns / 1e9:g} s of {start_ns}, where the gt starts")

    window_times_ns: Int64[ndarray, "n_window"] = accel.times_ns[inside]
    after: Int64[ndarray, "n_window"] = np.clip(np.searchsorted(gt.times_ns, window_times_ns), 0, gt.times_ns.size - 1)
    before: Int64[ndarray, "n_window"] = np.clip(after - 1, 0, gt.times_ns.size - 1)
    nearest: Int64[ndarray, "n_window"] = np.where(
        np.abs(gt.times_ns[before] - window_times_ns) <= np.abs(gt.times_ns[after] - window_times_ns), before, after
    )
    world_accel_xyz: Float64[ndarray, "n_window 3"] = Rotation.from_quat(gt.quaternions_xyzw[nearest]).apply(accel.values_xyz[inside])
    mean_xyz: Float64[ndarray, "3"] = world_accel_xyz.mean(axis=0)

    axis_index: int = int(np.argmax(np.abs(mean_xyz)))
    names: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = POSITIVE_WORLD_AXES if mean_xyz[axis_index] >= 0.0 else NEGATIVE_WORLD_AXES
    return MeasuredUp(axis=names[axis_index], fraction=float(abs(mean_xyz[axis_index]) / STANDARD_GRAVITY_MS2))


def read_camera_index(data: bytes) -> list[CameraRow]:
    """Parse a ``cam<N>/data.csv`` into typed rows, in file order.

    Args:
        data: Whole csv, as read out of the archive.

    Returns:
        One ``CameraRow`` per data row; the file order *is* the presentation order.
    """
    reader: csv.DictReader[str] = csv.DictReader(io.StringIO(data.decode()))
    reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
    return [serde.from_dict(CameraRow, {name: row[name] for name in (CAMERA_TIMESTAMP_FIELD, "filename")}) for row in reader]


def read_numeric_csv(data: bytes, *, num_values: int) -> TimestampedSamples:
    """Parse one all-numeric MSD csv (``imu0``, ``mag0``, ``gt``).

    ``numpy.loadtxt`` rather than a per-row typed dataclass: these run at 1 kHz,
    so an hour-long sequence is 3.6M rows, where a pyserde row costs 9.8 s and
    2.1 GB of live objects against loadtxt's 0.8 s and one array. The timestamps
    are read in their own ``int64`` pass instead of being cast down from the
    float table, so no stamp can be rounded no matter how large the clock grows.

    Args:
        data: Whole csv, as read out of the archive.
        num_values: Columns to keep after the timestamp (6 for an IMU, 3 for a
            magnetometer); trailing columns are ignored.

    Returns:
        The clock and the value table, row-aligned.
    """
    times_ns: Int64[ndarray, "n_samples"] = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)
    values: Float64[ndarray, "n_samples n_values"] = np.loadtxt(
        io.BytesIO(data), delimiter=",", skiprows=1, usecols=range(1, 1 + num_values), dtype=np.float64, ndmin=2
    )
    return TimestampedSamples(times_ns=times_ns, values=values)


def nominal_fps(times_ns: Int64[ndarray, "n_samples"]) -> int:
    """Container frame rate for one camera, from the median inter-sample gap.

    Only the mp4 header cares: ``log_video_stream(times_ns=...)`` stamps every
    sample with its real capture time afterwards, so this just has to be close
    enough that a player without the rrd shows the clip at the right speed. The
    median (not the mean) keeps one dropped frame from halving it.

    Args:
        times_ns: Capture times of the camera's frames, ascending.

    Returns:
        Frames per second, at least 1.
    """
    if times_ns.size < 2:
        return 1
    median_gap_ns: float = float(np.median(np.diff(times_ns)))
    if median_gap_ns <= 0.0:
        raise ValueError(f"camera timestamps do not advance (median gap {median_gap_ns} ns); the csv is unusable")
    return max(1, round(1e9 / median_gap_ns))
