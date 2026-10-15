"""VSLAM-LAB's per-sequence layout, read through pyserde: the one door every VSLAM-LAB benchmark comes in by.

[VSLAM-LAB](https://github.com/VSLAM-LAB/VSLAM-LAB) converts some thirty SLAM benchmarks (TUM RGB-D, EuRoC,
7-Scenes, ETH3D, KITTI, ...) into one directory per sequence::

    <root>/<BENCHMARK>/<sequence>/
        calibration.yaml   cameras: pinhole + radtan distortion, T_BS; imus: noise terms, T_BS
        rgb.csv            ts_<stream> (ns), path_<stream> per image stream (rgb_0, rgb_1, depth_0, depth_1)
        groundtruth.csv    ts (ns), tx (m), ty (m), tz (m), qx, qy, qz, qw: world_T_body, scalar last
        imu_0.csv          ts (ns), wx..wz (rad s^-1), ax..az (m s^-2)
        rgb_0/ rgb_1/ depth_0/ depth_1/   the frames rgb.csv names
        state_groundtruth_estimate0.csv   EuRoC only: EuRoC's own world_T_imu0, which dataforge's ``download``
                                          extracts from the sequence zip VSLAM-LAB keeps (``read_euroc_gt``)

``T_BS`` is body_T_sensor as a row-major 4x4. calibration.yaml is a record read whole, so each entry is a pyserde
record: unknown YAML keys are allowed (their schema is not ours to police). rgb.csv is read by its header, one
stamp and one path column per stream ``StreamName`` lists. groundtruth.csv and imu_0.csv are streams (EuRoC's run at 200 Hz), read as Arrow columns. Either way a file that
does not parse is a ``ValueError`` naming it. VSLAM-LAB's EuRoC groundtruth.csv is not a body pose (cam0 positions from
TUM's supp_v2 with a constant placeholder rotation), so EuRoC's ground truth comes in through its own file instead.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import SerdeError, coerce, field, serde
from serde.yaml import from_yaml
from yaml import YAMLError

from dataforge.euroc import GT_VALUE_COLUMNS, GtTrajectory, TimestampedSamples, gt_trajectory, read_numeric_csv, sanitized
from dataforge.logging_toolkit import ImuChannel

StreamName: TypeAlias = Literal["rgb_0", "rgb_1", "depth_0", "depth_1"]
"""Image streams rgb.csv can list; ``depth_<i>`` is registered to ``rgb_<i>``."""
TIMESTAMP_COLUMN: str = "ts (ns)"
"""First column of groundtruth.csv and imu_0.csv."""
POSE_COLUMNS: tuple[str, ...] = ("tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw")
"""groundtruth.csv after the stamp: world_t_body, then world_R_body scalar last."""
IMU_COLUMNS: tuple[str, ...] = ("wx (rad s^-1)", "wy (rad s^-1)", "wz (rad s^-1)", "ax (m s^-2)", "ay (m s^-2)", "az (m s^-2)")
"""imu_0.csv after the stamp: angular velocity, then linear acceleration."""
EUROC_GT_MEMBER: str = "mav0/state_groundtruth_estimate0/data.csv"
"""EuRoC's own ground truth inside each sequence zip: world_T_imu0 (Vicon / Leica MS50) at 200 Hz, quaternion w first."""
EUROC_GT_FILE: str = "state_groundtruth_estimate0.csv"
"""That member, verbatim, in the sequence directory, where ``download`` puts it."""


def nanoseconds(value: str) -> int:
    """A ``ts (ns)`` cell, exactly: most benchmarks write integers, some float literals (``10000000000.0``).

    ``Decimal`` because epoch stamps (~1.4e18) are beyond float64's exact integer range.
    """
    return int(Decimal(value))


def row_major_4x4(values: list[float]) -> Float64[ndarray, "4 4"]:
    """A ``T_BS`` list as the 4x4 it writes row-major; anything but 16 numbers is refused."""
    if len(values) != 16:
        raise ValueError(f"T_BS holds {len(values)} numbers, not a 4x4")
    return np.array(values, dtype=np.float64).reshape(4, 4)


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class CameraEntry:
    """One ``cameras:`` entry of calibration.yaml."""

    cam_name: StreamName
    """Stream it calibrates, ``rgb_<i>``."""
    cam_type: Literal["gray", "rgb", "rgb+depth"]
    """Pixel content; ``rgb+depth`` means ``depth_name`` is registered to this camera."""
    focal_length: tuple[float, float]
    """fx, fy in pixels."""
    principal_point: tuple[float, float]
    """cx, cy in pixels."""
    image_dimension: tuple[int, int]
    """Width, height in pixels."""
    fps: float
    """Nominal frame rate."""
    body_T_sensor: Float64[ndarray, "4 4"] = field(rename="T_BS", deserializer=row_major_4x4)
    """Pose in the body (rig) frame, VSLAM-LAB's ``T_BS``."""
    distortion_type: Literal["radtan4", "radtan5"] | None = None
    """Brown-Conrady with k1 k2 p1 p2 (radtan4) or k1 k2 p1 p2 k3 (radtan5); absent for an undistorted camera."""
    distortion_coefficients: tuple[float, ...] = ()
    """The coefficients ``distortion_type`` names, in that order."""
    depth_name: StreamName | None = None
    """Depth stream registered to this camera (``depth_<i>``), on an ``rgb+depth`` camera."""
    depth_factor: float | None = None
    """Native depth units per metre: metres = pixel / depth_factor."""

    def __post_init__(self) -> None:
        expected: int = {None: 0, "radtan4": 4, "radtan5": 5}[self.distortion_type]
        if len(self.distortion_coefficients) != expected:
            raise ValueError(f"{self.cam_name}: {self.distortion_type} takes {expected} coefficients, got {len(self.distortion_coefficients)}")
        if (self.depth_name is None) != (self.depth_factor is None):
            raise ValueError(f"{self.cam_name}: depth_name and depth_factor come together")


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class ImuEntry:
    """One ``imus:`` entry: the continuous-time noise model VSLAM-LAB hands its VIO baselines (OKVIS names)."""

    imu_name: str
    """Stream it describes, ``imu_<i>``."""
    fps: float
    """Nominal sample rate."""
    body_T_sensor: Float64[ndarray, "4 4"] = field(rename="T_BS", deserializer=row_major_4x4)
    """Pose in the body (rig) frame, VSLAM-LAB's ``T_BS``."""
    sigma_g_c: float | None = None
    """Gyro white-noise density, rad/s/sqrt(Hz)."""
    sigma_a_c: float | None = None
    """Accelerometer white-noise density, m/s^2/sqrt(Hz)."""
    sigma_gw_c: float | None = None
    """Gyro bias random walk, rad/s^2/sqrt(Hz)."""
    sigma_aw_c: float | None = None
    """Accelerometer bias random walk, m/s^3/sqrt(Hz)."""


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Calibration:
    """A whole calibration.yaml."""

    cameras: tuple[CameraEntry, ...]
    """Cameras in ``rgb_<i>`` order."""
    imus: tuple[ImuEntry, ...] = ()
    """IMUs in ``imu_<i>`` order; none on a camera-only benchmark."""


@dataclass(frozen=True, slots=True)
class FrameIndex:
    """One image stream of rgb.csv: when each frame was captured, and where it lies."""

    times_ns: Int64[ndarray, "n_frames"]
    """Capture stamps, nanoseconds, in file order."""
    paths: tuple[str, ...]
    """Frame files, relative to the sequence directory."""


@dataclass(frozen=True, slots=True)
class ImuStream:
    """imu_0.csv split into its two channels on one clock."""

    gyro: ImuChannel
    """Angular velocity, rad/s."""
    accel: ImuChannel
    """Linear acceleration, m/s^2."""


def read_calibration(path: Path) -> Calibration:
    """Read a calibration.yaml; a malformed one is a ``ValueError`` naming it."""
    try:
        return from_yaml(Calibration, path.read_text())
    except (SerdeError, YAMLError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def read_frame_index(path: Path) -> dict[StreamName, FrameIndex]:
    """Read rgb.csv into one ``FrameIndex`` per stream its header lists, by its ``ts_<stream> (ns)`` and ``path_<stream>``.

    Every row must fill both cells of every listed stream: a short row or an empty cell is a ``ValueError`` naming the
    file and the stream, never a stream silently dropped or a frame path that is the sequence directory.
    """
    with path.open(newline="") as stream:
        reader: csv.DictReader[str] = csv.DictReader(stream)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        rows: list[dict[str, str | None]] = list(reader)
    streams: dict[StreamName, FrameIndex] = {}
    for name in get_args(StreamName):
        stamp_column, path_column = f"ts_{name} (ns)", f"path_{name}"
        if stamp_column not in reader.fieldnames and path_column not in reader.fieldnames:
            continue
        cells: list[tuple[str | None, str | None]] = [(row.get(stamp_column), row.get(path_column)) for row in rows]
        try:
            if not all(stamp and file for stamp, file in cells):
                raise ValueError("a row lacks its stamp or its file")
            times_ns: Int64[ndarray, "n_frames"] = np.array([nanoseconds(str(stamp)) for stamp, _ in cells], dtype=np.int64)
        except (ValueError, ArithmeticError) as error:
            raise ValueError(f"{path}: {name}: {error}") from error
        streams[name] = FrameIndex(times_ns=times_ns, paths=tuple(str(file) for _, file in cells))
    if not rows or "rgb_0" not in streams:
        raise ValueError(f"{path}: no rgb_0 frames")
    return streams


def read_samples(path: Path, columns: tuple[str, ...]) -> TimestampedSamples:
    """A stream csv as its ``ts (ns)`` clock and a float64 table of ``columns``, in that order.

    The stamp is read as text and converted by ``nanoseconds``, so it stays exact; a missing column or an empty or
    non-numeric cell is a ``ValueError`` naming the file.
    """
    options: pacsv.ConvertOptions = pacsv.ConvertOptions(
        column_types={TIMESTAMP_COLUMN: pa.string(), **dict.fromkeys(columns, pa.float64())}, include_columns=[TIMESTAMP_COLUMN, *columns], null_values=[]
    )
    try:
        table: pa.Table = pacsv.read_csv(path, convert_options=options)
        times_ns: Int64[ndarray, "n_samples"] = np.array([nanoseconds(stamp) for stamp in table[TIMESTAMP_COLUMN].to_pylist()], dtype=np.int64)
    except (KeyError, ValueError, ArithmeticError) as error:
        raise ValueError(f"{path}: {error}") from error
    return TimestampedSamples(times_ns=times_ns, values=np.column_stack([table[name].to_numpy() for name in columns]))


def read_poses(path: Path) -> GtTrajectory:
    """Read groundtruth.csv as world_T_body, a dropout row repaired as in every EuRoC-style file (``euroc.sanitized``)."""
    samples: TimestampedSamples = read_samples(path, POSE_COLUMNS)
    return sanitized(samples.times_ns, samples.values[:, :3], samples.values[:, 3:])


def read_euroc_gt(path: Path) -> GtTrajectory:
    """EuRoC's own state_groundtruth_estimate0 as world_T_body, through the EuRoC reader MSD's gt also takes.

    VSLAM-LAB's EuRoC body frame is imu0 (its ``T_BS`` is the identity), the frame this file poses, so the rows need no
    transform; ``euroc.gt_trajectory`` turns the w-first quaternion around and repairs a dropout row.
    """
    try:
        return gt_trajectory(read_numeric_csv(path.read_bytes(), num_values=GT_VALUE_COLUMNS))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def read_imu(path: Path) -> ImuStream:
    """Read imu_0.csv."""
    samples: TimestampedSamples = read_samples(path, IMU_COLUMNS)
    return ImuStream(
        gyro=ImuChannel(times_ns=samples.times_ns, values_xyz=np.ascontiguousarray(samples.values[:, :3])),
        accel=ImuChannel(times_ns=samples.times_ns, values_xyz=np.ascontiguousarray(samples.values[:, 3:])),
    )
