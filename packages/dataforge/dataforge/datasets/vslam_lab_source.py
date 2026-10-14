"""VSLAM-LAB's per-sequence layout, read through pyserde: the one door every VSLAM-LAB benchmark comes in by.

[VSLAM-LAB](https://github.com/VSLAM-LAB/VSLAM-LAB) converts some thirty SLAM benchmarks (TUM RGB-D, EuRoC,
7-Scenes, ETH3D, KITTI, ...) into one directory per sequence::

    <root>/<BENCHMARK>/<sequence>/
        calibration.yaml   cameras: pinhole + radtan distortion, T_BS; imus: noise terms, T_BS
        rgb.csv            ts_<stream> (ns), path_<stream> per image stream (rgb_0, rgb_1, depth_0, depth_1)
        groundtruth.csv    ts (ns), tx (m), ty (m), tz (m), qx, qy, qz, qw: world_T_body, scalar last
        imu_0.csv          ts (ns), wx..wz (rad s^-1), ax..az (m s^-2)
        rgb_0/ rgb_1/ depth_0/ depth_1/   the frames rgb.csv names

``T_BS`` is body_T_sensor as a row-major 4x4. Every file here is a third-party format read whole, so each row
or entry is a pyserde record: unknown YAML keys are allowed (their schema is not ours to police), and a file that
does not parse is a ``ValueError`` naming it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import SerdeError, coerce, field, from_dict, serde
from serde.yaml import from_yaml
from yaml import YAMLError

from dataforge.euroc import GtTrajectory, sanitized
from dataforge.logging_toolkit import ImuChannel

StreamName: TypeAlias = Literal["rgb_0", "rgb_1", "depth_0", "depth_1"]
"""Image streams rgb.csv can list; ``depth_<i>`` is registered to ``rgb_<i>``."""


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


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class FrameRow:
    """One rgb.csv row; a stream's two columns are absent together when the benchmark lacks it."""

    ts_rgb_0: int = field(rename="ts_rgb_0 (ns)", deserializer=nanoseconds)
    path_rgb_0: str = field(rename="path_rgb_0")
    ts_rgb_1: int | None = field(default=None, rename="ts_rgb_1 (ns)", deserializer=nanoseconds)
    path_rgb_1: str | None = field(default=None, rename="path_rgb_1")
    ts_depth_0: int | None = field(default=None, rename="ts_depth_0 (ns)", deserializer=nanoseconds)
    path_depth_0: str | None = field(default=None, rename="path_depth_0")
    ts_depth_1: int | None = field(default=None, rename="ts_depth_1 (ns)", deserializer=nanoseconds)
    path_depth_1: str | None = field(default=None, rename="path_depth_1")


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class PoseRow:
    """One groundtruth.csv row: world_T_body."""

    ts: int = field(rename="ts (ns)", deserializer=nanoseconds)
    tx: float = field(rename="tx (m)")
    ty: float = field(rename="ty (m)")
    tz: float = field(rename="tz (m)")
    qx: float
    qy: float
    qz: float
    qw: float


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class ImuRow:
    """One imu_0.csv row: angular velocity, then linear acceleration."""

    ts: int = field(rename="ts (ns)", deserializer=nanoseconds)
    wx: float = field(rename="wx (rad s^-1)")
    wy: float = field(rename="wy (rad s^-1)")
    wz: float = field(rename="wz (rad s^-1)")
    ax: float = field(rename="ax (m s^-2)")
    ay: float = field(rename="ay (m s^-2)")
    az: float = field(rename="az (m s^-2)")


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


RowT = TypeVar("RowT")


def read_rows(path: Path, row_type: type[RowT]) -> list[RowT]:  # noqa: UP047 — pyserde/beartype use the runtime TypeVar.
    """Every data row of a VSLAM-LAB csv as ``row_type``; a malformed file is a ``ValueError`` naming it."""
    with path.open(newline="") as stream:
        reader: csv.DictReader[str] = csv.DictReader(stream)
        try:
            return [from_dict(row_type, {name.strip(): value for name, value in row.items()}) for row in reader]
        except (SerdeError, ValueError, ArithmeticError) as error:
            raise ValueError(f"{path}: {error}") from error


def read_calibration(path: Path) -> Calibration:
    """Read a calibration.yaml; a malformed one is a ``ValueError`` naming it."""
    try:
        return from_yaml(Calibration, path.read_text())
    except (SerdeError, YAMLError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def read_frame_index(path: Path) -> dict[StreamName, FrameIndex]:
    """Read rgb.csv into one ``FrameIndex`` per stream it lists."""
    rows: list[FrameRow] = read_rows(path, FrameRow)
    columns: dict[StreamName, tuple[list[int | None], list[str | None]]] = {
        "rgb_0": ([row.ts_rgb_0 for row in rows], [row.path_rgb_0 for row in rows]),
        "rgb_1": ([row.ts_rgb_1 for row in rows], [row.path_rgb_1 for row in rows]),
        "depth_0": ([row.ts_depth_0 for row in rows], [row.path_depth_0 for row in rows]),
        "depth_1": ([row.ts_depth_1 for row in rows], [row.path_depth_1 for row in rows]),
    }
    streams: dict[StreamName, FrameIndex] = {}
    for name, (stamps, files) in columns.items():
        if rows and all(stamp is not None for stamp in stamps) and all(file is not None for file in files):
            streams[name] = FrameIndex(times_ns=np.array(stamps, dtype=np.int64), paths=tuple(str(file) for file in files))
    if "rgb_0" not in streams:
        raise ValueError(f"{path}: no rgb_0 frames")
    return streams


def read_poses(path: Path) -> GtTrajectory:
    """Read groundtruth.csv as world_T_body, a dropout row repaired as in every EuRoC-style file (``euroc.sanitized``)."""
    rows: list[PoseRow] = read_rows(path, PoseRow)
    return sanitized(
        np.array([row.ts for row in rows], dtype=np.int64),
        np.array([[row.tx, row.ty, row.tz] for row in rows], dtype=np.float64).reshape(-1, 3),
        np.array([[row.qx, row.qy, row.qz, row.qw] for row in rows], dtype=np.float64).reshape(-1, 4),
    )


def read_imu(path: Path) -> ImuStream:
    """Read imu_0.csv."""
    rows: list[ImuRow] = read_rows(path, ImuRow)
    times_ns: Int64[ndarray, "n_samples"] = np.array([row.ts for row in rows], dtype=np.int64)
    return ImuStream(
        gyro=ImuChannel(times_ns=times_ns, values_xyz=np.array([[row.wx, row.wy, row.wz] for row in rows], dtype=np.float64).reshape(-1, 3)),
        accel=ImuChannel(times_ns=times_ns, values_xyz=np.array([[row.ax, row.ay, row.az] for row in rows], dtype=np.float64).reshape(-1, 3)),
    )
