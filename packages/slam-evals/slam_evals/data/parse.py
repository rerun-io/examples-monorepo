"""Parsers for VSLAM-LAB sequence CSV + calibration files.

Strategy: pyserde dataclass schemas describe the per-row CSV / per-camera
YAML shapes (column → field via ``field(rename=...)``, with a custom
``_ns`` deserializer for nanosecond timestamps that are sometimes written
as float literals — DRUNKARDS emits ``10000000000.0`` rather than
``10000000000``). For each input file we ``csv.DictReader`` (or
``serde.yaml.from_yaml``) into typed row instances, then stack the rows
into the numpy aggregate types (`RgbCsv`, `GroundTruth`, `ImuSamples`,
`Calibration`) consumed by the ingester.

All timestamp columns end up as ``int64`` nanoseconds; vector / scalar
data as ``float64`` so downstream consumers can mix them with Rerun's
``Scalars`` and ``Transform3D`` archetypes without per-call casts.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import SerdeError, coerce, field, from_dict, serde
from serde.yaml import from_yaml

from slam_evals.data.types import Calibration, CameraIntrinsics

# ─── helpers ─────────────────────────────────────────────────────────────────


def _ns(value: str | int | float) -> int:
    """Parse a nanosecond timestamp written as int OR float (e.g. ``10000000000.0``)."""
    return int(float(value))


# ─── per-row CSV schemas (pyserde-deserialized) ──────────────────────────────


@serde(type_check=coerce)
@dataclass
class _RgbRow:
    """One row from ``rgb.csv``. Optional columns default to ``None`` so the
    same schema covers mono, stereo, rgbd, and stereo+rgbd layouts."""

    ts_rgb_0_ns: int = field(rename="ts_rgb_0 (ns)", deserializer=_ns)
    path_rgb_0: str = field(rename="path_rgb_0")
    ts_rgb_1_ns: int | None = field(default=None, rename="ts_rgb_1 (ns)", deserializer=_ns)
    path_rgb_1: str | None = field(default=None, rename="path_rgb_1")
    ts_depth_0_ns: int | None = field(default=None, rename="ts_depth_0 (ns)", deserializer=_ns)
    path_depth_0: str | None = field(default=None, rename="path_depth_0")
    ts_depth_1_ns: int | None = field(default=None, rename="ts_depth_1 (ns)", deserializer=_ns)
    path_depth_1: str | None = field(default=None, rename="path_depth_1")


@serde(type_check=coerce)
@dataclass
class _GtRow:
    """One row from ``groundtruth.csv``: ``world_T_body`` translation + quaternion."""

    ts_ns: int = field(rename="ts (ns)", deserializer=_ns)
    tx: float = field(rename="tx (m)")
    ty: float = field(rename="ty (m)")
    tz: float = field(rename="tz (m)")
    qx: float
    qy: float
    qz: float
    qw: float


@serde(type_check=coerce)
@dataclass
class _ImuRow:
    """One row from ``imu_0.csv``: gyro (rad/s) + accel (m/s²)."""

    ts_ns: int = field(rename="ts (ns)", deserializer=_ns)
    wx: float = field(rename="wx (rad s^-1)")
    wy: float = field(rename="wy (rad s^-1)")
    wz: float = field(rename="wz (rad s^-1)")
    ax: float = field(rename="ax (m s^-2)")
    ay: float = field(rename="ay (m s^-2)")
    az: float = field(rename="az (m s^-2)")


# ─── numpy aggregate types consumed by the ingester ──────────────────────────


@dataclass(frozen=True, slots=True)
class RgbCsv:
    """Parsed ``rgb.csv``. Optional streams come back as ``None`` when their
    columns weren't in the source file.

    ``depth_1`` only appears in stereo-rgbd layouts like OPENLORIS-D400 — the
    depth stream pre-registered to ``rgb_1``."""

    ts_rgb_0_ns: Int64[ndarray, "n"]
    path_rgb_0: tuple[str, ...]
    ts_rgb_1_ns: Int64[ndarray, "n"] | None
    path_rgb_1: tuple[str, ...] | None
    ts_depth_0_ns: Int64[ndarray, "n"] | None
    path_depth_0: tuple[str, ...] | None
    ts_depth_1_ns: Int64[ndarray, "n"] | None
    path_depth_1: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class GroundTruth:
    ts_ns: Int64[ndarray, "n"]
    translation: Float64[ndarray, "n 3"]
    quaternion_xyzw: Float64[ndarray, "n 4"]


@dataclass(frozen=True, slots=True)
class ImuSamples:
    ts_ns: Int64[ndarray, "n"]
    gyro_rads: Float64[ndarray, "n 3"]
    accel_ms2: Float64[ndarray, "n 3"]


# ─── CSV → row dicts ─────────────────────────────────────────────────────────


def _read_dict_rows(path: Path) -> list[dict[str, str]]:
    """``csv.DictReader`` rows with header keys stripped of leading/trailing whitespace."""
    with path.open() as fh:
        reader = csv.DictReader(fh)
        return [{k.strip(): v for k, v in row.items() if k is not None} for row in reader]


# ─── public parsers ──────────────────────────────────────────────────────────


def parse_rgb_csv(path: Path) -> RgbCsv:
    rows: list[_RgbRow] = [from_dict(_RgbRow, r) for r in _read_dict_rows(path)]
    if not rows:
        raise ValueError(f"empty CSV: {path}")

    n = len(rows)

    def _opt_int(values: list[int | None]) -> Int64[ndarray, "n"] | None:
        if any(v is None for v in values):
            return None
        return np.fromiter(values, dtype=np.int64, count=n)

    def _opt_str(values: list[str | None]) -> tuple[str, ...] | None:
        # The any(v is None) guard narrows to all-strs, but pyrefly can't see
        # that through a generator expression. The list comprehension below
        # both narrows and copies — cheap at row counts in the thousands.
        if any(v is None for v in values):
            return None
        return tuple(v for v in values if v is not None)

    return RgbCsv(
        ts_rgb_0_ns=np.fromiter((r.ts_rgb_0_ns for r in rows), dtype=np.int64, count=n),
        path_rgb_0=tuple(r.path_rgb_0 for r in rows),
        ts_rgb_1_ns=_opt_int([r.ts_rgb_1_ns for r in rows]),
        path_rgb_1=_opt_str([r.path_rgb_1 for r in rows]),
        ts_depth_0_ns=_opt_int([r.ts_depth_0_ns for r in rows]),
        path_depth_0=_opt_str([r.path_depth_0 for r in rows]),
        ts_depth_1_ns=_opt_int([r.ts_depth_1_ns for r in rows]),
        path_depth_1=_opt_str([r.path_depth_1 for r in rows]),
    )


def parse_groundtruth(path: Path) -> GroundTruth:
    """Parse ``groundtruth.csv``. Empty body (header-only) returns zero-length
    correctly-shaped arrays — HAMLYN and a few HILTI sequences ship that way."""
    rows: list[_GtRow] = [from_dict(_GtRow, r) for r in _read_dict_rows(path)]
    if not rows:
        return GroundTruth(
            ts_ns=np.zeros((0,), dtype=np.int64),
            translation=np.zeros((0, 3), dtype=np.float64),
            quaternion_xyzw=np.zeros((0, 4), dtype=np.float64),
        )
    n = len(rows)
    return GroundTruth(
        ts_ns=np.fromiter((r.ts_ns for r in rows), dtype=np.int64, count=n),
        translation=np.asarray([[r.tx, r.ty, r.tz] for r in rows], dtype=np.float64),
        quaternion_xyzw=np.asarray([[r.qx, r.qy, r.qz, r.qw] for r in rows], dtype=np.float64),
    )


def parse_imu(path: Path) -> ImuSamples:
    rows: list[_ImuRow] = [from_dict(_ImuRow, r) for r in _read_dict_rows(path)]
    n = len(rows)
    return ImuSamples(
        ts_ns=np.fromiter((r.ts_ns for r in rows), dtype=np.int64, count=n),
        gyro_rads=np.asarray([[r.wx, r.wy, r.wz] for r in rows], dtype=np.float64),
        accel_ms2=np.asarray([[r.ax, r.ay, r.az] for r in rows], dtype=np.float64),
    )


# ─── calibration.yaml ────────────────────────────────────────────────────────


_IDENTITY_T_BS: tuple[float, ...] = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


@serde(type_check=coerce)
@dataclass
class _CameraYaml:
    """One entry from ``cameras:`` in calibration.yaml. Fields beyond what
    we model (``cam_model``, ``cam_type``, ``depth_name``, ``pixel_format``,
    …) are silently ignored — pyserde defaults to allowing unknown fields."""

    cam_name: str
    cam_type: str | None = None
    distortion_type: str | None = None
    focal_length: list[float] | None = None
    principal_point: list[float] | None = None
    distortion_coefficients: list[float] | None = None
    image_dimension: list[int] | None = None
    fps: float | None = None
    T_BS: list[float] | None = None
    depth_factor: float | None = None


@serde(type_check=coerce)
@dataclass
class _ImuYaml:
    """One entry from ``imus:``. ``imu_name`` and per-IMU noise terms
    (a_max, sigma_g_c, etc.) flow through unmodelled here — only ``fps``
    and ``T_BS`` are read directly. The noise terms re-surface on the
    ``imu_0`` layer's property bag when present."""

    fps: float | None = None
    T_BS: list[float] | None = None


@serde(type_check=coerce)
@dataclass
class _CalibrationYaml:
    cameras: list[_CameraYaml] = field(default_factory=list)
    imus: list[_ImuYaml] = field(default_factory=list)


def _to_camera_intrinsics(c: _CameraYaml) -> CameraIntrinsics:
    focal = c.focal_length or [0.0, 0.0]
    principal = c.principal_point or [0.0, 0.0]
    image_dim = c.image_dimension or [0, 0]
    distortion = tuple(float(x) for x in (c.distortion_coefficients or ()))
    t_bs = tuple(float(x) for x in c.T_BS) if c.T_BS else _IDENTITY_T_BS
    return CameraIntrinsics(
        cam_name=c.cam_name,
        width=int(image_dim[0]),
        height=int(image_dim[1]),
        fx=float(focal[0]),
        fy=float(focal[1]),
        cx=float(principal[0]),
        cy=float(principal[1]),
        distortion_type=c.distortion_type,
        distortion_coefficients=distortion,
        fps=c.fps,
        t_bs=t_bs,
        depth_factor=c.depth_factor,
    )


def parse_calibration(path: Path) -> Calibration | None:
    """Parse ``calibration.yaml``; return ``None`` on missing or malformed files."""
    if not path.exists():
        return None
    try:
        doc: _CalibrationYaml = from_yaml(_CalibrationYaml, path.read_text())
    except (SerdeError, yaml.YAMLError):
        return None

    cams = tuple(_to_camera_intrinsics(c) for c in doc.cameras)
    # Preserve the declared IMU's modelled fields (``fps``, ``T_BS``) as an
    # opaque dict for downstream ``has_imu_params`` detection and for the
    # ``imu_0`` layer's property bag. Other YAML fields (``imu_name``, noise
    # terms) are dropped by pyserde's default unknown-field handling — no
    # current consumer reads them.
    imu_params: dict[str, float | list[float]] | None = None
    if doc.imus:
        imu_params = {k: v for k, v in asdict(doc.imus[0]).items() if v is not None}

    return Calibration(cameras=cams, imu_params=imu_params)
