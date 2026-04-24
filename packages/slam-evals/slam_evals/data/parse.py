"""Parsers for VSLAM-LAB sequence CSV + calibration files.

All timestamp columns are returned as ``int64`` nanoseconds; vector/scalar data
as ``float64`` so downstream consumers can mix them with Rerun's ``Scalars``
and ``Transform3D`` archetypes without per-call casts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_evals.data.types import Calibration, CameraIntrinsics


@dataclass(frozen=True, slots=True)
class RgbCsv:
    """Parsed ``rgb.csv``. Columns that don't exist in the source come back as ``None``.

    ``depth_1`` only appears in stereo-rgbd layouts like OPENLORIS-D400. When present
    it's the depth stream pre-registered to ``rgb_1`` (same sensor).
    """

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


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open() as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[0], rows[1:]


def _parse_ns(value: str) -> int:
    """Parse a nanosecond timestamp that may be written as an int or float literal.

    Some datasets (e.g. DRUNKARDS) emit ``10000000000.0`` rather than
    ``10000000000``. Both round-trip to the same integer nanosecond value.
    """
    return int(float(value))


def parse_rgb_csv(path: Path) -> RgbCsv:
    header, rows = _read_csv(path)
    idx = {name.strip(): i for i, name in enumerate(header)}

    def col_int64(key: str) -> Int64[ndarray, "n"]:
        i = idx[key]
        return np.asarray([_parse_ns(r[i]) for r in rows], dtype=np.int64)

    def col_str(key: str) -> tuple[str, ...]:
        i = idx[key]
        return tuple(r[i] for r in rows)

    ts_rgb_0_ns: Int64[ndarray, "n"] = col_int64("ts_rgb_0 (ns)")
    path_rgb_0: tuple[str, ...] = col_str("path_rgb_0")

    ts_rgb_1_ns: Int64[ndarray, "n"] | None = col_int64("ts_rgb_1 (ns)") if "ts_rgb_1 (ns)" in idx else None
    path_rgb_1: tuple[str, ...] | None = col_str("path_rgb_1") if "path_rgb_1" in idx else None
    ts_depth_0_ns: Int64[ndarray, "n"] | None = col_int64("ts_depth_0 (ns)") if "ts_depth_0 (ns)" in idx else None
    path_depth_0: tuple[str, ...] | None = col_str("path_depth_0") if "path_depth_0" in idx else None
    ts_depth_1_ns: Int64[ndarray, "n"] | None = col_int64("ts_depth_1 (ns)") if "ts_depth_1 (ns)" in idx else None
    path_depth_1: tuple[str, ...] | None = col_str("path_depth_1") if "path_depth_1" in idx else None

    return RgbCsv(
        ts_rgb_0_ns=ts_rgb_0_ns,
        path_rgb_0=path_rgb_0,
        ts_rgb_1_ns=ts_rgb_1_ns,
        path_rgb_1=path_rgb_1,
        ts_depth_0_ns=ts_depth_0_ns,
        path_depth_0=path_depth_0,
        ts_depth_1_ns=ts_depth_1_ns,
        path_depth_1=path_depth_1,
    )


def parse_groundtruth(path: Path) -> GroundTruth:
    header, rows = _read_csv(path)
    idx = {name.strip(): i for i, name in enumerate(header)}

    # HAMLYN and a few HILTI sequences ship a groundtruth.csv with just the
    # header — return empty, correctly-shaped arrays rather than raising.
    if not rows:
        return GroundTruth(
            ts_ns=np.zeros((0,), dtype=np.int64),
            translation=np.zeros((0, 3), dtype=np.float64),
            quaternion_xyzw=np.zeros((0, 4), dtype=np.float64),
        )

    ts_ns: Int64[ndarray, "n"] = np.asarray([_parse_ns(r[idx["ts (ns)"]]) for r in rows], dtype=np.int64)
    translation: Float64[ndarray, "n 3"] = np.asarray(
        [[float(r[idx["tx (m)"]]), float(r[idx["ty (m)"]]), float(r[idx["tz (m)"]])] for r in rows],
        dtype=np.float64,
    )
    quaternion_xyzw: Float64[ndarray, "n 4"] = np.asarray(
        [[float(r[idx["qx"]]), float(r[idx["qy"]]), float(r[idx["qz"]]), float(r[idx["qw"]])] for r in rows],
        dtype=np.float64,
    )
    return GroundTruth(ts_ns=ts_ns, translation=translation, quaternion_xyzw=quaternion_xyzw)


def parse_imu(path: Path) -> ImuSamples:
    header, rows = _read_csv(path)
    idx = {name.strip(): i for i, name in enumerate(header)}

    ts_ns: Int64[ndarray, "n"] = np.asarray([_parse_ns(r[idx["ts (ns)"]]) for r in rows], dtype=np.int64)
    gyro: Float64[ndarray, "n 3"] = np.asarray(
        [[float(r[idx["wx (rad s^-1)"]]), float(r[idx["wy (rad s^-1)"]]), float(r[idx["wz (rad s^-1)"]])] for r in rows],
        dtype=np.float64,
    )
    accel: Float64[ndarray, "n 3"] = np.asarray(
        [[float(r[idx["ax (m s^-2)"]]), float(r[idx["ay (m s^-2)"]]), float(r[idx["az (m s^-2)"]])] for r in rows],
        dtype=np.float64,
    )
    return ImuSamples(ts_ns=ts_ns, gyro_rads=gyro, accel_ms2=accel)


_IDENTITY_T_BS: tuple[float, ...] = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _parse_camera(raw: dict) -> CameraIntrinsics:
    focal = raw.get("focal_length") or [0.0, 0.0]
    principal = raw.get("principal_point") or [0.0, 0.0]
    image_dim = raw.get("image_dimension") or [0, 0]
    distortion = tuple(float(x) for x in raw.get("distortion_coefficients") or ())
    t_bs_list = raw.get("T_BS")
    t_bs = tuple(float(x) for x in t_bs_list) if t_bs_list else _IDENTITY_T_BS
    return CameraIntrinsics(
        cam_name=str(raw["cam_name"]),
        width=int(image_dim[0]),
        height=int(image_dim[1]),
        fx=float(focal[0]),
        fy=float(focal[1]),
        cx=float(principal[0]),
        cy=float(principal[1]),
        distortion_type=str(raw["distortion_type"]) if raw.get("distortion_type") else None,
        distortion_coefficients=distortion,
        fps=float(raw["fps"]) if raw.get("fps") is not None else None,
        t_bs=t_bs,
        depth_factor=float(raw["depth_factor"]) if raw.get("depth_factor") is not None else None,
    )


def parse_calibration(path: Path) -> Calibration | None:
    """Parse ``calibration.yaml``; return ``None`` on missing / malformed files."""
    if not path.exists():
        return None
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return None
    if not isinstance(doc, dict):
        return None

    raw_cams = doc.get("cameras") or []
    cams = tuple(_parse_camera(cast(dict, c)) for c in raw_cams if isinstance(c, dict))

    imu_params: dict[str, float | list[float]] | None = None
    raw_imus = doc.get("imus") or []
    if raw_imus:
        first = cast(dict, raw_imus[0])
        imu_params = {k: v for k, v in first.items() if isinstance(v, int | float | list)}

    return Calibration(cameras=cams, imu_params=imu_params)
