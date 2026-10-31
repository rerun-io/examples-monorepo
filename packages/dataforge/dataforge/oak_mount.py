"""The OAK rig hangs on the Quest's left-eye frame by one static mount: a nominal prior in the base layer, replaced by
the fitted mount in the calibration layer (same components, so the newer write wins). This module turns that into
OAK world poses for code that needs them per frame."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from datafusion import col, lit
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetEntry

MOUNT_SOURCE = "oak_mount_source"
"""AnyValues field on the OAK rig: ``nominal`` (base prior) or ``fitted`` (calibration)."""


def _first(table: pa.Table, name: str):  # noqa: ANN202
    if name not in table.column_names:
        return None
    for value in table[name].to_pylist():
        while isinstance(value, list) and len(value) == 1:
            value = value[0]
        if value is not None:
            return value
    return None


def quaternion_matrices(xyzw: Float64[ndarray, "n 4"]) -> Float64[ndarray, "n 3 3"]:
    """Rotation matrices from xyzw quaternions (normalised first)."""
    x, y, z, w = (xyzw / np.linalg.norm(xyzw, axis=1, keepdims=True)).T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def compose_track(quest_t: Float64[ndarray, "n 3"], quest_xyzw: Float64[ndarray, "n 4"], mount: Float64[ndarray, "4 4"]) -> Float64[ndarray, "n 4 4"]:
    """``world_T_oak = world_T_quest @ quest_T_oak`` for every Quest sample."""
    world_T_quest = np.tile(np.eye(4), (len(quest_t), 1, 1))
    world_T_quest[:, :3, :3] = quaternion_matrices(quest_xyzw)
    world_T_quest[:, :3, 3] = quest_t
    return world_T_quest @ mount


def read_fitted_mount(view: DatasetEntry, oak: str) -> Float64[ndarray, "4 4"] | None:
    """The fitted Quest-left-eye → OAK-RGB mount of one segment's view, or None when only the prior (or nothing) exists."""
    static = view.filter_contents([oak]).reader(index=None).to_arrow_table()
    rotation, translation = _first(static, f"{oak}:Transform3D:mat3x3"), _first(static, f"{oak}:Transform3D:translation")
    if _first(static, f"{oak}:{MOUNT_SOURCE}") != "fitted" or rotation is None or translation is None:
        return None
    mount = np.eye(4)
    mount[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3).T  # Rerun mat3x3 is column-major
    mount[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return mount


def read_oak_track(
    view: DatasetEntry, oak: str, quest: str, *, start_ns: int | None = None, end_ns: int | None = None
) -> tuple[Float64[ndarray, "n"], Float64[ndarray, "n 4 4"]] | None:
    """OAK world poses on ``video_time`` (seconds, 4x4), or None when the segment has no fitted mount."""
    mount = read_fitted_mount(view, oak)
    if mount is None:
        return None
    frame = view.filter_contents([quest]).reader(index="video_time")
    if start_ns is not None:
        frame = frame.filter(col("video_time").cast(pa.int64()) >= lit(start_ns))
    if end_ns is not None:
        frame = frame.filter(col("video_time").cast(pa.int64()) <= lit(end_ns))
    poses = frame.select("video_time", f"{quest}:Transform3D:translation", f"{quest}:Transform3D:quaternion").to_arrow_table()
    poses = poses.filter(poses[f"{quest}:Transform3D:quaternion"].is_valid())
    if not len(poses):
        return None
    times_s = poses["video_time"].combine_chunks().cast(pa.int64()).to_numpy() * 1e-9
    translations = np.asarray(poses[f"{quest}:Transform3D:translation"].to_pylist(), dtype=np.float64).reshape(-1, 3)
    quaternions = np.asarray(poses[f"{quest}:Transform3D:quaternion"].to_pylist(), dtype=np.float64).reshape(-1, 4)
    return times_s, compose_track(translations, quaternions, mount)
