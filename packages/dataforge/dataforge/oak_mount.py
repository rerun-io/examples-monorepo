"""The OAK rig hangs on the Quest's left-eye frame by one mount: a nominal prior in the base layer, replaced by the
fitted mount in the calibration layer (see ``PRIOR_LEAD_NS``). This module turns that into OAK world poses for code
that needs them per frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from datafusion import col, lit
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetSelection
from scipy.spatial.transform import Rotation

MOUNT_SOURCE = "oak_mount_source"
"""AnyValues field on the OAK rig: ``nominal`` (base prior) or ``fitted`` (calibration)."""
PRIOR_LEAD_NS = 1_000_000
"""The base logs the nominal prior this long before the first Quest pose; calibration logs its fit at that pose. The
viewer resolves the latest timed transform (ahead of a static one), so the fit wins whichever layer was written last."""


def _first(table: pa.Table, name: str):  # noqa: ANN202
    if name not in table.column_names:
        return None
    for value in table[name].to_pylist():
        while isinstance(value, list) and len(value) == 1:
            value = value[0]
        if value is not None:
            return value
    return None


def _rigid(column_major_rotation: list[float] | Float64[ndarray, "9"], translation: list[float] | Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """A 4x4 pose from a Transform3D's stored values; Rerun's mat3x3 is column-major."""
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3] = np.asarray(column_major_rotation, dtype=np.float64).reshape(3, 3).T
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return pose


def quaternion_matrices(xyzw: Float64[ndarray, "n 4"]) -> Float64[ndarray, "n 3 3"]:
    """Rotation matrices from xyzw quaternions (normalised first)."""
    return Rotation.from_quat(xyzw).as_matrix()


def compose_track(quest_t: Float64[ndarray, "n 3"], quest_xyzw: Float64[ndarray, "n 4"], mount: Float64[ndarray, "4 4"]) -> Float64[ndarray, "n 4 4"]:
    """``world_T_oak = world_T_quest @ quest_T_oak`` for every Quest sample."""
    world_T_quest = np.tile(np.eye(4), (len(quest_t), 1, 1))
    world_T_quest[:, :3, :3] = quaternion_matrices(quest_xyzw)
    world_T_quest[:, :3, 3] = quest_t
    return world_T_quest @ mount


def read_fitted_mount(view: DatasetSelection, oak: str) -> Float64[ndarray, "4 4"] | None:
    """The fitted Quest-left-eye → OAK-RGB mount of one segment's view, or None when only the prior (or nothing) exists."""
    static = view.filter_contents([oak]).reader(index=None).to_arrow_table()
    rotation, translation = _first(static, f"{oak}:Transform3D:mat3x3"), _first(static, f"{oak}:Transform3D:translation")
    if _first(static, f"{oak}:{MOUNT_SOURCE}") != "fitted" or rotation is None or translation is None:
        return None
    return _rigid(rotation, translation)


def read_rig_track(
    view: DatasetSelection, rig: str, *, start_ns: int | None = None, end_ns: int | None = None
) -> tuple[Float64[ndarray, "n"], Float64[ndarray, "n 4 4"]] | None:
    """A rig's world poses on ``video_time`` (seconds, 4x4), or None when it has no pose in the window."""
    frame = view.filter_contents([rig]).reader(index="video_time")
    if start_ns is not None:
        frame = frame.filter(col("video_time").cast(pa.int64()) >= lit(start_ns))
    if end_ns is not None:
        frame = frame.filter(col("video_time").cast(pa.int64()) <= lit(end_ns))
    rows: pa.Table = frame.select("video_time", f"{rig}:Transform3D:translation", f"{rig}:Transform3D:quaternion").to_arrow_table()
    rows = rows.filter(rows[f"{rig}:Transform3D:quaternion"].is_valid())
    if not len(rows):
        return None
    times_s = rows["video_time"].combine_chunks().cast(pa.int64()).to_numpy() * 1e-9
    translations = np.asarray(rows[f"{rig}:Transform3D:translation"].to_pylist(), dtype=np.float64).reshape(-1, 3)
    quaternions = np.asarray(rows[f"{rig}:Transform3D:quaternion"].to_pylist(), dtype=np.float64).reshape(-1, 4)
    return times_s, compose_track(translations, quaternions, np.eye(4))


def read_oak_track(
    view: DatasetSelection, oak: str, quest: str, *, start_ns: int | None = None, end_ns: int | None = None, poses: DatasetSelection | None = None
) -> tuple[Float64[ndarray, "n"], Float64[ndarray, "n 4 4"]] | None:
    """OAK world poses on ``video_time`` (seconds, 4x4), or None when the segment has no fitted mount.

    The mount comes from ``view``; the Quest track from ``poses`` when the mount lives in a calibration-only layer.
    """
    mount = read_fitted_mount(view, oak)
    if mount is None:
        return None
    track = read_rig_track(view if poses is None else poses, quest, start_ns=start_ns, end_ns=end_ns)
    if track is None:
        return None
    return track[0], track[1] @ mount


@dataclass(frozen=True, slots=True)
class StaticCamera:
    """What a segment's layers hold statically for one camera; a field is None when no layer holds it."""

    rig_T_cam: Float64[ndarray, "4 4"] | None
    """Camera in its rig frame, RDF metres."""
    K: Float64[ndarray, "3 3"] | None
    """Encoded-image pinhole matrix."""
    resolution: tuple[int, int] | None
    """Image width, height."""
    distortion: Float64[ndarray, "c"] | None
    """OpenCV coefficients, k1 first."""


def read_static_cameras(calibration: DatasetSelection, base: DatasetSelection, cameras: list[str]) -> dict[str, StaticCamera]:
    """The static calibration of one segment's camera entities, calibration layer over base layer, component by component.

    The base holds factory facts (OAK and healthy-Quest stereo, lenses); the calibration layer holds fitted lenses and a
    borrowed camera-50 transform. A faulty headset's base has no right-eye transform, so an unresolved one stays None.
    """
    entities: list[str] = [path for camera in cameras for path in (camera, f"{camera}/pinhole")]
    layers: list[pa.Table] = [layer.filter_contents(entities).reader(index=None).to_arrow_table() for layer in (calibration, base)]

    def value(name: str) -> Float64[ndarray, "k"] | None:
        found = next((item for item in (_first(layer, name) for layer in layers) if item is not None), None)
        return None if found is None else np.asarray(found, dtype=np.float64).reshape(-1)

    result: dict[str, StaticCamera] = {}
    for camera in cameras:
        rotation, translation = value(f"{camera}:Transform3D:mat3x3"), value(f"{camera}:Transform3D:translation")
        matrix, size = value(f"{camera}/pinhole:Pinhole:image_from_camera"), value(f"{camera}/pinhole:Pinhole:resolution")
        result[camera] = StaticCamera(
            None if rotation is None or translation is None else _rigid(rotation, translation),
            None if matrix is None else matrix.reshape(3, 3).T,  # Rerun matrices are column-major
            None if size is None else (int(size[0]), int(size[1])),
            value(f"{camera}/pinhole:simplecv.components.DistortionCoefficients"),
        )
    return result
