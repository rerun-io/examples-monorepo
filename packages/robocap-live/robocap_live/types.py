"""robocap-live edge values and their Rerun mapping: the rig and SLAM poses the hands core reads, and a layer run's counts."""

from collections.abc import Iterator
from dataclasses import dataclass, fields

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.time import STATIC, TimeNs
from dagkit.types import Log
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import from_dict, serde, to_dict
from simplecv.rerun_custom_types import CameraDistortion
from simplecv.types.rig import RigCameras

RIG_ROOT: str = "/world/rig_00"
"""The rig entity: its cameras ``cam_00..cam_05`` (base layer) and its SLAM pose (``slam_rs`` layer)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CapRig:
    """A RoboCap rig's six fisheye cameras as the segment's base layer stores them under the rig root (static)."""

    rig: RigCameras
    """``cam_00..cam_05``: pinhole K, ``cam_T_world`` = camera from rig, resolution."""
    names: tuple[str, ...]
    """Each camera's own name (``left_front`` ...), in index order; the core checks the order."""
    fisheye62: Float64[ndarray, "c 8"]
    """Kannala-Brandt ``[k1..k6, p1, p2]`` per camera."""

    def to_rerun(self, root: str) -> list[Log]:
        logs: list[Log] = self.rig.to_rerun(root)
        for camera, name, coefficients in zip(self.rig.names, self.names, self.fisheye62, strict=True):
            logs.append(Log(f"{root}/{camera}", rr.AnyValues(name=name), static=True))
            logs.append(Log(f"{root}/{camera}/pinhole", CameraDistortion("kannala_brandt", coefficients), static=True))
        return logs

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return RigCameras.contents(root)

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "CapRig"]]:
        """Raises ``ValueError`` naming a camera whose lens is not Kannala-Brandt with eight coefficients."""
        for t_ns, rig in RigCameras.from_layer(table, root, timeline):
            names: list[str] = []
            fisheye62: list[Float64[ndarray, " k"]] = []
            for camera in rig.names:
                (name,) = table.column(f"{root}/{camera}:name")[0].as_py()
                (model,) = table.column(f"{root}/{camera}/pinhole:simplecv.components.DistortionModel")[0].as_py()
                coefficients = np.asarray(
                    table.column(f"{root}/{camera}/pinhole:simplecv.components.DistortionCoefficients")[0].as_py(), dtype=np.float64
                ).ravel()
                if model != "kannala_brandt" or coefficients.size != 8:
                    raise ValueError(f"{root}/{camera}: {model} lens with {coefficients.size} coefficients; expected Fisheye62 [k1..k6, p1, p2]")
                names.append(name)
                fisheye62.append(coefficients)
            yield t_ns, cls(rig, tuple(names), np.stack(fisheye62))


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CatalogPoses:
    """A segment's whole SLAM pose table (``slam_rs``: the rig's Transform3D rows), sorted, finite; one STATIC value."""

    t_ns: Int64[ndarray, " p"]
    world_from_rig: Float64[ndarray, "p 4 4"]

    def to_rerun(self, root: str) -> list[Log]:
        raise NotImplementedError("CatalogPoses is read from the slam_rs layer, which slam-rs writes; it is never written from here")

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "CatalogPoses"]]:
        translation_column: str = f"{root}:Transform3D:translation"
        if not table.num_rows or translation_column not in table.column_names:
            return
        t_ns: Int64[ndarray, " r"] = np.asarray(table.column(timeline).cast(pa.int64()))
        quaternion: Float64[ndarray, "r 4"] = _rows(table.column(f"{root}:Transform3D:quaternion"), 4)
        translation: Float64[ndarray, "r 3"] = _rows(table.column(translation_column), 3)
        valid: Bool[ndarray, " r"] = (
            np.isfinite(quaternion).all(axis=1) & np.isfinite(translation).all(axis=1) & (np.linalg.norm(quaternion, axis=1) > 0)
        )
        order: Int64[ndarray, " p"] = np.flatnonzero(valid)[np.argsort(t_ns[valid], kind="stable")]
        if not order.size:  # no usable pose: no table (the hands node then refuses the segment)
            return
        world_from_rig: Float64[ndarray, "p 4 4"] = np.tile(np.eye(4), (len(order), 1, 1))
        world_from_rig[:, :3, :3] = Rotation.from_quat(quaternion[order]).as_matrix()  # xyzw, normalised
        world_from_rig[:, :3, 3] = translation[order]
        yield STATIC, cls(t_ns[order], world_from_rig)


def _rows(column: pa.ChunkedArray, width: int) -> Float64[ndarray, "r w"]:
    """A ``list<fixed_size_list<width>>`` component column as one row per table row; NaN where the row has no value."""
    out: Float64[ndarray, "r w"] = np.full((len(column), width), np.nan)
    for row, cell in enumerate(column.to_pylist()):
        if cell:
            out[row] = np.asarray(cell[0] if isinstance(cell[0], list) else cell, dtype=np.float64)[:width]
    return out


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LayerRun:
    """One hands-layer run: what the layer holds, and the core's stage totals in seconds (its stages run in parallel and overlap)."""

    framesets: int
    with_pose: int
    held_pose: int
    """Framesets without their own pose, tracked on the newest earlier one (the runtime's pose store rule)."""
    tracked_left: int
    tracked_right: int
    reported_left: int
    """Framesets on which the hand was reported (tracked and confirmed; only reported hands are drawn)."""
    reported_right: int
    scale: float
    """The hand scale phi at the end."""
    scale_final: bool
    """Whether the live scale calibration finished."""
    nets_load_s: float
    """ONNX Runtime, the two sessions, the tracker, the layer file and the pipeline's threads."""
    detnet_s: float
    crops_s: float
    keynet_s: float
    fit_s: float
    tracker_s: float
    finish_s: float
    """Draining the pipeline and closing the file."""

    def to_rerun(self, root: str) -> list[Log]:
        return [Log(root, rr.AnyValues(**to_dict(self)), static=True)]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "LayerRun"]]:
        if table.num_rows and f"{root}:framesets" in table.column_names:
            yield STATIC, from_dict(cls, {f.name: table.column(f"{root}:{f.name}")[0].as_py()[0] for f in fields(cls)})
