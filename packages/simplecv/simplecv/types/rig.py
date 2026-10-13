"""A static pinhole camera rig and its Rerun mapping: ``<root>/<name>`` (cam_T_world) with a ``pinhole`` child."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.time import STATIC, TimeNs
from dagkit.types import Log
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import serde

FRUSTUM_DEPTH_M: float = 0.1
"""Image-plane distance of the logged frusta; dataforge base layers draw their GT frusta at the same depth."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RigCameras:
    """Per camera, in ``names`` order: pinhole intrinsics, world-to-camera pose (RDF, metric) and image size."""

    names: tuple[str, ...]
    """Camera entity names relative to the root, e.g. ``rig_00/cam_00``."""
    intrinsics: Float64[ndarray, "v 3 3"]
    cam_T_world: Float64[ndarray, "v 4 4"]
    resolution_wh: Int64[ndarray, "v 2"]

    def select(self, names: Sequence[str]) -> "RigCameras":
        """The cameras ``names`` in that order (KeyError names a missing one)."""
        index: dict[str, int] = {name: i for i, name in enumerate(self.names)}
        rows: list[int] = [index[name] for name in names]
        return RigCameras(tuple(names), self.intrinsics[rows], self.cam_T_world[rows], self.resolution_wh[rows])

    def scaled_to(self, height: int, width: int) -> "RigCameras":
        """Every camera's intrinsics rescaled to a ``height x width`` image (K rows 0 and 1 times the size ratios)."""
        ratio_xy: Float64[ndarray, "v 2"] = np.array([width, height], dtype=np.float64) / self.resolution_wh
        intrinsics: Float64[ndarray, "v 3 3"] = self.intrinsics.copy()
        intrinsics[:, :2] *= ratio_xy[:, :, None]
        return RigCameras(self.names, intrinsics, self.cam_T_world, np.tile(np.array([width, height], dtype=np.int64), (len(self.names), 1)))

    def to_rerun(self, root: str) -> list[Log]:
        """Static: a rig never changes over a stream. ``view_index`` keeps the camera order through ``from_layer``."""
        logs: list[Log] = []
        for i, (name, K, T, wh) in enumerate(zip(self.names, self.intrinsics, self.cam_T_world, self.resolution_wh, strict=True)):
            transform = rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3], relation=rr.TransformRelation.ChildFromParent)
            pinhole = rr.Pinhole(image_from_camera=K, resolution=wh, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=FRUSTUM_DEPTH_M)
            logs += [Log(f"{root}/{name}", transform, static=True), Log(f"{root}/{name}", rr.AnyValues(view_index=i), static=True)]
            logs.append(Log(f"{root}/{name}/pinhole", pinhole, static=True))
        return logs

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "RigCameras"]]:
        """Every camera under ``root`` with a static pose AND intrinsics; nothing when there is none.

        Cameras come back in the order ``to_rerun`` wrote them when every one carries a ``view_index``, else sorted by name
        (a dataset's own GT rig).
        """
        wanted: list[str] = [c for c in table.column_names if c.startswith(f"{root}/") and (":Transform3D:" in c or ":Pinhole:" in c or c.endswith(":view_index"))]
        row: dict[str, object] = {c: table.column(c)[0].as_py() for c in wanted} if table.num_rows else {}

        def value(column: str) -> Float64[ndarray, " k"]:
            return np.asarray(row[column], dtype=np.float64).reshape(-1)

        suffix: str = ":Transform3D:translation"
        posed: list[str] = [c.removeprefix(f"{root}/").removesuffix(suffix) for c in wanted if c.endswith(suffix) and row[c] is not None]
        names: list[str] = sorted(n for n in posed if row.get(f"{root}/{n}/pinhole:Pinhole:image_from_camera") is not None)
        if not names:
            return
        if all(row.get(f"{root}/{n}:view_index") is not None for n in names):
            names.sort(key=lambda n: (int(value(f"{root}/{n}:view_index")[0]), n))
        cam_T_world: Float64[ndarray, "v 4 4"] = np.tile(np.eye(4), (len(names), 1, 1))
        for i, name in enumerate(names):  # Rerun stores 3x3 matrices column-major
            cam_T_world[i, :3, :3] = value(f"{root}/{name}:Transform3D:mat3x3").reshape(3, 3).T
            cam_T_world[i, :3, 3] = value(f"{root}/{name}{suffix}")
            if row.get(f"{root}/{name}:Transform3D:relation") is None or int(value(f"{root}/{name}:Transform3D:relation")[0]) != 2:  # ChildFromParent
                cam_T_world[i] = np.linalg.inv(cam_T_world[i])
        intrinsics = np.stack([value(f"{root}/{n}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T for n in names])
        resolution = np.stack([value(f"{root}/{n}/pinhole:Pinhole:resolution") for n in names]).round().astype(np.int64)
        yield STATIC, cls(tuple(names), intrinsics, cam_T_world, resolution)
