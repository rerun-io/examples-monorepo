"""mamma's edge Types (SPEC 5.2) and their Rerun mapping.

Tracks and 2D landmarks live on the decode device (memory only, not serde; ``to_rerun`` copies to the host); fits and 3D
landmarks are host values. Per view, everything under ``<root>/<view>/pinhole/mamma`` is in engine pixels below one static
``Transform3D`` scale to the stored video size; persons are ``person_NN`` (``NN`` = the tracker's stable person id).
"""

import io
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from dagkit.time import STATIC, TimeNs
from dagkit.types import Log, column, row_times
from jaxtyping import Bool, Float32, Int32, Int64
from numpy import ndarray
from PIL import Image
from serde import serde
from torch import Tensor

PERSON_RGB: tuple[int, int, int] = (255, 140, 0)
"""Predicted masks, landmarks and meshes (the GT layers draw in their own colours)."""
MESH_ALBEDO: tuple[float, float, float, float] = (1.0, 0.55, 0.0, 0.8)


def _person(path: str) -> int:
    return int(path.rsplit("person_", 1)[1].split("/")[0])


def _entities(table: pa.Table, root: str, component: str) -> list[str]:
    """Entities under ``root`` that carry ``component``, sorted."""
    return sorted(c.removesuffix(f":{component}") for c in table.column_names if c.startswith(f"{root}/") and c.endswith(f":{component}"))


def _view_logs(root: str, views: tuple[str, ...], scale_xy: Float32[ndarray, "v 2"]) -> list[Log]:
    """Static per view: the engine-to-stored scale of everything under ``pinhole/mamma``, and the view order."""
    logs: list[Log] = []
    for i, view in enumerate(views):
        logs.append(Log(f"{root}/{view}/pinhole/mamma", rr.Transform3D(scale=[*scale_xy[i].tolist(), 1.0]), static=True))
        logs.append(Log(f"{root}/{view}/pinhole/mamma", rr.AnyValues(view_index=i), static=True))
    return logs


def _views(table: pa.Table, root: str) -> tuple[tuple[str, ...], Float32[ndarray, "v 2"]]:
    """The views ``_view_logs`` wrote, in their order, with their scales (a rig in the same recording has its own ``view_index``)."""
    paths: list[str] = [p for p in _entities(table, root, "view_index") if p.endswith("/pinhole/mamma")]
    first: list[list] = [next(c for c in column(table, f"{p}:view_index") if c is not None) for p in paths]
    paths = [p for _, p in sorted(zip([f[0] for f in first], paths, strict=True))]
    scales: list[list[float]] = [next(c for c in column(table, f"{p}:Transform3D:scale") if c is not None)[0][:2] for p in paths]
    return tuple(p.removeprefix(f"{root}/").removesuffix("/pinhole/mamma") for p in paths), np.asarray(scales, dtype=np.float32).reshape(-1, 2)


@dataclass(frozen=True, slots=True)
class MultiViewTracks:
    """Person masks of one frameset at engine size, one row per (view, person) track."""

    frame_index: int
    views: tuple[str, ...]
    view: Int64[ndarray, " k"]
    """Row -> index into ``views``."""
    person: Int64[ndarray, " k"]
    """Row -> person id, shared across views and stable over time."""
    masks: Bool[Tensor, "k h w"]
    boxes_xyxy: Float32[ndarray, "k 4"]
    """Engine pixels; NaN rows where the mask is empty."""
    scores: Float32[ndarray, " k"]
    """Tracker confidence (predicted mask IoU)."""
    scale_xy: Float32[ndarray, "v 2"]
    """Stored video pixels per engine pixel, per view."""

    def to_rerun(self, root: str) -> list[Log]:
        masks: Bool[ndarray, "k h w"] = self.masks.cpu().numpy()
        logs: list[Log] = _view_logs(root, self.views, self.scale_xy)
        for v, view in enumerate(self.views):
            rows: Int64[ndarray, " r"] = np.flatnonzero(self.view == v)
            for row in rows.tolist():
                image: Image.Image = Image.fromarray(masks[row].astype(np.uint8), mode="P")
                image.putpalette([0, 0, 0, *PERSON_RGB])
                png = io.BytesIO()
                image.save(png, format="PNG", transparency=0)
                path: str = f"{root}/{view}/pinhole/mamma/masks/person_{self.person[row]:02d}"
                logs.append(Log(path, rr.EncodedImage(contents=png.getvalue(), media_type="image/png", opacity=0.5, draw_order=11.0)))
            persons: str = f"{root}/{view}/pinhole/mamma/persons"
            logs.append(Log(persons, rr.Boxes2D(array=self.boxes_xyxy[rows], array_format=rr.Box2DFormat.XYXY, class_ids=self.person[rows] + 1)))
            values = rr.AnyValues(person=self.person[rows], score=self.scores[rows], frame_index=self.frame_index, mask_hw=list(self.masks.shape[1:]))
            logs.append(Log(persons, values))
        return logs

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "MultiViewTracks"]]:
        """Masks come back as host tensors (a resumed value moves to the device in its consumer)."""
        views, scale_xy = _views(table, root)
        names: tuple[str, ...] = ("person", "score", "frame_index", "Boxes2D:centers", "Boxes2D:half_sizes", "mask_hw")
        columns: list[list[list]] = [[column(table, f"{root}/{view}/pinhole/mamma/persons:{name}") for name in names] for view in views]
        for row, t_ns in enumerate(row_times(table, timeline)):
            if all(view[2][row] is None for view in columns):
                continue
            person, score, frame, center, half, mask_hw = ([np.asarray(view[k][row] or [], dtype=np.float32) for view in columns] for k in range(len(names)))
            mask_row: pa.Table = table.slice(row, 1)
            blobs: list[bytes] = [
                bytes(column(mask_row, f"{root}/{view}/pinhole/mamma/masks/person_{int(p):02d}:EncodedImage:blob")[0][0])
                for view, people in zip(views, person, strict=True)
                for p in people.tolist()
            ]
            centers, halves = (np.concatenate([b.reshape(-1, 2) for b in boxes]) for boxes in (center, half))
            xyxy: Float32[ndarray, "k 4"] = np.concatenate([centers - halves, centers + halves], axis=1)
            masks: Bool[ndarray, "k h w"] = (
                np.stack([np.asarray(Image.open(io.BytesIO(b))) > 0 for b in blobs]) if blobs else np.zeros((0, *next(m for m in mask_hw if m.size).astype(int)), dtype=bool)
            )
            yield t_ns, cls(
                int(next(f for f in frame if f.size)[0]), views, np.repeat(np.arange(len(views), dtype=np.int64), [p.size for p in person]),
                np.concatenate(person).astype(np.int64), torch.as_tensor(masks),
                xyxy, np.concatenate(score), scale_xy,
            )  # fmt: skip


@dataclass(frozen=True, slots=True)
class MultiViewLandmarks:
    """MammaNet's dense landmarks of one frameset, one row per (view, person) track, in engine pixels."""

    frame_index: int
    views: tuple[str, ...]
    view: Int64[ndarray, " k"]
    person: Int64[ndarray, " k"]
    xy: Float32[Tensor, "k n 2"]
    log_variance: Float32[Tensor, "k n"]
    """Log-variance in normalised crop units: sigma_px = exp(0.5 * log_variance) * crop_px / 2."""
    visibility: Float32[Tensor, "k n"]
    floor_contact: Float32[Tensor, "k n"]
    scale_xy: Float32[ndarray, "v 2"]
    """Stored video pixels per engine pixel, per view."""

    def to_rerun(self, root: str) -> list[Log]:
        xy, log_variance, visibility, floor_contact = (t.cpu().numpy() for t in (self.xy, self.log_variance, self.visibility, self.floor_contact))
        logs: list[Log] = _view_logs(root, self.views, self.scale_xy)
        logs += [Log(f"{root}/{view}/pinhole/mamma/landmarks", rr.AnyValues(frame_index=self.frame_index)) for view in self.views]
        for row in range(self.view.size):
            path: str = f"{root}/{self.views[self.view[row]]}/pinhole/mamma/landmarks/person_{self.person[row]:02d}"
            colors: Int64[ndarray, "n 3"] = np.where(visibility[row][:, None] > 0.5, np.asarray(PERSON_RGB), 90)
            logs.append(Log(path, rr.Points2D(xy[row], radii=1.5, colors=colors)))
            values = rr.AnyValues(log_variance=log_variance[row], visibility=visibility[row], floor_contact=floor_contact[row])
            logs.append(Log(path, values))
        return logs

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "MultiViewLandmarks"]]:
        """An empty frameset comes back with ``k = n = 0`` (as ``LandmarksNode`` emits it)."""
        views, scale_xy = _views(table, root)
        paths: list[str] = [p for p in _entities(table, root, "floor_contact") if "/pinhole/mamma/landmarks/person_" in p]
        names: tuple[str, ...] = ("Points2D:positions", "log_variance", "visibility", "floor_contact")
        columns: dict[str, list[list]] = {path: [column(table, f"{path}:{name}") for name in names] for path in paths}
        frames: list[list] = [column(table, f"{root}/{view}/pinhole/mamma/landmarks:frame_index") for view in views]
        for row, t_ns in enumerate(row_times(table, timeline)):
            frame: list | None = next((f[row] for f in frames if f[row] is not None), None)
            if frame is None:
                continue
            keyed: list[tuple[int, int, str]] = sorted(
                (views.index(p.removeprefix(f"{root}/").split("/pinhole/")[0]), _person(p), p) for p in paths if columns[p][3][row] is not None
            )
            n: int = len(columns[keyed[0][2]][3][row]) if keyed else 0
            xy, log_variance, visibility, floor_contact = (
                torch.as_tensor(np.asarray([columns[p][k][row] for _, _, p in keyed], dtype=np.float32).reshape(len(keyed), n, *tail))
                for k, tail in enumerate(((2,), (), (), ()))
            )
            yield t_ns, cls(
                int(frame[0]), views, np.asarray([k[0] for k in keyed], dtype=np.int64),
                np.asarray([k[1] for k in keyed], dtype=np.int64), xy, log_variance, visibility, floor_contact, scale_xy,
            )  # fmt: skip


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SmplxFits:
    """SMPL-X fits (neutral body, 16 betas) of the persons fitted at one frame time; world metres, z up."""

    frame_index: int
    persons: tuple[int, ...]
    vertices: Float32[ndarray, "p v 3"]
    pose: Float32[ndarray, "p 165"]
    """Axis-angle: global, body, jaw, eyes, hands."""
    betas: Float32[ndarray, "p b"]
    trans: Float32[ndarray, "p 3"]

    def to_rerun(self, root: str) -> list[Log]:
        logs: list[Log] = []
        for i, person in enumerate(self.persons):
            logs.append(Log(f"{root}/person_{person:02d}/body/mesh", rr.Mesh3D.from_fields(vertex_positions=self.vertices[i])))
            params = rr.AnyValues(pose=self.pose[i], betas=self.betas[i], trans=self.trans[i], frame_index=self.frame_index)
            logs.append(Log(f"{root}/person_{person:02d}/body/smplx", params))
        return logs

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "SmplxFits"]]:
        people: list[str] = [p.removesuffix("/body/smplx") for p in _entities(table, root, "frame_index") if p.endswith("/body/smplx")]
        names: tuple[str, ...] = ("pose", "betas", "trans", "frame_index")
        columns: dict[str, list[list]] = {p: [column(table, f"{p}/body/mesh:Mesh3D:vertex_positions")] + [column(table, f"{p}/body/smplx:{n}") for n in names] for p in people}
        for row, t_ns in enumerate(row_times(table, timeline)):
            fitted: list[str] = [p for p in people if columns[p][4][row] is not None]
            if fitted:
                vertices, pose, betas, trans = (np.asarray([columns[p][k][row] for p in fitted], dtype=np.float32) for k in range(4))
                yield t_ns, cls(int(columns[fitted[0]][4][row][0]), tuple(_person(p) for p in fitted), vertices, pose, betas, trans)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BodyTopology:
    """The SMPL-X triangle list, sent once per person when its fit starts (the fits carry only vertices)."""

    persons: tuple[int, ...]
    faces: Int32[ndarray, "f 3"]

    def to_rerun(self, root: str) -> list[Log]:
        """Static per person: the source tag on the person node (as the GT people carry ``mosh_gt``) and the mesh topology."""
        mesh = rr.Mesh3D.from_fields(triangle_indices=self.faces, albedo_factor=MESH_ALBEDO)
        source = rr.AnyValues(source="mamma_pred")
        return [Log(f"{root}/person_{p:02d}{path}", data, static=True) for p in self.persons for path, data in (("", source), ("/body/mesh", mesh))]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "BodyTopology"]]:
        meshes: list[str] = _entities(table, root, "Mesh3D:triangle_indices")
        if meshes:
            faces: list = next(c for c in column(table, f"{meshes[0]}:Mesh3D:triangle_indices") if c is not None)
            yield STATIC, cls(tuple(_person(p) for p in meshes), np.asarray(faces, dtype=np.int32).reshape(-1, 3))


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Landmarks3d:
    """Triangulated dense landmarks of one tick per person; invalid points are 0 (SPEC 5.2 policy)."""

    frame_index: int
    persons: tuple[int, ...]
    points: Float32[ndarray, "p n 3"]
    valid: Bool[ndarray, "p n"]

    def to_rerun(self, root: str) -> list[Log]:
        logs: list[Log] = []
        for i, person in enumerate(self.persons):
            logs.append(Log(f"{root}/person_{person:02d}/landmarks3d", rr.Points3D(self.points[i][self.valid[i]], radii=0.008, colors=PERSON_RGB)))
            logs.append(Log(f"{root}/person_{person:02d}/landmarks3d", rr.AnyValues(valid=self.valid[i], frame_index=self.frame_index)))
        return logs

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "Landmarks3d"]]:
        paths: list[str] = [p for p in _entities(table, root, "valid") if p.endswith("/landmarks3d")]
        columns: dict[str, list[list]] = {p: [column(table, f"{p}:{name}") for name in ("Points3D:positions", "valid", "frame_index")] for p in paths}
        for row, t_ns in enumerate(row_times(table, timeline)):
            present: list[str] = [p for p in paths if columns[p][1][row] is not None]
            if not present:
                continue
            valid: Bool[ndarray, "p n"] = np.asarray([columns[p][1][row] for p in present], dtype=bool)
            points: Float32[ndarray, "p n 3"] = np.zeros((*valid.shape, 3), dtype=np.float32)
            for i, path in enumerate(present):
                points[i][valid[i]] = np.asarray(columns[path][0][row] or [], dtype=np.float32).reshape(-1, 3)
            yield t_ns, cls(int(columns[present[0]][2][row][0]), tuple(_person(p) for p in present), points, valid)
