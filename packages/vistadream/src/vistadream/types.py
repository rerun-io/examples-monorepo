"""Edge Types of the single-image graph (SPEC 5.4) and their Rerun mapping.

Host Types are pyserde records; ``Splats`` and the ``Scene`` that holds it keep device tensors (memory only, ``to_rerun`` copies
to the host). Cameras are RDF; a pose is ``cam_T_world`` (world to camera), logged as a ``ChildFromParent`` transform.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from dagkit.time import TimeNs
from dagkit.types import Log, cells, flat
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from serde import serde
from torch import Tensor

Stage: TypeAlias = Literal["no-outpaint", "outpaint", "coarse"]
"""``no-outpaint``: the image as is; ``outpaint``: FLUX fills a border first; ``coarse``: then adds inpainted views."""


def images(table: pa.Table, entity: str, archetype: str, timeline: str, dtype: type) -> dict[TimeNs, ndarray]:
    """Every ``Image`` / ``DepthImage`` logged at ``entity`` as an ``h w [c]`` array of ``dtype``, keyed by time."""
    formats: dict[TimeNs, pa.Scalar] = cells(table, f"{entity}:{archetype}:format", timeline)
    found: dict[TimeNs, ndarray] = {}
    for t, cell in cells(table, f"{entity}:{archetype}:buffer", timeline).items():
        spec: dict = formats[t].as_py()[0]
        image: ndarray = np.frombuffer(flat(cell, np.uint8).tobytes(), dtype=dtype).reshape(spec["height"], spec["width"], -1)
        found[t] = image[..., 0] if image.shape[-1] == 1 else image
    return found


def _mask(image: UInt8[ndarray, "h w"] | None) -> Bool[ndarray, "h w"] | None:
    return None if image is None else image > 0


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RgbImage:
    """The input image after resizing (sides multiples of 32, long side capped)."""

    rgb: UInt8[ndarray, "h w 3"]

    def to_rerun(self, root: str) -> list[Log]:
        return [Log(root, rr.Image(self.rgb, color_model="RGB"))]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "RgbImage"]]:
        for t, rgb in images(table, root, "Image", timeline, np.uint8).items():
            yield t, cls(rgb)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class OutpaintCanvas:
    """The image the depth model sees: the input, or the input inside a border FLUX filled."""

    rgb: UInt8[ndarray, "h w 3"]
    mask: Bool[ndarray, "h w"]
    """True where the canvas was outpainted (all False without outpainting)."""
    box_xywh: Int64[ndarray, "4"]
    """Where the input sits in the canvas (pixels)."""

    def to_rerun(self, root: str) -> list[Log]:
        return [
            Log(f"{root}/rgb", rr.Image(self.rgb, color_model="RGB")),
            Log(f"{root}/mask", rr.Image(self.mask.astype(np.uint8) * 255, color_model="L")),
            Log(root, rr.AnyValues(box_xywh=self.box_xywh)),
        ]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root, f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "OutpaintCanvas"]]:
        masks: dict[TimeNs, ndarray] = images(table, f"{root}/mask", "Image", timeline, np.uint8)
        boxes: dict[TimeNs, pa.Scalar] = cells(table, f"{root}:box_xywh", timeline)
        for t, rgb in images(table, f"{root}/rgb", "Image", timeline, np.uint8).items():
            yield t, cls(rgb, masks[t] > 0, np.asarray(boxes[t].as_py(), dtype=np.int64).reshape(4))


def _pinhole(K: Float64[ndarray, "3 3"], height: int, width: int) -> rr.Pinhole:
    return rr.Pinhole(image_from_camera=K, resolution=[width, height], camera_xyz=rr.ViewCoordinates.RDF)


def _intrinsics(table: pa.Table, entity: str, timeline: str) -> dict[TimeNs, Float64[ndarray, "3 3"]]:
    """Rerun stores 3x3 matrices column-major."""
    return {t: flat(c, np.float32).reshape(3, 3).T.astype(np.float64) for t, c in cells(table, f"{entity}:Pinhole:image_from_camera", timeline).items()}


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RelativeDepth:
    """MoGe's scale-free depth of the canvas, its validity mask and the intrinsics it estimated."""

    depth: Float32[ndarray, "h w"]
    """Non-finite predictions are 0."""
    valid: Bool[ndarray, "h w"]
    K: Float64[ndarray, "3 3"]

    def to_rerun(self, root: str) -> list[Log]:
        return [
            Log(root, _pinhole(self.K, *self.depth.shape)),
            Log(f"{root}/depth", rr.DepthImage(self.depth, meter=1.0)),
            Log(f"{root}/valid", rr.Image(self.valid.astype(np.uint8) * 255, color_model="L")),
        ]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root, f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "RelativeDepth"]]:
        valid: dict[TimeNs, ndarray] = images(table, f"{root}/valid", "Image", timeline, np.uint8)
        intrinsics: dict[TimeNs, ndarray] = _intrinsics(table, root, timeline)
        for t, depth in images(table, f"{root}/depth", "DepthImage", timeline, np.float32).items():
            yield t, cls(depth, valid[t] > 0, intrinsics[t])


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CameraView:
    """A posed pinhole image: the input, an outpainted or inpainted training view, or a render of the scene."""

    rgb: UInt8[ndarray, "h w 3"]
    K: Float64[ndarray, "3 3"]
    cam_T_world: Float64[ndarray, "4 4"]
    depth: Float32[ndarray, "h w"] | None = None
    """Relative depth of a training view (renders carry none)."""
    valid: Bool[ndarray, "h w"] | None = None
    """MoGe's validity mask of ``depth``."""
    supervise: Bool[ndarray, "h w"] | None = None
    """Pixels the training loss compares; None = every pixel."""

    def to_rerun(self, root: str) -> list[Log]:
        transform = rr.Transform3D(translation=self.cam_T_world[:3, 3], mat3x3=self.cam_T_world[:3, :3], relation=rr.TransformRelation.ChildFromParent)
        logs: list[Log] = [Log(root, transform), Log(f"{root}/pinhole", _pinhole(self.K, *self.rgb.shape[:2]))]
        logs.append(Log(f"{root}/pinhole/rgb", rr.Image(self.rgb, color_model="RGB")))
        if self.depth is not None:
            logs.append(Log(f"{root}/pinhole/depth", rr.DepthImage(self.depth, meter=1.0)))
        masks: tuple[tuple[str, Bool[ndarray, "h w"] | None], ...] = (("valid", self.valid), ("supervise", self.supervise))
        logs += [Log(f"{root}/pinhole/{name}", rr.Image(m.astype(np.uint8) * 255, color_model="L")) for name, m in masks if m is not None]
        return logs

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root, f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "CameraView"]]:
        intrinsics: dict[TimeNs, ndarray] = _intrinsics(table, f"{root}/pinhole", timeline)
        rotations: dict[TimeNs, pa.Scalar] = cells(table, f"{root}:Transform3D:mat3x3", timeline)
        translations: dict[TimeNs, pa.Scalar] = cells(table, f"{root}:Transform3D:translation", timeline)
        depths: dict[TimeNs, ndarray] = images(table, f"{root}/pinhole/depth", "DepthImage", timeline, np.float32)
        masks: dict[str, dict[TimeNs, ndarray]] = {n: images(table, f"{root}/pinhole/{n}", "Image", timeline, np.uint8) for n in ("valid", "supervise")}
        for t, rgb in images(table, f"{root}/pinhole/rgb", "Image", timeline, np.uint8).items():
            cam_T_world: Float64[ndarray, "4 4"] = np.eye(4)
            cam_T_world[:3, :3] = flat(rotations[t], np.float32).reshape(3, 3).T
            cam_T_world[:3, 3] = flat(translations[t], np.float32)
            yield t, cls(rgb, intrinsics[t], cam_T_world, depths.get(t), _mask(masks["valid"].get(t)), _mask(masks["supervise"].get(t)))


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SceneViews:
    """The scene's training views in training order: the outpainted canvas (if any), the input, then views ``expand`` added."""

    views: tuple[CameraView, ...]
    input_index: int
    """Which view is the input image (identity pose)."""

    @property
    def input(self) -> CameraView:
        return self.views[self.input_index]

    def names(self) -> list[str]:
        """Entity names under the root: ``input``, ``outpaint`` for the canvas, ``view_<i>`` for added views."""
        return ["input" if i == self.input_index else "outpaint" if i < self.input_index else f"view_{i:02d}" for i in range(len(self.views))]

    def to_rerun(self, root: str) -> list[Log]:
        return [log for name, view in zip(self.names(), self.views, strict=True) for log in view.to_rerun(f"{root}/{name}")]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "SceneViews"]]:
        """One value per time the input view has; each snapshot holds the views logged at that time (``expand`` adds some)."""
        suffix: str = "/pinhole/rgb:Image:buffer"
        names: list[str] = [c.removeprefix(f"{root}/").removesuffix(suffix) for c in table.column_names if c.startswith(f"{root}/") and c.endswith(suffix)]
        order: list[str] = sorted(names, key=lambda n: (n != "outpaint", n))  # outpaint, input, view_02, view_03, ...
        per_view: dict[str, dict[TimeNs, CameraView]] = {n: dict(CameraView.from_layer(table, f"{root}/{n}", timeline)) for n in order}
        for t in per_view.get("input", {}):
            present: list[str] = [n for n in order if t in per_view[n]]
            yield t, cls(tuple(per_view[n][t] for n in present), present.index("input"))


@dataclass(frozen=True, slots=True)
class Splats:
    """Activated 3D Gaussians with a degree-0 colour, on the device that made them (GPU edge value, not serde)."""

    centers: Float32[Tensor, "n 3"]
    scales: Float32[Tensor, "n 3"]
    """Per-axis standard deviations (scene units)."""
    quats_wxyz: Float32[Tensor, "n 4"]
    """Unit quaternions, scalar first (gsplat's order)."""
    rgb: Float32[Tensor, "n 3"]
    """Base colour in [0, 1]."""
    opacity: Float32[Tensor, " n"]
    """Peak opacity in [0, 1]."""

    def to_rerun(self, root: str) -> list[Log]:
        """Rerun's native ``GaussianSplats3D``: RGBA = base colour and opacity (uint8), xyzw quaternions, linear scales."""
        rgba: UInt8[Tensor, "n 4"] = (torch.cat([self.rgb, self.opacity[:, None]], dim=1) * 255.0).round().clamp(0, 255).byte()
        splats = rr.GaussianSplats3D(
            centers=self.centers.detach().cpu().numpy(),
            scales=self.scales.detach().cpu().numpy(),
            quaternions=self.quats_wxyz[:, [1, 2, 3, 0]].detach().cpu().numpy(),
            colors=rgba.cpu().numpy(),
            spherical_harmonics_degree=0,
        )
        return [Log(root, splats)]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "Splats"]]:
        """Host tensors; colour and opacity come back quantised to 1/255 (Rerun stores RGBA8)."""
        columns: list[dict[TimeNs, pa.Scalar]] = [cells(table, f"{root}:GaussianSplats3D:{c}", timeline) for c in ("centers", "scales", "quaternions", "colors")]
        for t, centers in columns[0].items():
            rgba: Float32[ndarray, "n 4"] = flat(columns[3][t], np.uint32)[:, None].view(np.uint8)[:, ::-1].astype(np.float32) / 255.0  # 0xRRGGBBAA
            quats: Float32[ndarray, "n 4"] = flat(columns[2][t], np.float32).reshape(-1, 4)[:, [3, 0, 1, 2]]
            yield t, cls(*(torch.from_numpy(np.ascontiguousarray(a)) for a in (flat(centers, np.float32).reshape(-1, 3), flat(columns[1][t], np.float32).reshape(-1, 3), quats, rgba[:, :3], rgba[:, 3])))


@dataclass(frozen=True, slots=True)
class Scene:
    """The Gaussian scene: activated 3D Gaussians on the device that made them, with the training views they were fitted to

    One value, so a render never meets splats without their views."""

    splats: Splats
    views: SceneViews

    def to_rerun(self, root: str) -> list[Log]:
        """The splats at ``<root>/splats``, the views under ``root`` (``SceneViews.names``)."""
        return [*self.splats.to_rerun(f"{root}/splats"), *self.views.to_rerun(root)]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "Scene"]]:
        views: dict[TimeNs, SceneViews] = dict(SceneViews.from_layer(table, root, timeline))
        for t, splats in Splats.from_layer(table, f"{root}/splats", timeline):
            yield t, cls(splats, views[t])
