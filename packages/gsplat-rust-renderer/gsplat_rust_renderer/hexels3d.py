"""`Hexels3D`: planar hexagonal primitives (hexel PLY contract) as a custom Rerun archetype.

The custom viewer renders `Hexels3D` with the gsplat-core hexel primitive. The same entity also
carries a `Points3D` fallback (centres in their base colour), so stock Rerun shows the centres.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast, get_args

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, UInt8
from numpy import ndarray
from plyfile import PlyData, PlyElement

HexelVariant: TypeAlias = Literal["v1", "v2", "v3"]
HEXELS_ENTITY: str = "/world/hexels"
# Screen-space radius of the Points3D fallback, in UI points (the custom viewer's capacity fallback matches it).
FALLBACK_POINT_RADIUS: float = 1.5
BASE_PROPERTIES: tuple[str, ...] = (
    *("x", "y", "z", "qw", "qx", "qy", "qz", "su", "sv", "opacity", "sharpness"),
    *(f"{part}_{channel}" for part in ("c0", "gu", "gv") for channel in "rgb"),
)


def property_names(variant: HexelVariant) -> list[str]:
    """Vertex properties of a 3D hexel PLY in contract order (packages/hexels/docs/hexel-ply.md)."""
    extra: list[str] = {"v1": [], "v2": [f"a_{i // 3}{i % 3}" for i in range(9)], "v3": [f"sh_{i}" for i in range(45)]}[variant]
    return [*BASE_PROPERTIES, *extra]


def descriptor(field: str, component_type: str | None) -> rr.ComponentDescriptor:
    """`Hexels3D:<field>`; untyped fields keep `component_type=None`."""
    return rr.ComponentDescriptor(f"Hexels3D:{field}", archetype="Hexels3D", component_type=component_type)


def untyped(field: str, values: Float32[ndarray, "n"] | Float32[ndarray, "n k"]) -> rr.DescribedComponentBatch:
    """f32 rows (scalars or fixed-size lists) under an untyped `Hexels3D` descriptor."""
    flat: pa.Array = pa.array(values.reshape(-1), type=pa.float32())
    array: pa.Array = flat if values.ndim == 1 else pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    field_descriptor: rr.ComponentDescriptor = descriptor(field, None)
    return rr.DescribedComponentBatch(rr.AnyBatchValue(field_descriptor, array), field_descriptor)


@dataclass(frozen=True)
class Hexels3D(rr.AsComponents):
    """Activated, render-ready hexel arrays in f32: the PLY values unchanged except the quaternion order."""

    centers: Float32[ndarray, "n 3"]
    """World-space centres p."""
    quaternions_xyzw: Float32[ndarray, "n 4"]
    """Unit rotations, Rerun order xyzw (the PLY stores wxyz); tangents t_u = R e_x, t_v = R e_y."""
    scales: Float32[ndarray, "n 2"]
    """Tangent scales (su, sv) in world units."""
    opacities: Float32[ndarray, "n"]
    """Activated opacity o in [0, 1]."""
    sharpness: Float32[ndarray, "n"]
    """Edge sharpness beta > 0 of the soft hexagon support."""
    color_field: Float32[ndarray, "n 9"]
    """Base colour c0 and its u/v gradients, RGB each; unconstrained, so not an RGBA8 colour."""
    view_linear: Float32[ndarray, "n 9"] | None = None
    """V2 only: row-major linear view matrix A, colour term A d."""
    sh_coefficients: Float32[ndarray, "n 45"] | None = None
    """V3 only: SH degrees 1-3, coefficient-major with RGB innermost (upstream basis order)."""

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        """Typed batches where a Rerun component fits; f32 lists without a component type otherwise."""
        batches: list[rr.DescribedComponentBatch] = [
            rr.components.Position3DBatch(self.centers).described(descriptor("centers", "rerun.components.Position3D")),
            rr.components.RotationQuatBatch(self.quaternions_xyzw).described(descriptor("quaternions", "rerun.components.RotationQuat")),
            rr.components.Vector2DBatch(self.scales).described(descriptor("scales", "rerun.components.Vector2D")),
            rr.components.OpacityBatch(self.opacities).described(descriptor("opacities", "rerun.components.Opacity")),
            untyped("sharpness", self.sharpness),
            untyped("color_field", self.color_field),
        ]
        if self.view_linear is not None:
            batches.append(untyped("view_linear", self.view_linear))
        if self.sh_coefficients is not None:
            batches.append(untyped("sh_coefficients", self.sh_coefficients))
        return batches

    @property
    def variant(self) -> HexelVariant:
        """V3 carries SH, V2 the linear view matrix, V1 neither."""
        return "v3" if self.sh_coefficients is not None else "v2" if self.view_linear is not None else "v1"

    def points_fallback(self) -> rr.Points3D:
        """Centres with clamped base colours as small screen-space points, for stock viewers.

        A world-space radius from the hexel scales turns large background hexels into spheres that cover the view.
        """
        colors: UInt8[ndarray, "n 3"] = np.round(np.clip(self.color_field[:, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
        return rr.Points3D(positions=self.centers, colors=colors, radii=rr.Radius.ui_points(FALLBACK_POINT_RADIUS))


def read_hexel_ply(path: Path) -> Hexels3D:
    """Read a 3D hexel PLY and check its header against the contract; 2D fits are rejected."""
    ply: PlyData = PlyData.read(str(path))
    comments: list[str] = list(ply.comments)
    if "hexels_format 1" not in comments:
        raise ValueError(f"{path}: not a hexel PLY (missing comment hexels_format 1)")
    variants: list[str] = [comment.removeprefix("hexels_variant ") for comment in comments if comment.startswith("hexels_variant ")]
    if len(variants) != 1:
        raise ValueError(f"{path}: expected one comment hexels_variant, found {variants}")
    if variants[0] not in get_args(HexelVariant):
        raise ValueError(f"{path}: hexels_variant {variants[0]} is not a 3D variant (v1, v2, v3)")
    variant: HexelVariant = cast(HexelVariant, variants[0])
    vertex: PlyElement = ply["vertex"]
    names: list[str] = [prop.name for prop in vertex.properties]
    expected: list[str] = property_names(variant)
    if names != expected or any(prop.val_dtype not in ("f4", "float32", "float") for prop in vertex.properties):
        raise ValueError(f"{path}: vertex properties do not match the {variant} contract order")
    rows: Float32[ndarray, "n k"] = np.stack([np.asarray(vertex.data[name], dtype=np.float32) for name in names], axis=1)
    if len(rows) == 0 or not np.isfinite(rows).all():
        raise ValueError(f"{path}: empty or non-finite hexel PLY")
    return Hexels3D(
        centers=rows[:, 0:3],
        quaternions_xyzw=np.ascontiguousarray(rows[:, [4, 5, 6, 3]]),
        scales=rows[:, 7:9],
        opacities=rows[:, 9],
        sharpness=rows[:, 10],
        color_field=rows[:, 11:20],
        view_linear=rows[:, 20:29] if variant == "v2" else None,
        sh_coefficients=rows[:, 20:65] if variant == "v3" else None,
    )


def write_hexel_ply(path: Path, hexels: Hexels3D, source: str = "gsplat-rust-renderer", step: int = 0) -> None:
    """Write a contract hexel PLY (the inverse of `read_hexel_ply`; the rotation goes back to wxyz)."""
    columns: list[Float32[ndarray, "n"] | Float32[ndarray, "n k"]] = [
        hexels.centers, hexels.quaternions_xyzw[:, [3, 0, 1, 2]], hexels.scales, hexels.opacities[:, None], hexels.sharpness[:, None], hexels.color_field,
    ]
    view: Float32[ndarray, "n k"] | None = hexels.sh_coefficients if hexels.sh_coefficients is not None else hexels.view_linear
    rows: Float32[ndarray, "n k"] = np.concatenate([*columns, *([view] if view is not None else [])], axis=1).astype(np.float32)
    names: list[str] = property_names(hexels.variant)
    data: ndarray = np.empty(len(rows), dtype=[(name, "<f4") for name in names])
    for index, name in enumerate(names):
        data[name] = rows[:, index]
    comments: list[str] = ["hexels_format 1", f"hexels_variant {hexels.variant}", f"hexels_source {source}", f"hexels_step {step}"]
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<", comments=comments).write(str(path))


def log_hexels(hexels: Hexels3D, entity: str = HEXELS_ENTITY, *, recording: rr.RecordingStream | None = None) -> Float32[ndarray, "2 3"]:
    """Log static `Hexels3D` plus its `Points3D` fallback; return the 2nd/98th percentile centre bounds."""
    rr.log(entity, hexels, hexels.points_fallback(), static=True, recording=recording)
    return np.percentile(hexels.centers, [2.0, 98.0], axis=0).astype(np.float32)
