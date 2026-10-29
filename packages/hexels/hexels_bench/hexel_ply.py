"""Hexel PLY: the portable, render-ready Hexels model (contract `hexels-work/contracts/hexel-ply.md`).

Binary little-endian PLY, one float32 row per hexel, values already activated. The Rust reader in
gsplat-core consumes the same layout; this module is the Python writer and a reader for tests.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Float32
from numpy import ndarray

HexelVariant: TypeAlias = Literal["v1", "v2", "v3"]

FORMAT_VERSION: int = 1
BASE_PROPERTIES: tuple[str, ...] = (
    "x", "y", "z",
    "qw", "qx", "qy", "qz",
    "su", "sv",
    "opacity", "sharpness",
    "c0_r", "c0_g", "c0_b",
    "gu_r", "gu_g", "gu_b",
    "gv_r", "gv_g", "gv_b",
)  # fmt: skip
LINEAR_VIEW_PROPERTIES: tuple[str, ...] = tuple(f"a_{row}{col}" for row in range(3) for col in range(3))
SH_PROPERTIES: tuple[str, ...] = tuple(f"sh_{index}" for index in range(45))


def property_names(variant: HexelVariant) -> tuple[str, ...]:
    """PLY vertex properties of a variant, in file order."""
    if variant == "v2":
        return BASE_PROPERTIES + LINEAR_VIEW_PROPERTIES
    if variant == "v3":
        return BASE_PROPERTIES + SH_PROPERTIES
    return BASE_PROPERTIES


@dataclass(frozen=True, slots=True)
class HexelCloud:
    """Render-ready hexels: activated parameters, one row per hexel."""

    variant: HexelVariant
    """Appearance variant: v1 spatial color, v2 + linear view term, v3 + SH degree 1..3."""
    centers_xyz: Float32[ndarray, "n 3"]
    """Hexel centers p in world space."""
    quats_wxyz: Float32[ndarray, "n 4"]
    """Unit quaternions, w first; t_u = R e_x, t_v = R e_y, normal = R e_z."""
    scales_uv: Float32[ndarray, "n 2"]
    """Tangent scales (su, sv) in world units."""
    opacities: Float32[ndarray, "n"]
    """Opacity o in [0, 1]."""
    sharpness: Float32[ndarray, "n"]
    """Edge sharpness beta > 0 of the soft hexagon support."""
    base_rgb: Float32[ndarray, "n 3"]
    """Base color c0."""
    grad_u_rgb: Float32[ndarray, "n 3"]
    """Color gradient along u (per unit of su)."""
    grad_v_rgb: Float32[ndarray, "n 3"]
    """Color gradient along v (per unit of sv)."""
    view_linear: Float32[ndarray, "n 3 3"] | None = None
    """V2 only: A[i][j], i = RGB channel, j = component of the camera-to-center direction."""
    sh_coeffs: Float32[ndarray, "n 15 3"] | None = None
    """V3 only: SH degree 1..3 coefficients in upstream basis order, RGB innermost."""

    def __post_init__(self) -> None:
        count: int = self.centers_xyz.shape[0]
        for name in ("quats_wxyz", "scales_uv", "opacities", "sharpness", "base_rgb", "grad_u_rgb", "grad_v_rgb"):
            if getattr(self, name).shape[0] != count:
                raise ValueError(f"{name} has {getattr(self, name).shape[0]} rows, expected {count}")
        if (self.view_linear is not None) != (self.variant == "v2"):
            raise ValueError(f"view_linear must be set exactly for v2, variant is {self.variant}")
        if (self.sh_coeffs is not None) != (self.variant == "v3"):
            raise ValueError(f"sh_coeffs must be set exactly for v3, variant is {self.variant}")
        for extra in (self.view_linear, self.sh_coeffs):
            if extra is not None and extra.shape[0] != count:
                raise ValueError(f"view-dependent coefficients have {extra.shape[0]} rows, expected {count}")

    @property
    def count(self) -> int:
        return int(self.centers_xyz.shape[0])

    def to_rows(self) -> Float32[ndarray, "n k"]:
        """All properties as one float32 matrix in PLY column order."""
        columns: list[Float32[ndarray, "n _"]] = [
            self.centers_xyz,
            self.quats_wxyz,
            self.scales_uv,
            self.opacities[:, None],
            self.sharpness[:, None],
            self.base_rgb,
            self.grad_u_rgb,
            self.grad_v_rgb,
        ]
        if self.view_linear is not None:
            columns.append(self.view_linear.reshape(self.count, 9))
        if self.sh_coeffs is not None:
            columns.append(self.sh_coeffs.reshape(self.count, 45))
        return np.ascontiguousarray(np.concatenate(columns, axis=1), dtype=np.float32)

    @classmethod
    def from_rows(cls, variant: HexelVariant, rows: Float32[ndarray, "n k"]) -> "HexelCloud":
        """Inverse of `to_rows`."""
        expected: int = len(property_names(variant))
        if rows.shape[1] != expected:
            raise ValueError(f"{variant} needs {expected} columns, got {rows.shape[1]}")
        count: int = rows.shape[0]
        return cls(
            variant=variant,
            centers_xyz=rows[:, 0:3].copy(),
            quats_wxyz=rows[:, 3:7].copy(),
            scales_uv=rows[:, 7:9].copy(),
            opacities=rows[:, 9].copy(),
            sharpness=rows[:, 10].copy(),
            base_rgb=rows[:, 11:14].copy(),
            grad_u_rgb=rows[:, 14:17].copy(),
            grad_v_rgb=rows[:, 17:20].copy(),
            view_linear=rows[:, 20:29].reshape(count, 3, 3).copy() if variant == "v2" else None,
            sh_coeffs=rows[:, 20:65].reshape(count, 15, 3).copy() if variant == "v3" else None,
        )


@dataclass(frozen=True, slots=True)
class HexelPly:
    """A hexel PLY file: the hexels plus the provenance comments the contract requires."""

    cloud: HexelCloud
    """The hexels."""
    source: str
    """Checkpoint path or run id the hexels came from."""
    step: int
    """Training step of the checkpoint."""


def write_float_ply(path: Path, comments: list[str], names: tuple[str, ...], rows: Float32[ndarray, "n k"]) -> None:
    """Binary little-endian PLY of float32 `rows` under one `comment` line per entry (contract hexel-ply.md)."""
    header: list[str] = [
        "ply",
        "format binary_little_endian 1.0",
        *(f"comment {comment}" for comment in comments),
        f"element vertex {len(rows)}",
        *(f"property float {name}" for name in names),
        "end_header",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(("\n".join(header) + "\n").encode("ascii"))
        file.write(rows.astype("<f4", copy=False).tobytes())


def write_hexel_ply(path: Path, ply: HexelPly) -> None:
    """Write `ply` as a 3D hexel PLY."""
    if "\n" in ply.source:
        raise ValueError("source must be a single line")
    comments: list[str] = [f"hexels_format {FORMAT_VERSION}", f"hexels_variant {ply.cloud.variant}", f"hexels_source {ply.source}", f"hexels_step {ply.step}"]
    write_float_ply(path, comments, property_names(ply.cloud.variant), ply.cloud.to_rows())


def read_hexel_ply(path: Path) -> HexelPly:
    """Read a hexel PLY written by `write_hexel_ply` (strict: exact property list and order)."""
    data: bytes = path.read_bytes()
    marker: bytes = b"end_header\n"
    end: int = data.find(marker)
    if not data.startswith(b"ply\n") or end < 0:
        raise ValueError(f"{path}: not a PLY file")
    header: list[str] = data[:end].decode("ascii").splitlines()
    comments: dict[str, str] = {}
    properties: list[str] = []
    count: int = -1
    for line in header[1:]:
        fields: list[str] = line.split(" ", 2)
        if fields[0] == "format" and line != "format binary_little_endian 1.0":
            raise ValueError(f"{path}: unsupported format line {line!r}")
        if fields[0] == "comment" and len(fields) == 3:
            comments[fields[1]] = fields[2]
        elif fields[0] == "element":
            if fields[1] != "vertex":
                raise ValueError(f"{path}: unexpected element {fields[1]!r}")
            count = int(fields[2])
        elif fields[0] == "property":
            kind, name = fields[1], fields[2]
            if kind != "float":
                raise ValueError(f"{path}: property {name} is {kind}, expected float")
            properties.append(name)
    if comments.get("hexels_format") != str(FORMAT_VERSION):
        raise ValueError(f"{path}: hexels_format {comments.get('hexels_format')!r}, expected {FORMAT_VERSION}")
    variant_text: str = comments.get("hexels_variant", "")
    if variant_text not in ("v1", "v2", "v3"):
        raise ValueError(f"{path}: hexels_variant {variant_text!r}")
    variant: HexelVariant = variant_text
    if tuple(properties) != property_names(variant):
        raise ValueError(f"{path}: properties do not match variant {variant}")
    payload: bytes = data[end + len(marker) :]
    width: int = len(properties)
    if count < 0 or len(payload) != count * width * 4:
        raise ValueError(f"{path}: payload is {len(payload)} bytes, expected {count} x {width} float32")
    rows: Float32[ndarray, "n k"] = np.frombuffer(payload, dtype="<f4").reshape(count, width).astype(np.float32)
    return HexelPly(cloud=HexelCloud.from_rows(variant, rows), source=comments.get("hexels_source", ""), step=int(comments.get("hexels_step", "-1")))
