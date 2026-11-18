"""Cube faces of an Insta360 lens: virtual pinholes that share the lens's optical centre, and their remap tables.

Each lens gives five faces, all in its own frame (RDF, ``+Z`` = the optical axis, pixel centres at integers):

- ``front``: the ``N x N`` face of a 90 deg cube on the optical axis.
- ``right``, ``left``, ``down``, ``up``: the 90 deg cube face that looks along ``+X``, ``-X``, ``+Y`` and ``-Y`` of the
  lens, cropped to the half on the lens's side plus a strip past the 90 deg seam: ``side_size(N)`` pixels across the
  seam direction (``0.6 N``: tangents ``-1 .. +0.2`` from the lens side, about 11 deg past the seam; the lens half field
  is 100 deg), ``N`` along it. The focal stays the full face's, ``fx = fy = N / 2``, and the principal point stays at
  the full face's centre, so it lies off-centre in the crop.

Face axes. The right and left faces keep the lens's ``+Y`` as image down, the down and up faces keep the lens's ``+X``
as image right, so each face reads like the lens image turned a quarter about one axis. On the right and down faces
the lens axis is toward image left and top; on the left and up faces toward image right and bottom.

Because every face shares its lens's centre, ``T_lens_face = (R_lens_face, 0)`` is exact and a face's pose in the rig
is ``T_imu_face = T_imu_lens * (R_lens_face, 0)``. A remap table maps each face pixel to the lens pixel that sees the
same ray, through ``insta360_io.MeiCamera.project`` (the Rust Mei model); a ray that does not project maps to ``-1``,
which ``cv2.remap`` fills with black.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import insta360_io
import numpy as np
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray

DIRECTIONS: tuple[str, ...] = ("front", "left", "right", "up", "down")
"""The five faces of a lens, in logging order."""
SIDE_FRACTION: float = 0.6
"""Width of a side face across the seam direction, as a fraction of ``N``: tangents ``-1 .. +0.2``."""
ALIGN: int = 32
"""Every face side is a multiple of this (video encoders and SfM feature grids like it)."""
REFERENCE_LENS: int = 1920
"""The lens size the default face size is stated for."""
REFERENCE_FACE: int = 800
"""The default face size at ``REFERENCE_LENS``."""

# Columns are the face's x, y, z axes in lens coordinates (R_lens_face).
_AXES: dict[str, Float64[ndarray, "3 3"]] = {
    "front": np.eye(3),
    "right": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    "left": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "down": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
}


def align(value: float) -> int:
    """``value`` rounded to the nearest multiple of ``ALIGN``, at least ``ALIGN``."""
    return max(ALIGN, ALIGN * round(value / ALIGN))


def default_face_size(lens_size: int) -> int:
    """``N`` for a lens image ``lens_size`` pixels wide: ``lens_size * 800 / 1920`` to a multiple of 32 (1920 -> 800)."""
    return align(lens_size * REFERENCE_FACE / REFERENCE_LENS)


def side_size(face_size: int) -> int:
    """A side face's extent across its seam direction: ``0.6 N`` to a multiple of 32 (800 -> 480)."""
    return align(SIDE_FRACTION * face_size)


@dataclass(frozen=True, slots=True)
class CubeFace:
    """One virtual pinhole of one lens."""

    lens: int
    """Lens index, 0 or 1."""
    direction: str
    """One of ``DIRECTIONS``."""
    lens_from_face: Float64[ndarray, "3 3"]
    """``R_lens_face``: face vectors into the lens frame; columns are the face's x, y, z axes."""
    intrinsics: Float64[ndarray, "3 3"]
    """``K`` in the face's own (cropped) pixels, pixel centres at integers."""
    width: int
    """Pixels."""
    height: int
    """Pixels."""

    @property
    def name(self) -> str:
        """``cube_<lens>_<direction>``: the entity name under the rig."""
        return f"cube_{self.lens}_{self.direction}"

    def rays(self) -> Float64[ndarray, "h w 3"]:
        """The ray of every pixel centre in the face frame, ``z = 1``."""
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        k: Float64[ndarray, "3 3"] = self.intrinsics
        return np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], axis=-1)


def lens_faces(lens: int, face_size: int) -> list[CubeFace]:
    """The five faces of lens ``lens`` with front size ``face_size``.

    Raises:
        ValueError: If ``face_size`` is not a positive multiple of ``ALIGN``.
    """
    if face_size <= 0 or face_size % ALIGN:
        raise ValueError(f"face size {face_size} is not a positive multiple of {ALIGN}")
    n: int = face_size
    m: int = side_size(n)
    focal: float = n / 2.0
    centre: float = (n - 1) / 2.0
    faces: list[CubeFace] = []
    for direction in DIRECTIONS:
        width, height = {"front": (n, n), "right": (m, n), "left": (m, n), "down": (n, m), "up": (n, m)}[direction]
        # Right and down keep the full face's first columns / rows; left and up its last ones.
        cx: float = centre - (n - m if direction == "left" else 0)
        cy: float = centre - (n - m if direction == "up" else 0)
        intrinsics: Float64[ndarray, "3 3"] = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
        faces.append(CubeFace(lens=lens, direction=direction, lens_from_face=_AXES[direction].copy(), intrinsics=intrinsics, width=width, height=height))
    return faces


def face_from_rig(face: CubeFace, lens_from_rig: Float64[ndarray, "4 4"]) -> Float64[ndarray, "4 4"]:
    """``T_face_rig = (R_lens_face^T, 0) * T_lens_rig``: exact, the face shares the lens centre."""
    face_from_lens: Float64[ndarray, "4 4"] = np.eye(4)
    face_from_lens[:3, :3] = face.lens_from_face.T
    return face_from_lens @ lens_from_rig


@dataclass(frozen=True, slots=True)
class RemapTable:
    """Where each pixel of a face reads its lens image."""

    map_x: Float32[ndarray, "h w"]
    """Lens x per face pixel; ``-1`` where the ray does not project."""
    map_y: Float32[ndarray, "h w"]
    """Lens y per face pixel; ``-1`` where the ray does not project."""
    valid: Bool[ndarray, "h w"]
    """True where the ray projects into the lens field."""


def remap_table(face: CubeFace, camera: insta360_io.MeiCamera) -> RemapTable:
    """The table of ``face`` on the lens ``camera`` (built once per clip)."""
    rays: Float64[ndarray, "n 3"] = face.rays().reshape(-1, 3) @ face.lens_from_face.T
    pixels, valid = camera.project(np.ascontiguousarray(rays))
    pixels = np.where(valid[:, None], pixels, -1.0)
    shape: tuple[int, int] = (face.height, face.width)
    return RemapTable(
        map_x=pixels[:, 0].astype(np.float32).reshape(shape), map_y=pixels[:, 1].astype(np.float32).reshape(shape), valid=valid.reshape(shape)
    )


def remap(image: UInt8[ndarray, "h w c"], table: RemapTable) -> UInt8[ndarray, "fh fw c"]:
    """Bilinear resample of a lens image through ``table``; black outside the lens field."""
    return cv2.remap(image, table.map_x, table.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
