"""The cubemap of an Insta360 frameset: six ``N x N`` pinhole faces at ONE centre, joined from the two lens images.

Cube frame = lens 0's axes (RDF, ``+Z`` = lens 0's optical axis); every face sits at the rig centre, the midpoint of
the two lens centres. Faces (``fx = fy = N / 2``, principal point ``cx = cy = (N - 1) / 2``, pixel centres at integers):

- ``cube6_front0`` looks along ``+Z0`` (lens 0's axis), ``cube6_front1`` along ``-Z0`` (lens 1's axis within the
  factory's 0.1 deg), image down = ``+Y0`` on both;
- ``cube6_left`` / ``cube6_right`` look along ``-X0`` / ``+X0`` with image down ``+Y0``; ``cube6_up`` / ``cube6_down``
  along ``-Y0`` / ``+Y0`` with image right ``+X0``.

Each pixel is read from ONE lens: lens 0 where its ray has ``z0 >= 0``, lens 1 elsewhere, so on the four side faces
the seam is the face's centre line, with no overlap and no blend. Both lenses are read as if seen from the one centre
(only their rotations enter): a scene at infinity joins exactly, a near one shows the lens baseline's parallax at the
seam. Each lens supplies at most 90 deg off its axis, inside its 100 deg field, so no face pixel lies outside a lens
(``face_table`` checks it). A table maps each face pixel through ``insta360_io.MeiCamera.project`` (the Rust Mei model).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import insta360_io
import numpy as np
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray

ALIGN: int = 32
"""The face size is a multiple of this."""
REFERENCE_LENS: int = 1920
"""The lens size the default face size is stated for."""
REFERENCE_FACE: int = 800
"""The default face size at ``REFERENCE_LENS``."""

FACES: tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...] = (
    ("cube6_front0", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("cube6_front1", ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))),
    ("cube6_left", ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))),
    ("cube6_right", ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))),
    ("cube6_up", ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))),
    ("cube6_down", ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))),
)
"""Name and ``R_cube_face`` as matrix rows (its columns are the face's x, y, z in lens-0 axes), in camera order
``cam_02 .. cam_07``."""


def default_face_size(lens_size: int) -> int:
    """``N`` for a lens image ``lens_size`` pixels wide: ``lens_size * 800 / 1920`` to a multiple of 32 (1920 -> 800)."""
    return max(ALIGN, ALIGN * round(lens_size * REFERENCE_FACE / REFERENCE_LENS / ALIGN))


@dataclass(frozen=True, slots=True)
class CubeFace:
    """One face of the cubemap."""

    name: str
    """``cube6_<direction>``: the camera node's name."""
    cube_from_face: Float64[ndarray, "3 3"]
    """``R_cube_face`` = ``R_lens0_face``: columns are the face's x, y, z axes in lens-0 axes."""
    size: int
    """``N``: the face is ``N x N``."""

    @property
    def intrinsics(self) -> Float64[ndarray, "3 3"]:
        """``K``: ``fx = fy = N / 2``, centred principal point."""
        centre: float = (self.size - 1) / 2.0
        return np.array([[self.size / 2.0, 0.0, centre], [0.0, self.size / 2.0, centre], [0.0, 0.0, 1.0]])

    def rays(self) -> Float64[ndarray, "h w 3"]:
        """The ray of every pixel centre in the face frame, ``z = 1``."""
        u, v = np.meshgrid(np.arange(self.size, dtype=np.float64), np.arange(self.size, dtype=np.float64))
        k: Float64[ndarray, "3 3"] = self.intrinsics
        return np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], axis=-1)


def cube_faces(size: int) -> list[CubeFace]:
    """The six faces at size ``N``.

    Raises:
        ValueError: If ``size`` is not a positive multiple of ``ALIGN``.
    """
    if size <= 0 or size % ALIGN:
        raise ValueError(f"face size {size} is not a positive multiple of {ALIGN}")
    return [CubeFace(name=name, cube_from_face=np.array(rows), size=size) for name, rows in FACES]


def rig_centre(lens_from_rig: list[Float64[ndarray, "4 4"]]) -> Float64[ndarray, "3"]:
    """The midpoint of the two lens centres in the rig (IMU) frame, from each lens's ``T_lens_rig``."""
    centres: list[Float64[ndarray, "3"]] = [-pose[:3, :3].T @ pose[:3, 3] for pose in lens_from_rig]
    return (centres[0] + centres[1]) / 2.0


def face_from_rig(face: CubeFace, lens_from_rig: list[Float64[ndarray, "4 4"]]) -> Float64[ndarray, "4 4"]:
    """``T_face_rig``: rotation ``(R_rig_lens0 R_cube_face)^T``, centre at ``rig_centre``."""
    face_R_rig: Float64[ndarray, "3 3"] = (lens_from_rig[0][:3, :3].T @ face.cube_from_face).T
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3] = face_R_rig
    pose[:3, 3] = -face_R_rig @ rig_centre(lens_from_rig)
    return pose


@dataclass(frozen=True, slots=True)
class FaceTable:
    """Where each pixel of a face reads the two lens images."""

    map_x: Float32[ndarray, "2 h w"]
    """Lens x per face pixel, for lens 0 and lens 1 (``-1`` where that lens does not see the ray or its image ends)."""
    map_y: Float32[ndarray, "2 h w"]
    """Lens y per face pixel, for lens 0 and lens 1."""
    use_lens0: Bool[ndarray, "h w"]
    """True where the pixel is read from lens 0 (its ray has ``z0 >= 0``)."""


def face_table(
    face: CubeFace, cameras: list[insta360_io.MeiCamera], lens1_from_lens0: Float64[ndarray, "3 3"], lens_size: tuple[int, int]
) -> FaceTable:
    """The table of ``face`` on the two lens cameras, whose images are ``lens_size`` (width, height) (built once per clip).

    Raises:
        ValueError: If a face pixel lies outside the lens it is read from.
    """
    rays: Float64[ndarray, "n 3"] = face.rays().reshape(-1, 3) @ face.cube_from_face.T
    shape: tuple[int, int] = (face.size, face.size)
    use_lens0: Bool[ndarray, "h w"] = (rays[:, 2] >= 0.0).reshape(shape)
    maps: list[Float64[ndarray, "n 2"]] = []
    seen: list[Bool[ndarray, "h w"]] = []
    for camera, lens_rays in zip(cameras, (rays, rays @ lens1_from_lens0.T), strict=True):
        pixels, valid = camera.project(np.ascontiguousarray(lens_rays))
        # The model accepts the ray; the sensor must also hold the pixel it lands on.
        valid &= (pixels[:, 0] >= 0.0) & (pixels[:, 0] <= lens_size[0] - 1) & (pixels[:, 1] >= 0.0) & (pixels[:, 1] <= lens_size[1] - 1)
        maps.append(np.where(valid[:, None], pixels, -1.0))
        seen.append(valid.reshape(shape))
    outside: int = int((~np.where(use_lens0, seen[0], seen[1])).sum())
    if outside:
        raise ValueError(f"{face.name}: {outside} pixels lie outside the lens they are read from")
    return FaceTable(
        map_x=np.stack([m[:, 0].reshape(shape) for m in maps]).astype(np.float32),
        map_y=np.stack([m[:, 1].reshape(shape) for m in maps]).astype(np.float32),
        use_lens0=use_lens0,
    )


def join(images: tuple[UInt8[ndarray, "h w *c"], UInt8[ndarray, "h w *c"]], table: FaceTable, *, nearest: bool = False) -> UInt8[ndarray, "n n *c"]:
    """The face from the two lens images: bilinear for pictures, ``nearest`` for label images (masks)."""
    interpolation: int = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    lens0, lens1 = (
        cv2.remap(image, table.map_x[k], table.map_y[k], interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for k, image in enumerate(images)
    )
    pick: Bool[ndarray, "n n *c"] = table.use_lens0 if lens0.ndim == 2 else table.use_lens0[..., None]
    return np.where(pick, lens0, lens1)
