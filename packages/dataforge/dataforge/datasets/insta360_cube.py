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

``CubeJoin`` joins on one device (``cuda`` or ``cpu``): the lens pair is stacked into one image (lens 0, a zero row,
lens 1), so each face is ONE bilinear remap of it with a map that already picks the lens, no select; a mask face is ONE
gather from the two native-resolution lens masks, nearest as ``cv2.remap`` rounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import insta360_io
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor

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
    """Where each pixel of a face reads the lens it is read from."""

    map_x: Float32[ndarray, "n n"]
    """Lens x per face pixel, in the lens the pixel is read from."""
    map_y: Float32[ndarray, "n n"]
    """Lens y per face pixel."""
    use_lens0: Bool[ndarray, "n n"]
    """True where the pixel is read from lens 0 (its ray has ``z0 >= 0``)."""


def project_pair(
    cameras: list[insta360_io.MeiCamera], lens1_from_lens0: Float64[ndarray, "3 3"], lens_size: tuple[int, int], rays: Float64[ndarray, "n 3"]
) -> tuple[Float64[ndarray, "2 n 2"], Bool[ndarray, "2 n"]]:
    """Each ray (lens-0 axes) as a pixel of lens 0 and of lens 1, and whether that lens holds it: its model accepts the ray
    and its ``lens_size`` (width, height) image holds the pixel."""
    pixels0, valid0 = cameras[0].project(np.ascontiguousarray(rays))
    pixels1, valid1 = cameras[1].project(np.ascontiguousarray(rays @ lens1_from_lens0.T))
    pixels: Float64[ndarray, "2 n 2"] = np.stack([pixels0, pixels1])
    on_sensor: Bool[ndarray, "2 n"] = (pixels[..., 0] >= 0.0) & (pixels[..., 0] <= lens_size[0] - 1) & (pixels[..., 1] >= 0.0) & (pixels[..., 1] <= lens_size[1] - 1)
    return pixels, np.stack([valid0, valid1]) & on_sensor


def lens1_row(lens_height: int) -> int:
    """The stacked pair's first lens-1 row: lens 0 on ``[0, h)``, the zero row ``h``, lens 1 from ``h + 1``."""
    return lens_height + 1


def stack_pair(lens0: UInt8[Tensor, "b 3 h w"], lens1: UInt8[Tensor, "b 3 h w"]) -> Float32[Tensor, "b 3 r w"]:
    """The stacked pair on the lenses' device: lens 0 on rows ``[0, h)``, a zero row (the remap's border on both sides of
    the seam), lens 1 on ``[h + 1, 2h + 1)``."""
    b, c, h, w = lens0.shape
    stacked: Float32[Tensor, "b 3 r w"] = torch.empty((b, c, lens1_row(h) + h, w), dtype=torch.float32, device=lens0.device)
    stacked[:, :, :h] = lens0
    stacked[:, :, h : lens1_row(h)] = 0.0
    stacked[:, :, lens1_row(h) :] = lens1
    return stacked


def stack_pair_hwc(lens0: UInt8[Tensor, "b 3 h w"], lens1: UInt8[Tensor, "b 3 h w"]) -> UInt8[ndarray, "b r w 3"]:
    """``stack_pair`` on the host, channels last, as ``cv2.remap`` reads it."""
    b, c, h, w = lens0.shape
    stacked: UInt8[ndarray, "b r w 3"] = np.empty((b, lens1_row(h) + h, w, c), dtype=np.uint8)
    stacked[:, :h] = lens0.permute(0, 2, 3, 1).numpy()
    stacked[:, h : lens1_row(h)] = 0
    stacked[:, lens1_row(h) :] = lens1.permute(0, 2, 3, 1).numpy()
    return stacked


def stacked_grid(xy: Float32[ndarray, "*shape 2"] | Float64[ndarray, "*shape 2"], lens_size: tuple[int, int]) -> Float32[ndarray, "*shape 2"]:
    """Stacked-pair pixels as ``grid_sample`` coordinates (``align_corners=True``): x over ``[0, w - 1]``, y over ``[0, 2h]``."""
    width, height = lens_size
    return (xy.astype(np.float64) * [2.0 / (width - 1), 2.0 / (2 * height)] - 1.0).astype(np.float32)


def face_table(
    face: CubeFace, cameras: list[insta360_io.MeiCamera], lens1_from_lens0: Float64[ndarray, "3 3"], lens_size: tuple[int, int]
) -> FaceTable:
    """The table of ``face`` on the two lens cameras, whose images are ``lens_size`` (width, height) (built once per clip).

    Raises:
        ValueError: If a face pixel lies outside the lens it is read from.
    """
    rays: Float64[ndarray, "n 3"] = face.rays().reshape(-1, 3) @ face.cube_from_face.T
    shape: tuple[int, int] = (face.size, face.size)
    use_lens0: Bool[ndarray, " n"] = rays[:, 2] >= 0.0
    lens_pixels, held = project_pair(cameras, lens1_from_lens0, lens_size, rays)
    pixels: Float64[ndarray, "n 2"] = np.where(use_lens0[:, None], lens_pixels[0], lens_pixels[1])
    outside: int = int((~np.where(use_lens0, held[0], held[1])).sum())
    if outside:
        raise ValueError(f"{face.name}: {outside} pixels lie outside the lens they are read from")
    return FaceTable(
        map_x=pixels[:, 0].reshape(shape).astype(np.float32),
        map_y=pixels[:, 1].reshape(shape).astype(np.float32),
        use_lens0=use_lens0.reshape(shape),
    )


class CubeJoin:
    """The faces of lens pairs on one device: each face is one bilinear remap of the stacked pair.

    ``cuda``: one ``grid_sample`` for all faces. ``cpu``: one ``cv2.remap`` per face; torch's OpenMP workers spin after each
    op and starve the PyAV decode threads (clip B 60 s: 31 s, 22 s with ``OMP_WAIT_POLICY=PASSIVE``).
    """

    def __init__(self, tables: list[FaceTable], lens_size: tuple[int, int], device: torch.device) -> None:
        """``tables`` in face order; ``lens_size``: lens image ``(width, height)``."""
        height: int = lens_size[1]
        self.device: torch.device = device
        self.size: int = tables[0].map_x.shape[0]
        # Rows of the stacked pair (``stack_pair``).
        x: Float64[ndarray, "f n n"] = np.stack([table.map_x for table in tables]).astype(np.float64)
        y: Float64[ndarray, "f n n"] = np.stack([np.where(table.use_lens0, table.map_y, table.map_y.astype(np.float64) + lens1_row(height)) for table in tables])
        if device.type == "cpu":
            self.maps: Float32[ndarray, "f 2 n n"] = np.stack([x, y], axis=1).astype(np.float32)
            return
        grid: Float32[ndarray, "f n n 2"] = stacked_grid(np.stack([x, y], axis=-1), lens_size)
        self.grid: Float32[Tensor, "1 fn n 2"] = torch.from_numpy(grid.reshape(1, -1, self.size, 2)).to(device)

    def faces(self, lens0: UInt8[Tensor, "b 3 h w"], lens1: UInt8[Tensor, "b 3 h w"]) -> UInt8[Tensor, "f b 3 n n"]:
        """Every face of ``b`` lens pairs (RGB, channels first, on the device), bilinear like ``cv2.INTER_LINEAR``; face-major."""
        b, c = lens0.shape[:2]
        if self.device.type == "cpu":
            stacked: UInt8[ndarray, "b r w 3"] = stack_pair_hwc(lens0, lens1)
            faces: UInt8[ndarray, "f b n n 3"] = np.stack(
                [[cv2.remap(pair, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for pair in stacked] for map_x, map_y in self.maps]
            )
            return torch.from_numpy(np.ascontiguousarray(faces.transpose(0, 1, 4, 2, 3)))
        stacked_gpu: Float32[Tensor, "b 3 r w"] = stack_pair(lens0, lens1)
        joined: Float32[Tensor, "b 3 fn n"] = F.grid_sample(stacked_gpu, self.grid.expand(b, -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True)
        return joined.round_().to(torch.uint8).view(b, c, -1, self.size, self.size).permute(2, 0, 1, 3, 4).contiguous()


class MaskJoin:
    """The faces of lens mask pairs (label images): one nearest gather of all faces, on ``device``.

    The masks are read and written on the host; on ``cuda`` only the pair goes up and the faces come down (clip 007's
    3,203 pairs: the host gather took 6 s of the 14 s remap).
    """

    def __init__(self, tables: list[FaceTable], lens_size: tuple[int, int], device: torch.device | None = None) -> None:
        """``tables`` in face order; ``lens_size``: lens image ``(width, height)``; ``device`` default the host."""
        self.tables: list[FaceTable] = tables
        self.lens_size: tuple[int, int] = lens_size
        self.device: torch.device = device or torch.device("cpu")
        self.index: dict[tuple[tuple[int, int], tuple[int, int]], Int64[Tensor, "f n n"]] = {}

    def faces(self, mask0: UInt8[ndarray, "h0 w0"], mask1: UInt8[ndarray, "h1 w1"]) -> UInt8[ndarray, "f n n"]:
        """Every face of a lens mask pair at any resolution that covers the lens image (contract v2)."""
        key: tuple[tuple[int, int], tuple[int, int]] = (mask0.shape, mask1.shape)
        if key not in self.index:
            self.index[key] = torch.from_numpy(np.stack([self._index(table, key) for table in self.tables])).to(self.device)
        pair: UInt8[ndarray, " m"] = np.concatenate([mask0.ravel(), mask1.ravel(), np.zeros(1, dtype=np.uint8)])
        return torch.take(torch.from_numpy(pair).to(self.device), self.index[key]).cpu().numpy()

    def _index(self, table: FaceTable, mask_shapes: tuple[tuple[int, int], tuple[int, int]]) -> Int64[ndarray, "n n"]:
        """Where each face pixel reads ``[mask0, mask1, 0]`` flattened: ``cv2.remap``'s nearest lens pixel (rounded half to
        even), then the contract-v2 lookup into a ``hm x wm`` mask, ``(c, r) -> ((2r+1) hm // 2h, (2c+1) wm // 2w)``; a pixel
        off the lens reads 0."""
        width, height = self.lens_size
        (h0, w0), (h1, w1) = mask_shapes
        col: Int64[ndarray, "n n"] = np.rint(table.map_x).astype(np.int64)
        row: Int64[ndarray, "n n"] = np.rint(table.map_y).astype(np.int64)
        hm: Int64[ndarray, "n n"] = np.where(table.use_lens0, h0, h1)
        wm: Int64[ndarray, "n n"] = np.where(table.use_lens0, w0, w1)
        index: Int64[ndarray, "n n"] = np.where(table.use_lens0, 0, h0 * w0) + (2 * row + 1) * hm // (2 * height) * wm + (2 * col + 1) * wm // (2 * width)
        inside: Bool[ndarray, "n n"] = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        return np.where(inside, index, h0 * w0 + h1 * w1)
