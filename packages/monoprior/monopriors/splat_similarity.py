"""Move a trained 3D Gaussian splat (INRIA ``.ply``) by a similarity ``p' = s R p + t``.

3DGS is equivariant under a similarity, so a splat trained in one frame can be moved instead of retrained:
positions ``s R p + t``, rotations ``R q``, log-scales ``+ ln s``, opacity and the SH DC term unchanged, normals
rotated, and the view-dependent SH bands 1-3 rotated so that the colour seen along ``R d`` after the move equals
the colour seen along ``d`` before. The band rotation matrices are fitted by least squares on the 3DGS real-SH
basis (the same constants, signs and coefficient order as the renderer), which is exact because each band is
closed under rotation. Every other PLY property and the property order are kept.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

SH_C1: float = 0.4886025119029199
SH_C2: tuple[float, ...] = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3: tuple[float, ...] = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)
BAND_SLICES: tuple[slice, ...] = (slice(0, 3), slice(3, 8), slice(8, 15))
"""Bands 1, 2, 3 inside the 15 rest coefficients of one colour channel."""

PLY_DTYPES: dict[str, str] = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "int": "i4", "int32": "i4", "uint": "u4"}


@dataclass(frozen=True, slots=True)
class Similarity:
    """``p' = scale * rotation @ p + translation``."""

    scale: float
    """Positive uniform scale."""
    rotation: Float64[ndarray, "3 3"]
    """Proper rotation matrix."""
    translation: Float64[ndarray, "3"]
    """Translation applied after rotation and scale."""

    def __post_init__(self) -> None:
        """Validate the scale and the rotation."""
        if not self.scale > 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-6) or np.linalg.det(self.rotation) < 0.0:
            raise ValueError("rotation must be a proper rotation matrix")

    def inverse(self) -> "Similarity":
        """The similarity that undoes this one."""
        rotation: Float64[ndarray, "3 3"] = self.rotation.T
        return Similarity(scale=1.0 / self.scale, rotation=rotation, translation=-(rotation @ self.translation) / self.scale)

    def apply(self, points: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
        """Move points."""
        return self.scale * points @ self.rotation.T + self.translation


def sh_basis_rest(directions: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 15"]:
    """The 3DGS real-SH basis of bands 1-3 for unit directions, in the renderer's coefficient order and signs."""
    x: Float64[ndarray, " n"] = directions[:, 0]
    y: Float64[ndarray, " n"] = directions[:, 1]
    z: Float64[ndarray, " n"] = directions[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    return np.stack(
        [
            -SH_C1 * y,
            SH_C1 * z,
            -SH_C1 * x,
            SH_C2[0] * x * y,
            SH_C2[1] * y * z,
            SH_C2[2] * (2.0 * zz - xx - yy),
            SH_C2[3] * x * z,
            SH_C2[4] * (xx - yy),
            SH_C3[0] * y * (3.0 * xx - yy),
            SH_C3[1] * x * y * z,
            SH_C3[2] * y * (4.0 * zz - xx - yy),
            SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            SH_C3[4] * x * (4.0 * zz - xx - yy),
            SH_C3[5] * z * (xx - yy),
            SH_C3[6] * x * (xx - 3.0 * yy),
        ],
        axis=-1,
    )


def sh_rest_rotation(rotation: Float64[ndarray, "3 3"]) -> Float64[ndarray, "15 15"]:
    """Block-diagonal matrix ``D`` with ``basis(R d) @ (D c) == basis(d) @ c`` for every direction ``d``."""
    rng: np.random.Generator = np.random.default_rng(0)
    directions: Float64[ndarray, "n 3"] = rng.normal(size=(512, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    before: Float64[ndarray, "n 15"] = sh_basis_rest(directions)
    after: Float64[ndarray, "n 15"] = sh_basis_rest(directions @ rotation.T)
    matrix: Float64[ndarray, "15 15"] = np.zeros((15, 15))
    for band in BAND_SLICES:
        matrix[band, band] = np.linalg.lstsq(after[:, band], before[:, band], rcond=None)[0]
    return matrix


def read_ply(path: Path) -> tuple[bytes, np.ndarray]:
    """Header bytes and the vertex table (structured array) of a binary little-endian, vertex-only PLY.

    Raises:
        ValueError: For another format or more than one element.
    """
    raw: bytes = path.read_bytes()
    end: int = raw.index(b"end_header\n") + len(b"end_header\n")
    header: bytes = raw[:end]
    lines: list[str] = header.decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"{path}: only binary_little_endian PLY is supported")
    elements: list[str] = [line for line in lines if line.startswith("element ")]
    if len(elements) != 1 or not elements[0].startswith("element vertex "):
        raise ValueError(f"{path}: expected one vertex element, found {elements}")
    count: int = int(elements[0].split()[2])
    fields: list[tuple[str, str]] = [(line.split()[2], "<" + PLY_DTYPES[line.split()[1]]) for line in lines if line.startswith("property ")]
    table: np.ndarray = np.frombuffer(raw, dtype=np.dtype(fields), count=count, offset=end)
    return header, table


def transform_gaussians(table: np.ndarray, similarity: Similarity) -> np.ndarray:
    """A moved copy of an INRIA Gaussian table (structured array with x..z, rot_0..3 = w x y z, scale_0..2, f_rest_*)."""
    names: tuple[str, ...] = table.dtype.names or ()
    out: np.ndarray = table.copy()
    positions: Float64[ndarray, "n 3"] = np.stack([table[name] for name in ("x", "y", "z")], axis=1).astype(np.float64)
    moved: Float64[ndarray, "n 3"] = similarity.apply(positions)
    for index, name in enumerate(("x", "y", "z")):
        out[name] = moved[:, index]
    if {"nx", "ny", "nz"} <= set(names):
        normals: Float64[ndarray, "n 3"] = np.stack([table[name] for name in ("nx", "ny", "nz")], axis=1).astype(np.float64) @ similarity.rotation.T
        for index, name in enumerate(("nx", "ny", "nz")):
            out[name] = normals[:, index]
    quat_wxyz: Float64[ndarray, "n 4"] = np.stack([table[f"rot_{index}"] for index in range(4)], axis=1).astype(np.float64)
    quat_wxyz /= np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    rotated: Float64[ndarray, "n 4"] = (Rotation.from_matrix(similarity.rotation) * Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])).as_quat()
    for index, column in enumerate((3, 0, 1, 2)):
        out[f"rot_{index}"] = rotated[:, column]
    for index in range(3):
        out[f"scale_{index}"] = table[f"scale_{index}"].astype(np.float64) + np.log(similarity.scale)
    rest_names: list[str] = sorted((name for name in names if name.startswith("f_rest_")), key=lambda name: int(name.removeprefix("f_rest_")))
    if rest_names:
        per_channel: int = len(rest_names) // 3
        if per_channel not in (3, 8, 15) or len(rest_names) != 3 * per_channel:
            raise ValueError(f"{len(rest_names)} f_rest properties: expected SH degree 1-3 (9, 24 or 45)")
        rest: Float64[ndarray, "n 3 k"] = np.stack([table[name] for name in rest_names], axis=1).astype(np.float64).reshape(-1, 3, per_channel)
        matrix: Float64[ndarray, "k k"] = sh_rest_rotation(similarity.rotation)[:per_channel, :per_channel]
        rotated_rest: Float64[ndarray, "n m"] = (rest @ matrix.T).reshape(-1, 3 * per_channel)
        for index, name in enumerate(rest_names):
            out[name] = rotated_rest[:, index]
    return out


def transform_ply(source: Path, destination: Path, similarity: Similarity) -> int:
    """Write ``source`` moved by ``similarity`` to ``destination`` with the identical header; returns the splat count."""
    header, table = read_ply(source)
    moved: np.ndarray = transform_gaussians(table, similarity)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        file.write(header)
        file.write(moved.tobytes())
    return len(moved)


def gaussian_covariances(table: np.ndarray) -> Float64[ndarray, "n 3 3"]:
    """World covariances ``R_q S S R_q^T`` of an INRIA Gaussian table (log-scales, w x y z quaternions)."""
    quat_wxyz: Float64[ndarray, "n 4"] = np.stack([table[f"rot_{index}"] for index in range(4)], axis=1).astype(np.float64)
    rotations: Float64[ndarray, "n 3 3"] = Rotation.from_quat(
        quat_wxyz[:, [1, 2, 3, 0]] / np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    ).as_matrix()
    scales: Float64[ndarray, "n 3"] = np.exp(np.stack([table[f"scale_{index}"] for index in range(3)], axis=1).astype(np.float64))
    return np.einsum("nij,nj,nkj->nik", rotations, scales**2, rotations)


def sh_colour(table: np.ndarray, directions: Float64[ndarray, "n 3"]) -> Float32[ndarray, "n 3"]:
    """View-dependent colour term (DC + rest) of each Gaussian along its direction, before the 3DGS +0.5 offset."""
    names: tuple[str, ...] = table.dtype.names or ()
    rest_names: list[str] = sorted((name for name in names if name.startswith("f_rest_")), key=lambda name: int(name.removeprefix("f_rest_")))
    per_channel: int = len(rest_names) // 3
    rest: Float64[ndarray, "n 3 k"] = np.stack([table[name] for name in rest_names], axis=1).astype(np.float64).reshape(-1, 3, per_channel)
    dc: Float64[ndarray, "n 3"] = np.stack([table[f"f_dc_{index}"] for index in range(3)], axis=1).astype(np.float64) * 0.28209479177387814
    basis: Float64[ndarray, "n k"] = sh_basis_rest(directions / np.linalg.norm(directions, axis=1, keepdims=True))[:, :per_channel]
    return (dc + np.einsum("nck,nk->nc", rest, basis)).astype(np.float32)
