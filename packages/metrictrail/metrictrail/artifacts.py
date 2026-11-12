"""Read the per-frame artifacts of a MetricTrail (ViPE ``pipeline=metrictrail``) output directory.

Upstream's formats (ViPE @ ead2e00, ``vipe/utils/io.py`` and ``vipe/metrictrail/pointcloud.py``), read here once at
the boundary; this package never writes them:

* ``intrinsics/<name>.npz``: ``data`` [N, 4] float ``fx, fy, cx, cy`` and ``inds`` (the artifact frame of each row);
* ``depth/<name>.zip``: one half-float OpenEXR per artifact frame (``<index>.exr``, channel ``Z``), metric Z-depth in
  metres, 0 where the depth provider had no depth;
* ``pointcloud/<name>.ply``: ``python -m vipe.metrictrail.pointcloud`` output, binary little-endian, ``x y z`` float32
  then ``red green blue`` uint8 per vertex.

The pose artifact and the run record are read by :mod:`metrictrail.eval.vipe_output`.
"""

import importlib
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray


def read_intrinsics(run_dir: Path, name: str, num_frames: int) -> Float64[ndarray, "n 4"]:
    """``fx, fy, cx, cy`` of every artifact frame (NaN where the artifact has no row).

    Raises:
        ValueError: If the archive misses ``data``/``inds`` or the rows are not 4-vectors.
    """
    path: Path = run_dir / "intrinsics" / f"{name}.npz"
    with np.load(path, allow_pickle=False) as archive:
        if "data" not in archive.files or "inds" not in archive.files:
            raise ValueError(f"{path}: expected data and inds, got {archive.files}")
        data: Float64[ndarray, "k 4"] = np.asarray(archive["data"], dtype=np.float64)
        inds: Int64[ndarray, "k"] = np.asarray(archive["inds"], dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != 4 or inds.shape != (data.shape[0],):
        raise ValueError(f"{path}: data {data.shape} / inds {inds.shape} are not N fx, fy, cx, cy rows")
    intrinsics: Float64[ndarray, "n 4"] = np.full((num_frames, 4), np.nan, dtype=np.float64)
    keep: ndarray = (inds >= 0) & (inds < num_frames)
    intrinsics[inds[keep]] = data[keep]
    return intrinsics


def depth_indices(path: Path) -> list[int]:
    """Artifact frame indices stored in a ViPE depth zip, ascending."""
    with zipfile.ZipFile(path, "r") as archive:
        return sorted(int(Path(member).stem) for member in archive.namelist() if member.endswith(".exr"))


def iter_depth_zip(path: Path, indices: list[int] | None = None) -> Iterator[tuple[int, Float32[ndarray, "h w"]]]:
    """Yield ``(artifact frame, metric depth in metres)`` from a ViPE depth zip, in index order.

    Args:
        path: ``depth/<name>.zip``.
        indices: Frames to decode; None decodes every frame.

    Raises:
        ValueError: If an EXR has no ``Z`` channel.
    """
    open_exr: ModuleType = importlib.import_module("OpenEXR")  # compiled, no stubs for pyrefly
    wanted: set[int] | None = set(indices) if indices is not None else None
    with zipfile.ZipFile(path, "r") as archive:
        members: list[tuple[int, str]] = sorted((int(Path(member).stem), member) for member in archive.namelist() if member.endswith(".exr"))
        for index, member in members:
            if wanted is not None and index not in wanted:
                continue
            with archive.open(member) as handle:
                exr = open_exr.InputFile(handle)
                window = exr.header()["dataWindow"]
                height: int = window.max.y - window.min.y + 1
                width: int = window.max.x - window.min.x + 1
                channels: list[bytes] = exr.channels(["Z"])
                if not channels:
                    raise ValueError(f"{path}:{member}: no Z channel")
                depth: Float32[ndarray, "h w"] = np.frombuffer(channels[0], dtype=np.float16).reshape(height, width).astype(np.float32)
            yield index, depth


@dataclass(frozen=True, slots=True)
class ColoredCloud:
    """A coloured point cloud, optionally with per-point unit normals."""

    points_xyz: Float32[ndarray, "n 3"]
    """Positions, metres, in the frame the producer states."""
    colors_rgb: UInt8[ndarray, "n 3"]
    """Per-point colour."""
    normals_xyz: Float32[ndarray, "n 3"] | None = None
    """Unit normals in the same frame as the positions (MoGe-2's normals of the fused frames), if the producer had them."""


PLY_VERTEX_DTYPE: np.dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
"""Vertex record of upstream's ``write_binary_point_cloud``."""


def read_binary_ply(path: Path) -> ColoredCloud:
    """Read upstream's binary little-endian PLY (``x y z`` float32, ``red green blue`` uint8).

    Raises:
        ValueError: If the header is not exactly that layout.
    """
    raw: bytes = path.read_bytes()
    marker: bytes = b"end_header\n"
    end: int = raw.find(marker)
    if end < 0:
        raise ValueError(f"{path}: no PLY header")
    header: list[str] = raw[:end].decode("ascii").splitlines()
    properties: list[str] = [line.split()[-1] for line in header if line.startswith("property")]
    if header[0] != "ply" or "format binary_little_endian 1.0" not in header or properties != list(PLY_VERTEX_DTYPE.names or ()):
        raise ValueError(f"{path}: expected a binary little-endian PLY with x y z red green blue, got {header}")
    count: int = next(int(line.split()[-1]) for line in header if line.startswith("element vertex"))
    vertices: ndarray = np.frombuffer(raw, dtype=PLY_VERTEX_DTYPE, count=count, offset=end + len(marker))
    points_xyz: Float32[ndarray, "n 3"] = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    colors_rgb: UInt8[ndarray, "n 3"] = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]])
    return ColoredCloud(points_xyz=points_xyz, colors_rgb=colors_rgb)


def subsample_cloud(cloud: ColoredCloud, max_points: int, seed: int = 0) -> ColoredCloud:
    """At most ``max_points`` points of ``cloud``, chosen uniformly at random (order kept)."""
    if cloud.points_xyz.shape[0] <= max_points:
        return cloud
    generator: np.random.Generator = np.random.default_rng(seed)
    keep: Int64[ndarray, "k"] = np.sort(generator.choice(cloud.points_xyz.shape[0], size=max_points, replace=False)).astype(np.int64)
    return ColoredCloud(
        points_xyz=cloud.points_xyz[keep],
        colors_rgb=cloud.colors_rgb[keep],
        normals_xyz=None if cloud.normals_xyz is None else cloud.normals_xyz[keep],
    )
