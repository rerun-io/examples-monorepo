"""Read the per-frame artifacts of a MetricTrail (ViPE ``pipeline=metrictrail``) output directory.

Upstream's formats (ViPE @ ead2e00, ``vipe/utils/io.py``), read here once at
the boundary; this package never writes them:

* ``intrinsics/<name>.npz``: ``data`` [N, 4] float ``fx, fy, cx, cy`` and ``inds`` (the artifact frame of each row);
* ``depth/<name>.zip``: one half-float OpenEXR per artifact frame (``<index>.exr``, channel ``Z``), metric Z-depth in
  metres, 0 where the depth provider had no depth.

The pose artifact and the run record are read by :mod:`metrictrail.eval.vipe_output`.
"""

import importlib
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
from jaxtyping import Float32, Float64, Int64
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
