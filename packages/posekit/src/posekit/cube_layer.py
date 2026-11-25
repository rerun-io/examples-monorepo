"""The framesets of a dataforge Insta360 cube layer, which the mask layers must cover.

dataforge writes the cube layer (``--cube``) for every ``cube_stride``-th frameset from frameset 0 and records the
sampling in the ``cube`` recording property (``stride``, ``framesets``). That property is the one source of the stride:
the maskers follow it when the recording has a cube layer, so their masks land on every face stamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import rerun as rr
from rerun.catalog import DatasetEntry

DEFAULT_STRIDE: int = 10
"""Frameset stride without a cube layer: dataforge's default ``cube_stride`` (10 at 30 fps = 3 Hz)."""
STRIDE_COLUMN: str = "property:cube:stride"
"""Segment-table column of the cube layer's stride."""
FRAMESETS_COLUMN: str = "property:cube:framesets"
"""Segment-table column of the number of framesets in the cube layer."""


@dataclass(frozen=True, slots=True)
class CubeSampling:
    """The framesets of a cube layer: ``0, stride, 2 stride, ...``, ``framesets`` of them."""

    stride: int
    """Framesets between two cube framesets."""
    framesets: int
    """Cube framesets in the layer."""


def segment_cube_sampling(dataset: DatasetEntry, segment: str) -> CubeSampling | None:
    """The cube sampling of ``segment`` from its ``cube`` property (``None`` without a cube layer).

    Raises:
        ValueError: If ``dataset`` has no segment ``segment``.
    """
    table: pa.Table = pa.Table.from_batches(dataset.segment_table().collect())
    rows: list[dict[str, object]] = [row for row in table.to_pylist() if row["rerun_segment_id"] == segment]
    if not rows:
        raise ValueError(f"no segment {segment!r} in dataset {dataset.name!r}")
    stride, framesets = rows[0].get(STRIDE_COLUMN), rows[0].get(FRAMESETS_COLUMN)
    if not isinstance(stride, list) or not isinstance(framesets, list) or not stride or not framesets:
        return None
    return CubeSampling(int(stride[0]), int(framesets[0]))


def read_cube_sampling(rrd: Path, segment: str) -> CubeSampling | None:
    """The cube sampling recorded in a cube-layer file of ``segment`` (``None`` without a ``cube`` property).

    Raises:
        ValueError: If ``rrd`` is not a recording of ``segment``.
    """
    with rr.server.Server(datasets={"cube": [str(rrd)]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("cube")
        segments: list[str] = list(dataset.segment_ids())
        if segments != [segment]:
            raise ValueError(f"{rrd} holds recordings {segments}; expected the cube layer of {segment!r}")
        return segment_cube_sampling(dataset, segment)


def sampled_framesets(option: str, stride: int | None, cube: CubeSampling | None, start: int, end: int) -> range:
    """The framesets a masker samples in ``[start, end)``: the cube layer's when there is one, else every ``stride``-th.

    Args:
        option: The CLI option that sets ``stride``, for the error message.
        stride: The requested stride; ``None`` follows the cube layer (:data:`DEFAULT_STRIDE` without one).
        cube: The recording's cube sampling, if it has a cube layer.
        start: First frameset index.
        end: Frameset index after the last.

    Raises:
        ValueError: If ``stride`` disagrees with the cube layer's, or ``start`` is not a cube frameset.
    """
    if cube is None:
        return range(start, end, DEFAULT_STRIDE if stride is None else stride)
    if stride is not None and stride != cube.stride:
        raise ValueError(
            f"{option} {stride} disagrees with the cube layer's stride {cube.stride} (recording property cube:stride); "
            f"omit {option} to follow the cube layer"
        )
    if start % cube.stride:
        raise ValueError(f"start {start} is not a cube frameset (a multiple of the cube stride {cube.stride})")
    return range(start, min(end, cube.stride * cube.framesets), cube.stride)


__all__ = ("DEFAULT_STRIDE", "CubeSampling", "read_cube_sampling", "sampled_framesets", "segment_cube_sampling")
