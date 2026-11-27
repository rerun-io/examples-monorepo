"""The framesets of a dataforge Insta360 cube layer, which the mask layers must cover.

dataforge writes the cube layer (``--cube``) for every ``cube_stride``-th frameset from frameset 0 and records the
sampling in the ``cube`` recording property (``stride``, ``framesets``). That property is the one source of the stride:
the maskers follow it when the recording has a cube layer, so their masks land on every face stamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rerun as rr
from rerun.catalog import DatasetEntry
from simplecv.catalog_layers import segment_properties

DEFAULT_STRIDE: int = 10
"""Frameset stride without a cube layer: dataforge's default ``cube_stride`` (10 at 30 fps = 3 Hz)."""


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
        ValueError: If ``dataset`` has no segment ``segment``, or its stride or frameset count is not an integer.
    """
    cube: dict[str, list[object]] = segment_properties(dataset, segment, "cube")
    if "stride" not in cube or "framesets" not in cube:
        return None
    stride, framesets = cube["stride"][0], cube["framesets"][0]
    if not isinstance(stride, int) or not isinstance(framesets, int):
        raise ValueError(f"segment {segment!r}: the cube property's stride {stride!r} and framesets {framesets!r} are not integers")
    return CubeSampling(stride, framesets)


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


def cube_sampling(dataset: DatasetEntry, segment: str, cube_rrd: Path | None) -> CubeSampling | None:
    """The cube sampling of ``segment``: from ``cube_rrd`` when given, else from the segment's ``cube`` property."""
    return read_cube_sampling(cube_rrd, segment) if cube_rrd is not None else segment_cube_sampling(dataset, segment)


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


__all__ = ("DEFAULT_STRIDE", "CubeSampling", "cube_sampling", "read_cube_sampling", "sampled_framesets", "segment_cube_sampling")
