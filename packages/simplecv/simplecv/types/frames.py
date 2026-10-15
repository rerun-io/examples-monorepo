"""Decoded multi-camera frames: what ``VideoSource`` and ``CatalogLumaSource`` put on their ``frames`` port (memory only, no serde)."""

from dataclasses import dataclass

from dagkit.time import TimeNs
from jaxtyping import Int64, UInt8
from numpy import ndarray
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MultiViewFrames:
    """One synchronized sample of every camera; a single camera is ``v == 1``."""

    t_ns: TimeNs
    frame_index: int
    """Sample index into the segment's video streams."""
    views: tuple[str, ...]
    """Camera names in the source's ``cameras`` order."""
    rgb: UInt8[Tensor, "v 3 h w"]
    """On the decode device (CUDA for ``nvdec_rgb``)."""
    cam_t_ns: Int64[Tensor, " v"]
    """Each camera's own sample time."""
    native: UInt8[Tensor, "v 3 hh ww"] | None = None
    """The decode at stored size when ``rgb`` is an area resize of it (``VideoSource`` with ``keep_native``), else None."""


@dataclass(frozen=True, slots=True)
class LumaFrameset:
    """Raw Y planes of one frameset (``CatalogLumaSource``); ``None`` marks a camera missing from it."""

    t_ns: TimeNs
    views: tuple[str, ...]
    luma: tuple[UInt8[ndarray, "h w"] | None, ...]
    cam_t_ns: tuple[int | None, ...]
