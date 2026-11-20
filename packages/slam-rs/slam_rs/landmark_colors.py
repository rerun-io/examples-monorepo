"""Landmark colours: the image's colour under a keypoint where its track was first seen, kept per landmark id.

A landmark's id is its keypoint's (``LandmarkId::from(KeypointId)``), so the frontend's tracks name the pixel. The
colour comes from a small RGB copy of the frame (:attr:`slam_rs.catalog_feed.Frameset.colors`), area-resampled from the
decoded frame to :func:`slam_rs.catalog_feed.color_copy_size` on both decoders; each axis of a tracked position maps
onto it by its own extent ratio.
"""

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Bool, Float32, Int64, UInt8, UInt32
from numpy import ndarray


def pack_rgba(rgb: UInt8[ndarray, "n 3"], alpha: int = 255) -> UInt32[ndarray, " n"]:
    """Colours as Rerun stores them: one ``0xRRGGBBAA`` integer each.

    Args:
        rgb: One colour per row.
        alpha: The alpha every colour gets.

    Returns:
        The packed colours.
    """
    wide: UInt32[ndarray, "n 3"] = rgb.astype(np.uint32)
    return (wide[:, 0] << 24) | (wide[:, 1] << 16) | (wide[:, 2] << 8) | np.uint32(alpha)


def sample_rgb(positions: Float32[ndarray, "k 2"], color: UInt8[ndarray, "ch cw 3"], tracked_wh: tuple[int, int]) -> UInt8[ndarray, "k 3"]:
    """The colour copy's pixel under each tracked keypoint.

    Args:
        positions: Keypoints in the pixels of the tracked frame, ``(x, y)``.
        color: The frame's RGB copy.
        tracked_wh: ``(width, height)`` of the tracked frame, which the copy covers whole.

    Returns:
        One colour per keypoint; positions off the frame take the nearest edge pixel.
    """
    height, width = color.shape[:2]
    u: Int64[ndarray, " k"] = np.clip(np.floor(positions[:, 0] * (width / tracked_wh[0])).astype(np.int64), 0, width - 1)
    v: Int64[ndarray, " k"] = np.clip(np.floor(positions[:, 1] * (height / tracked_wh[1])).astype(np.int64), 0, height - 1)
    return color[v, u]


@dataclass(slots=True)
class LandmarkColors:
    """Packed RGBA per landmark id, indexed by the id; 0 means not seen yet (every stored colour has alpha 255)."""

    rgba: UInt32[ndarray, " n"] = field(default_factory=lambda: np.zeros(0, dtype=np.uint32))
    """The table; it grows to the largest id seen, and ids come from one sequential counter."""

    def remember(self, ids: Int64[ndarray, " k"], positions: Float32[ndarray, "k 2"], color: UInt8[ndarray, "ch cw 3"], tracked_wh: tuple[int, int]) -> None:
        """Give each id that has no colour yet the copy's colour under its keypoint; an id keeps its first colour.

        Args:
            ids: Track ids in one camera.
            positions: Their keypoints in that camera's tracked pixels.
            color: That camera's RGB copy.
            tracked_wh: ``(width, height)`` of that camera's tracked frame.
        """
        if len(ids) == 0:
            return
        if (needed := int(ids.max()) + 1) > len(self.rgba):
            grown: UInt32[ndarray, " m"] = np.zeros(max(needed, 2 * len(self.rgba)), dtype=np.uint32)
            grown[: len(self.rgba)] = self.rgba
            self.rgba = grown
        new: Bool[ndarray, " k"] = self.rgba[ids] == 0
        if new.any():
            self.rgba[ids[new]] = pack_rgba(sample_rgb(positions[new], color, tracked_wh))

    def lookup(self, ids: Int64[ndarray, " n"], unknown: tuple[int, int, int, int]) -> UInt32[ndarray, " n"]:
        """Packed RGBA per id: its remembered colour, else ``unknown``.

        Args:
            ids: Landmark ids.
            unknown: RGBA for an id with no colour.

        Returns:
            One packed colour per id.
        """
        known: UInt32[ndarray, " n"] = np.zeros(len(ids), dtype=np.uint32)
        inside: Bool[ndarray, " n"] = ids < len(self.rgba)
        known[inside] = self.rgba[ids[inside]]
        fallback: UInt32[ndarray, " 1"] = pack_rgba(np.array([unknown[:3]], dtype=np.uint8), unknown[3])
        return np.where(known == 0, fallback, known)
