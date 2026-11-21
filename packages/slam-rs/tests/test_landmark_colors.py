"""Landmark colours sample the RGB copy at the keypoint's own pixel, axis by axis, and keep the first colour an id got."""

import numpy as np
from jaxtyping import Float32, Int64, UInt8, UInt32
from numpy import ndarray

from slam_rs.landmark_colors import LandmarkColors, pack_rgba, sample_rgb

TRACKED_WH: tuple[int, int] = (100, 67)
"""A frame whose height does not divide by the copy's factor: a quarter of it is 25 x 16 (67 / 4 floored)."""


def grid() -> UInt8[ndarray, "16 25 3"]:
    """Every pixel of the copy names itself: red is its column, green its row, blue a constant."""
    rows, columns = np.mgrid[0:16, 0:25]
    return np.stack([columns * 10, rows * 10, np.full_like(rows, 7)], axis=-1).astype(np.uint8)


def test_each_axis_maps_by_its_own_extent() -> None:
    """y = 58 of 67 is row 13 of 16 (the width ratio, 25/100, would say row 14); x = 99 is the last column."""
    positions: Float32[ndarray, "k 2"] = np.array([[0.0, 0.0], [58.0, 58.0], [99.0, 66.9], [-3.0, 80.0]], dtype=np.float32)
    rgb: UInt8[ndarray, "k 3"] = sample_rgb(positions, grid(), TRACKED_WH)
    columns: list[int] = (rgb[:, 0] // 10).tolist()
    rows: list[int] = (rgb[:, 1] // 10).tolist()
    assert columns == [0, 14, 24, 0]
    assert rows == [0, 13, 15, 15], "off the frame takes the nearest edge"


def test_an_id_keeps_the_colour_it_was_first_seen_with() -> None:
    colours: LandmarkColors = LandmarkColors()
    ids: Int64[ndarray, " k"] = np.array([3, 900], dtype=np.int64)
    colours.remember(ids, np.array([[0.0, 0.0], [99.0, 66.0]], dtype=np.float32), grid(), TRACKED_WH)
    # Later the same ids sit elsewhere, and a new id joins; only the new one takes a colour from this frame.
    colours.remember(np.array([900, 3, 5], dtype=np.int64), np.array([[0.0, 0.0], [99.0, 66.0], [50.0, 33.0]], dtype=np.float32), grid(), TRACKED_WH)
    unknown: tuple[int, int, int, int] = (1, 2, 3, 4)
    looked_up: UInt32[ndarray, " n"] = colours.lookup(np.array([3, 900, 5, 4, 10_000], dtype=np.int64), unknown)
    expected: UInt32[ndarray, " n"] = np.concatenate([
        pack_rgba(np.array([[0, 0, 7], [240, 150, 7], [120, 70, 7]], dtype=np.uint8)),
        pack_rgba(np.array([unknown[:3], unknown[:3]], dtype=np.uint8), unknown[3]),
    ])
    np.testing.assert_array_equal(looked_up, expected)


def test_pack_rgba_is_rerun_order() -> None:
    assert pack_rgba(np.array([[1, 2, 3]], dtype=np.uint8), 4).tolist() == [0x01020304]
