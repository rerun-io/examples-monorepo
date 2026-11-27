"""The tracked keypoints' colours: one stable, saturated hue per track id."""

import numpy as np
from jaxtyping import Int64, UInt8
from numpy import ndarray

from slam_rs.frontend_log import track_colors


def test_a_track_keeps_its_colour_and_neighbours_do_not_share_one() -> None:
    ids: Int64[ndarray, " n_tracks"] = np.arange(200, dtype=np.int64)
    colors: UInt8[ndarray, "n_tracks 3"] = track_colors(ids)
    assert colors.shape == (200, 3)
    assert colors.dtype == np.uint8
    # The colour is a pure function of the id, so a track that survives a frame
    # keeps it however the array is ordered.
    assert np.array_equal(track_colors(ids[::-1]), colors[::-1])
    # Hue only: every colour is saturated, so none of them is grey on grey imagery.
    assert np.all(colors.max(axis=1) == 255)
    assert np.all(colors.min(axis=1) == 0)
    # Neighbouring ids land far apart on the hue circle.
    assert np.all(np.abs(colors[1:].astype(np.int64) - colors[:-1].astype(np.int64)).sum(axis=1) > 30)


def test_an_empty_track_list_gives_an_empty_colour_list() -> None:
    assert track_colors(np.zeros(0, dtype=np.int64)).shape == (0, 3)
