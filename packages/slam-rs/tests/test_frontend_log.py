"""What the frontend's Rerun layer promises, without a viewer: colours and the C++ dumps."""

from pathlib import Path

import numpy as np
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray

from slam_rs.frontend_log import DEFAULT_DUMPS_DIR, read_cpp_dumps, track_colors


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


def test_the_committed_cpp_dumps_are_read_on_the_feeds_own_clock() -> None:
    """The dumps key on ``video_time``, so a frameset finds its own without associating."""
    dumps: dict[int, list[Float32[ndarray, "n_keypoints 2"]]] = read_cpp_dumps(DEFAULT_DUMPS_DIR)
    assert len(dumps) == 8
    assert 0 in dumps and 18507000 in dumps, sorted(dumps)
    first: list[Float32[ndarray, "n_keypoints 2"]] = dumps[0]
    assert len(first) == 2, "the smoke segment is a two-camera rig"
    assert first[0].shape == (66, 2), "what dump_flow.cpp recorded for camera 0 of frameset 0"
    assert first[0].dtype == np.float32
    assert np.all((first[0] >= 0.0) & (first[0] < 960.0)), "keypoints lie inside the 960x960 frame"


def test_a_directory_without_dumps_leaves_the_overlay_empty(tmp_path: Path) -> None:
    assert read_cpp_dumps(tmp_path) == {}
    assert read_cpp_dumps(tmp_path / "does-not-exist") == {}
