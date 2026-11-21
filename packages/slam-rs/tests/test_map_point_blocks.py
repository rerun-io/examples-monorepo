"""The map points' delta logging: only the keyframe blocks whose points changed are drawn again."""

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray

from slam_rs.vio_log import MapPointBlocks


def map_of(hosts: list[int], points_per_host: int = 3) -> tuple[Int64[ndarray, " p"], Int64[ndarray, " p"], Float32[ndarray, "p 3"]]:
    """Point ids, hosts and world positions of a map whose keyframe ``h`` hosts ``points_per_host`` points near x = h."""
    point_hosts: Int64[ndarray, " p"] = np.repeat(np.asarray(hosts, dtype=np.int64), points_per_host)
    point_ids: Int64[ndarray, " p"] = np.arange(point_hosts.shape[0], dtype=np.int64) + 100
    positions: Float32[ndarray, "p 3"] = np.stack(
        [point_hosts.astype(np.float32), np.zeros_like(point_hosts, dtype=np.float32), np.arange(point_hosts.shape[0], dtype=np.float32)], axis=1
    )
    return point_ids, point_hosts, positions


def test_the_first_draw_logs_every_block_and_an_unchanged_map_logs_none() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    ids, hosts, positions = map_of([10, 20, 30, 40, 50])
    first = blocks.changed(ids, hosts, positions)
    assert [block for block, _, _ in first] == [0, 1, 2]
    assert sum(len(block_ids) for _, block_ids, _ in first) == ids.shape[0]
    assert blocks.changed(ids, hosts, positions) == []


def test_only_the_block_whose_points_moved_past_the_tolerance_is_drawn_again() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2, tolerance_m=0.001)
    ids, hosts, positions = map_of([10, 20, 30, 40])
    blocks.changed(ids, hosts, positions)
    nudged = positions.copy()
    nudged[hosts == 30] += np.float32(0.0005)
    assert blocks.changed(ids, hosts, nudged) == [], "half a millimetre is under the tolerance"
    moved = positions.copy()
    moved[hosts == 30] += np.float32(0.01)
    redraw = blocks.changed(ids, hosts, moved)
    assert [block for block, _, _ in redraw] == [1]
    np.testing.assert_array_equal(redraw[0][2], moved[(hosts == 30) | (hosts == 40)])


def test_new_keyframes_fill_later_blocks_and_a_block_keeps_its_keyframes() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    ids, hosts, positions = map_of([10, 20, 30])
    blocks.changed(ids, hosts, positions)
    ids, hosts, positions = map_of([10, 20, 30, 40, 50])
    redraw = blocks.changed(ids, hosts, positions)
    # Block 1 held keyframe 30 alone and gains 40; block 2 starts with 50. Block 0 is untouched.
    assert [block for block, _, _ in redraw] == [1, 2]


def test_a_block_whose_points_vanished_is_returned_empty_and_a_point_set_change_redraws() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    ids, hosts, positions = map_of([10, 20, 30, 40])
    blocks.changed(ids, hosts, positions)
    keep = hosts < 30
    emptied = blocks.changed(ids[keep], hosts[keep], positions[keep])
    assert len(emptied) == 1 and emptied[0][0] == 1 and emptied[0][1].shape == (0,) and emptied[0][2].shape == (0, 3)
    culled = blocks.changed(ids[keep][1:], hosts[keep][1:], positions[keep][1:])
    assert [block for block, _, _ in culled] == [0], "one culled point changes its block's point set"


def test_an_empty_map_draws_nothing_and_emptying_the_map_clears_every_block() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.float32))
    assert blocks.changed(*empty) == []
    blocks.changed(*map_of([10, 20, 30]))
    assert [(block, ids.shape[0]) for block, ids, _ in blocks.changed(*empty)] == [(0, 0), (1, 0)]
