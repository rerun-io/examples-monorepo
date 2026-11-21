"""The map points' delta logging: the map's changes go into per-host points, and only the keyframe blocks they touched
are drawn again."""

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray

from slam_rs.vio_log import MapPointBlocks

Points = tuple[Int64[ndarray, " p"], Int64[ndarray, " p"], Float32[ndarray, "p 3"]]


def points_of(hosts: list[int], points_per_host: int = 3) -> Points:
    """Point ids, hosts and world positions: keyframe ``h`` hosts ``points_per_host`` points near x = h, ids from 100 + 3h."""
    point_hosts: Int64[ndarray, " p"] = np.repeat(np.asarray(hosts, dtype=np.int64), points_per_host)
    point_ids: Int64[ndarray, " p"] = np.asarray([100 + points_per_host * h + k for h in hosts for k in range(points_per_host)], dtype=np.int64)
    positions: Float32[ndarray, "p 3"] = np.stack(
        [point_hosts.astype(np.float32), np.zeros_like(point_hosts, dtype=np.float32), (point_ids % 7).astype(np.float32)], axis=1
    )
    return point_ids, point_hosts, positions


def apply(blocks: MapPointBlocks, keyframes: list[int], changed: list[int], points: Points) -> None:
    """Feed one map delta: the map's keyframes, the hosts a write touched, and those hosts' points."""
    blocks.apply(np.asarray(keyframes, dtype=np.int64), np.asarray(changed, dtype=np.int64), *points)


def test_a_first_delta_draws_every_block_and_a_quiet_one_none() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    apply(blocks, [10, 20, 30, 40, 50], [10, 20, 30, 40, 50], points_of([10, 20, 30, 40, 50]))
    first = blocks.redraw()
    assert [block for block, _, _ in first] == [0, 1, 2]
    assert sum(len(ids) for _, ids, _ in first) == 15
    apply(blocks, [10, 20, 30, 40, 50], [], points_of([]))
    assert blocks.redraw() == []


def test_only_the_block_whose_points_moved_past_the_tolerance_is_drawn_again() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2, tolerance_m=0.001)
    apply(blocks, [10, 20, 30, 40], [10, 20, 30, 40], points_of([10, 20, 30, 40]))
    blocks.redraw()
    ids, hosts, positions = points_of([30])
    apply(blocks, [10, 20, 30, 40], [30], (ids, hosts, positions + np.float32(0.0005)))
    assert blocks.redraw() == [], "half a millimetre is under the tolerance"
    apply(blocks, [10, 20, 30, 40], [30], (ids, hosts, positions + np.float32(0.01)))
    redraw = blocks.redraw()
    assert [block for block, _, _ in redraw] == [1]
    expected_ids, _, expected = points_of([30, 40])
    expected[:3] += np.float32(0.01)
    np.testing.assert_array_equal(redraw[0][1], expected_ids)
    np.testing.assert_array_equal(redraw[0][2], expected)


def test_changes_between_draws_add_up_and_new_keyframes_fill_later_blocks() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    apply(blocks, [10, 20, 30], [10, 20, 30], points_of([10, 20, 30]))
    blocks.redraw()
    # Two reads before the next draw: keyframe 40 joins block 1 (beside 30), then 50 opens block 2. Block 0 is untouched.
    apply(blocks, [10, 20, 30, 40], [40], points_of([40]))
    apply(blocks, [10, 20, 30, 40, 50], [50], points_of([50]))
    assert [block for block, _, _ in blocks.redraw()] == [1, 2]


def test_a_removed_keyframe_empties_its_points_and_a_block_left_with_none_comes_back_empty() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    apply(blocks, [10, 20, 30, 40], [10, 20, 30, 40], points_of([10, 20, 30, 40]))
    blocks.redraw()
    # Keyframe 40 is culled: block 1 keeps 30's points.
    apply(blocks, [10, 20, 30], [40], points_of([]))
    redraw = blocks.redraw()
    assert [block for block, _, _ in redraw] == [1] and len(redraw[0][1]) == 3
    # Then 30 goes too: block 1 has no points left.
    apply(blocks, [10, 20], [30], points_of([]))
    emptied = blocks.redraw()
    assert len(emptied) == 1 and emptied[0][0] == 1 and emptied[0][1].shape == (0,) and emptied[0][2].shape == (0, 3)


def test_a_host_whose_points_were_all_culled_redraws_its_block() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    apply(blocks, [10, 20], [10, 20], points_of([10, 20]))
    blocks.redraw()
    apply(blocks, [10, 20], [20], points_of([]))
    redraw = blocks.redraw()
    assert [block for block, _, _ in redraw] == [0] and len(redraw[0][1]) == 3
