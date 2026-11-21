"""The map points' delta logging on arrays: each map delta replaces its changed hosts' points, and only the keyframe
blocks whose points moved or left are drawn again."""

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray

from slam_rs.map_point_blocks import MapPointBlocks

Points = tuple[Int64[ndarray, " c"], Int64[ndarray, " p"], Float32[ndarray, "p 3"]]


def points_of(hosts: list[int], points_per_host: int = 3) -> Points:
    """Point counts, ids and world positions as a delta gives them: keyframe ``h`` hosts ``points_per_host`` points
    near x = h, ids from 100 + 3h, in descending id order within the host (the back end orders by camera, not id)."""
    counts: Int64[ndarray, " c"] = np.full(len(hosts), points_per_host, dtype=np.int64)
    point_ids: Int64[ndarray, " p"] = np.asarray(
        [100 + points_per_host * h + k for h in hosts for k in reversed(range(points_per_host))], dtype=np.int64
    )
    point_hosts: Float32[ndarray, " p"] = np.repeat(np.asarray(hosts, dtype=np.float32), points_per_host)
    positions: Float32[ndarray, "p 3"] = np.stack([point_hosts, np.zeros_like(point_hosts), (point_ids % 7).astype(np.float32)], axis=1)
    return counts, point_ids, positions


def apply(blocks: MapPointBlocks, keyframes: list[int], changed: list[int], points: Points) -> None:
    """Feed one map delta, one revision newer than the last: the map's keyframes, the hosts whose points moved, and
    those hosts' points."""
    counts, ids, positions = points
    if len(counts) < len(changed):  # hosts that lost every point come last, with none
        counts = np.concatenate([counts, np.zeros(len(changed) - len(counts), dtype=np.int64)])
    blocks.apply(blocks.revision + 1, np.asarray(keyframes, dtype=np.int64), np.asarray(changed, dtype=np.int64), counts, ids, positions)


def test_a_first_delta_draws_every_block_and_a_quiet_one_none() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    apply(blocks, [10, 20, 30, 40, 50], [10, 20, 30, 40, 50], points_of([10, 20, 30, 40, 50]))
    first = blocks.redraw()
    assert [block for block, _, _ in first] == [0, 1, 2]
    assert sum(len(ids) for _, ids, _ in first) == 15
    apply(blocks, [10, 20, 30, 40, 50], [], points_of([]))
    assert blocks.redraw() == []


def test_a_delta_no_newer_than_the_last_one_taken_changes_nothing() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    apply(blocks, [10], [10], points_of([10]))
    blocks.redraw()
    counts, ids, positions = points_of([10])
    blocks.apply(blocks.revision, np.asarray([10]), np.asarray([10]), counts, ids, positions + np.float32(1.0))
    assert blocks.redraw() == []


def test_only_the_host_that_moved_past_the_tolerance_redraws_its_block() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2, tolerance_m=0.001)
    apply(blocks, [10, 20, 30, 40], [10, 20, 30, 40], points_of([10, 20, 30, 40]))
    blocks.redraw()
    counts, ids, positions = points_of([30])
    apply(blocks, [10, 20, 30, 40], [30], (counts, ids, positions + np.float32(0.0005)))
    assert blocks.redraw() == [], "half a millimetre is under the tolerance"
    apply(blocks, [10, 20, 30, 40], [30], (counts, ids, positions + np.float32(0.01)))
    redraw = blocks.redraw()
    assert [block for block, _, _ in redraw] == [1]
    _, expected_ids, expected = points_of([30, 40])
    expected[:3] += np.float32(0.01)
    np.testing.assert_array_equal(redraw[0][1], expected_ids)
    np.testing.assert_array_equal(redraw[0][2], expected)


def test_the_tolerance_is_a_distance_not_a_per_axis_bound() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2, tolerance_m=0.001)
    apply(blocks, [10], [10], points_of([10]))
    blocks.redraw()
    counts, ids, positions = points_of([10])
    diagonal: Float32[ndarray, " 3"] = np.array([0.0008, 0.0008, 0.0], dtype=np.float32)  # 1.13 mm, each axis under 1 mm
    apply(blocks, [10], [10], (counts, ids, positions + diagonal))
    assert [block for block, _, _ in blocks.redraw()] == [0]


def test_small_moves_add_up_against_what_was_drawn() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2, tolerance_m=0.001)
    apply(blocks, [10], [10], points_of([10]))
    blocks.redraw()
    counts, ids, positions = points_of([10])
    along_x: Float32[ndarray, " 3"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    apply(blocks, [10], [10], (counts, ids, positions + np.float32(0.0006) * along_x))
    assert blocks.redraw() == []
    apply(blocks, [10], [10], (counts, ids, positions + np.float32(0.0012) * along_x))
    assert [block for block, _, _ in blocks.redraw()] == [0], "1.2 mm from the drawn points in two small steps"


def test_changes_between_draws_add_up_and_new_keyframes_fill_later_blocks() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    apply(blocks, [10, 20, 30], [10, 20, 30], points_of([10, 20, 30]))
    blocks.redraw()
    # Two reads before the next draw: keyframe 40 joins block 1 (beside 30), then 50 opens block 2. Block 0 is untouched.
    apply(blocks, [10, 20, 30, 40], [40], points_of([40]))
    apply(blocks, [10, 20, 30, 40, 50], [50], points_of([50]))
    assert [block for block, _, _ in blocks.redraw()] == [1, 2]


def test_a_keyframe_that_left_the_map_empties_its_points_and_a_block_left_with_none_comes_back_empty() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    apply(blocks, [10, 20, 30, 40], [10, 20, 30, 40], points_of([10, 20, 30, 40]))
    blocks.redraw()
    # Keyframe 40 is culled (the delta names no host; it is missing from the keyframes): block 1 keeps 30's points.
    apply(blocks, [10, 20, 30], [], points_of([]))
    redraw = blocks.redraw()
    assert [block for block, _, _ in redraw] == [1] and len(redraw[0][1]) == 3
    # Then 30 goes too: block 1 has no points left.
    apply(blocks, [10, 20], [], points_of([]))
    emptied = blocks.redraw()
    assert len(emptied) == 1 and emptied[0][0] == 1 and emptied[0][1].shape == (0,) and emptied[0][2].shape == (0, 3)
    assert blocks.block_of_host == {10: 0, 20: 0} and blocks.hosts_of_block == [[10, 20], []], "the culled hosts are forgotten"


def test_blocks_keep_their_place_when_keyframes_leave() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    apply(blocks, [10, 20], [10, 20], points_of([10, 20]))
    apply(blocks, [20], [], points_of([]))
    apply(blocks, [20, 30], [30], points_of([30]))
    assert blocks.block_of_host == {20: 0, 30: 1}, "30 opens block 1: block 0's place is 10's, which left"


def test_a_host_whose_points_were_all_culled_redraws_its_block() -> None:
    blocks = MapPointBlocks(keyframes_per_block=2)
    apply(blocks, [10, 20], [10, 20], points_of([10, 20]))
    blocks.redraw()
    apply(blocks, [10, 20], [20], points_of([]))
    redraw = blocks.redraw()
    assert [block for block, _, _ in redraw] == [0] and len(redraw[0][1]) == 3
