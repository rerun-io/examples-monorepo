"""The back end's verified loops drawn as line strips under the run (DESIGN §8, PR b2)."""

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray


def test_a_loop_is_a_segment_between_its_two_keyframe_positions() -> None:
    from slam_rs.map_loops import LOOPS_ENTITY, loop_segments

    assert LOOPS_ENTITY == "/world/runs/slam_rs/loops"
    keyframe_t_ns: Int64[ndarray, " k"] = np.array([10, 20, 30], dtype=np.int64)
    poses: Float64[ndarray, "k 7"] = np.zeros((3, 7), dtype=np.float64)
    poses[:, 0] = [1.0, 2.0, 3.0]
    poses[:, 6] = 1.0
    edges: Int64[ndarray, "l 2"] = np.array([[30, 10], [20, 30]], dtype=np.int64)
    segments = loop_segments(keyframe_t_ns, poses, edges)
    np.testing.assert_array_equal(segments, [[[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])


def test_no_loop_gives_no_segment() -> None:
    from slam_rs.map_loops import loop_segments

    segments = loop_segments(np.array([10], dtype=np.int64), np.zeros((1, 7)), np.zeros((0, 2), dtype=np.int64))
    assert segments.shape == (0, 2, 3)
