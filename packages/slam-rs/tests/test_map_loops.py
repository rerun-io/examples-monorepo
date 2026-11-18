"""The back end's verified loops and pose-graph corrections drawn as line strips under the run (DESIGN §8, PRs b2, c2)."""

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


def test_a_pgo_correction_is_a_segment_from_the_raw_to_the_corrected_keyframe() -> None:
    from slam_rs.map_loops import PGO_CORRECTIONS_ENTITY, correction_segments

    assert PGO_CORRECTIONS_ENTITY == "/world/runs/slam_rs/pgo/corrections"
    raw: Float64[ndarray, "k 7"] = np.zeros((3, 7), dtype=np.float64)
    raw[:, 0] = [1.0, 2.0, 3.0]
    raw[:, 6] = 1.0
    corrected: Float64[ndarray, "k 7"] = raw.copy()
    corrected[0, 1] = 0.5
    # A rotation alone moves no position: no segment.
    corrected[2, 5:7] = [np.sqrt(0.5), np.sqrt(0.5)]
    segments: Float64[ndarray, "m 2 3"] = correction_segments(raw, corrected)
    assert segments.shape == (1, 2, 3)
    np.testing.assert_array_equal(segments[0], [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
