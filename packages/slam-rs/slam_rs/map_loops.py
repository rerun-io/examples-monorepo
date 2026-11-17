"""The back end's verified loops, drawn as line strips under the run (DESIGN §8, PR b2).

A loop links the keyframe that found it to the keyframe it closes with. Fused loops (small drift, landmarks merged at
once) and loops that wait for the pose-graph optimization get different colours.
"""

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float64, Int64, UInt8
from numpy import ndarray

from slam_rs import _core

LOOPS_ENTITY: str = "/world/runs/slam_rs/loops"
"""Where the loop edges go, beside the map's keyframes and points."""
FUSED_LOOP_COLOR: tuple[int, int, int, int] = (255, 70, 200, 255)
"""A loop whose landmarks were fused at once: magenta."""
PGO_LOOP_COLOR: tuple[int, int, int, int] = (255, 225, 40, 255)
"""A verified loop left for the pose-graph optimization: yellow."""


def loop_segments(
    keyframe_t_ns: Int64[ndarray, " k"],
    keyframe_poses: Float64[ndarray, "k 7"],
    loop_edges: Int64[ndarray, "l 2"],
) -> Float64[ndarray, "l 2 3"]:
    """The two keyframe positions of each loop (the snapshot keeps only loops whose keyframes the map holds).

    Args:
        keyframe_t_ns: Map keyframe timestamps, ascending.
        keyframe_poses: ``[tx, ty, tz, qx, qy, qz, qw]`` per keyframe.
        loop_edges: ``[current, candidate]`` keyframe timestamps per loop.

    Returns:
        The ``[current, candidate]`` positions per loop.
    """
    rows: Int64[ndarray, "l 2"] = np.searchsorted(keyframe_t_ns, loop_edges).astype(np.int64)
    return keyframe_poses[rows, 0:3]


def log_loops(map_snapshot: _core.MapSnapshot) -> None:
    """Draw the snapshot's loops at the current cursor (none drawn is logged as an empty set, which clears old ones)."""
    segments: Float64[ndarray, "l 2 3"] = loop_segments(map_snapshot.keyframe_t_ns, map_snapshot.keyframe_poses, map_snapshot.loop_edges)
    fused: Bool[ndarray, " l"] = map_snapshot.loop_fused
    colors: UInt8[ndarray, "l 4"] = np.where(fused[:, None], np.array(FUSED_LOOP_COLOR, dtype=np.uint8), np.array(PGO_LOOP_COLOR, dtype=np.uint8))
    rr.log(LOOPS_ENTITY, rr.LineStrips3D(list(segments), colors=colors, radii=0.006))
