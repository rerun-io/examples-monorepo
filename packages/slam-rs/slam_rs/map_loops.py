"""The back end's verified loops and pose-graph corrections, drawn as line strips under the run (DESIGN §8, PRs b2, c2).

A loop links the keyframe that found it to the keyframe it closes with. Fused loops (small drift, landmarks merged at
once) and loops at or above the drift gate (the pose-graph optimization's) get different colours. A correction links a
keyframe's raw (window) position to where the pose-graph optimization moved it.
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
"""A verified loop at or above the drift gate, for the pose-graph optimization: yellow."""
PGO_CORRECTIONS_ENTITY: str = "/world/runs/slam_rs/pgo/corrections"
"""Where the pose-graph corrections go."""
PGO_CORRECTION_COLOR: tuple[int, int, int, int] = (120, 160, 255, 255)
"""A keyframe's raw-to-corrected move: blue."""


def loop_segments(
    keyframe_t_ns: Int64[ndarray, " k"],
    keyframe_poses: Float64[ndarray, "k 7"],
    loop_edges: Int64[ndarray, "l 2"],
    loop_fused: Bool[ndarray, " l"],
) -> tuple[Float64[ndarray, "m 2 3"], Bool[ndarray, " m"]]:
    """The two keyframe positions of each loop whose keyframes the map still holds.

    Args:
        keyframe_t_ns: Map keyframe timestamps, ascending.
        keyframe_poses: ``[tx, ty, tz, qx, qy, qz, qw]`` per keyframe.
        loop_edges: ``[current, candidate]`` keyframe timestamps per loop.
        loop_fused: Whether each loop was fused at once.

    Returns:
        The ``[current, candidate]`` positions per kept loop, and whether each kept loop was fused.
    """
    rows: Int64[ndarray, "l 2"] = np.searchsorted(keyframe_t_ns, loop_edges).astype(np.int64)
    inside: Bool[ndarray, "l 2"] = rows < len(keyframe_t_ns)
    inside[inside] = keyframe_t_ns[rows[inside]] == loop_edges[inside]
    kept: Bool[ndarray, " l"] = inside.all(axis=1)
    segments: Float64[ndarray, "m 2 3"] = keyframe_poses[rows[kept], 0:3].reshape(-1, 2, 3)
    return segments, loop_fused[kept]


def correction_segments(raw_poses: Float64[ndarray, "k 7"], corrected_poses: Float64[ndarray, "k 7"]) -> Float64[ndarray, "m 2 3"]:
    """The ``[raw, corrected]`` positions of each keyframe whose position the pose-graph optimization moved.

    Args:
        raw_poses: The window's ``[tx, ty, tz, qx, qy, qz, qw]`` per keyframe.
        corrected_poses: The map's, after the corrections.

    Returns:
        One segment per moved keyframe, in keyframe order.
    """
    moved: Bool[ndarray, " k"] = np.any(raw_poses[:, 0:3] != corrected_poses[:, 0:3], axis=1)
    return np.stack([raw_poses[moved, 0:3], corrected_poses[moved, 0:3]], axis=1)


def log_loops(map_snapshot: _core.MapSnapshot) -> None:
    """Draw the snapshot's loops and corrections at the current cursor (an empty set is logged too, which clears old ones)."""
    segments, fused = loop_segments(map_snapshot.keyframe_t_ns, map_snapshot.keyframe_poses, map_snapshot.loop_edges, map_snapshot.loop_fused)
    colors: UInt8[ndarray, "m 4"] = np.where(fused[:, None], np.array(FUSED_LOOP_COLOR, dtype=np.uint8), np.array(PGO_LOOP_COLOR, dtype=np.uint8))
    rr.log(LOOPS_ENTITY, rr.LineStrips3D(list(segments), colors=colors, radii=0.006))
    corrections: Float64[ndarray, "m 2 3"] = correction_segments(map_snapshot.keyframe_raw_poses, map_snapshot.keyframe_poses)
    rr.log(PGO_CORRECTIONS_ENTITY, rr.LineStrips3D(list(corrections), colors=PGO_CORRECTION_COLOR, radii=0.004))
