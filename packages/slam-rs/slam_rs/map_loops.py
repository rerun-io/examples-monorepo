"""The back end's verified loops, pose-graph corrections and pose graph, drawn as line strips under the run (DESIGN §8, PRs b2, c2, f1).

A loop links the keyframe that found it to the keyframe it closes with. Fused loops (landmarks merged, at once or after
a pose-graph optimization) and loops left unapplied get different colours. A correction links a keyframe's raw (window)
position to where the pose-graph optimization moved it.
"""

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float64, Int64, UInt8
from numpy import ndarray

from slam_rs import _core

RUN_ENTITY: str = "/world/runs/slam_rs"
"""The final run and the back end's map, beside the dataset's own ``/world/runs/gt`` (the catalog layer's ``slam_rs``)."""
LOOPS_ENTITY: str = f"{RUN_ENTITY}/loops"
"""Where the loop edges go, beside the map's keyframes and points."""
FUSED_LOOP_COLOR: tuple[int, int, int, int] = (255, 70, 200, 255)
"""A loop whose landmarks were fused (at once or after a pose-graph optimization): magenta."""
PGO_LOOP_COLOR: tuple[int, int, int, int] = (255, 225, 40, 255)
"""A verified loop left unapplied (at or above the drift gate, the pose-graph optimization off or refused): yellow."""
PGO_CORRECTIONS_ENTITY: str = f"{RUN_ENTITY}/pgo/corrections"
"""Where the pose-graph corrections go."""
PGO_CORRECTION_COLOR: tuple[int, int, int, int] = (120, 160, 255, 255)
"""A keyframe's raw-to-corrected move: blue."""
PGO_BEFORE_ENTITY: str = f"{RUN_ENTITY}/pgo/before"
"""The map keyframes' path before any pose-graph correction (the window's poses)."""
PGO_AFTER_ENTITY: str = f"{RUN_ENTITY}/pgo/after"
"""The map keyframes' path as the pose-graph optimizations left it."""
PGO_BEFORE_COLOR: tuple[int, int, int, int] = (150, 150, 165, 255)
"""Before: grey, as the raw VIO run."""
PGO_AFTER_COLOR: tuple[int, int, int, int] = (60, 200, 120, 255)
"""After: green."""
POSE_GRAPH_ENTITY: str = f"{RUN_ENTITY}/pose_graph"
"""The edges a pose-graph optimization takes: spanning tree, loop closures, high covisibility."""
POSE_GRAPH_COLORS: tuple[tuple[int, int, int, int], ...] = ((90, 90, 110, 160), (255, 225, 40, 255), (80, 170, 255, 110))
"""Per edge kind (``MapSnapshot.pose_graph_edge_kinds``): tree dark grey, loop closure yellow, high covisibility faint blue."""


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


def log_loops(keyframes: _core.MapKeyframes) -> None:
    """Draw the map's loops and corrections at the current cursor (an empty set is logged too, which clears old ones)."""
    segments: Float64[ndarray, "l 2 3"] = loop_segments(keyframes.t_ns, keyframes.poses, keyframes.loop_edges)
    fused: Bool[ndarray, " l"] = keyframes.loop_fused
    colors: UInt8[ndarray, "l 4"] = np.where(fused[:, None], np.array(FUSED_LOOP_COLOR, dtype=np.uint8), np.array(PGO_LOOP_COLOR, dtype=np.uint8))
    rr.log(LOOPS_ENTITY, rr.LineStrips3D(list(segments), colors=colors, radii=0.006))
    corrections: Float64[ndarray, "m 2 3"] = correction_segments(keyframes.raw_poses, keyframes.poses)
    rr.log(PGO_CORRECTIONS_ENTITY, rr.LineStrips3D(list(corrections), colors=PGO_CORRECTION_COLOR, radii=0.004))


def log_pose_graph(map_snapshot: _core.MapSnapshot) -> None:
    """Draw the map keyframes' path before and after the pose-graph corrections, and the pose graph's edges, at the current cursor."""
    rr.log(PGO_BEFORE_ENTITY, rr.LineStrips3D([map_snapshot.keyframe_raw_poses[:, 0:3]], colors=PGO_BEFORE_COLOR, radii=0.003))
    rr.log(PGO_AFTER_ENTITY, rr.LineStrips3D([map_snapshot.keyframe_poses[:, 0:3]], colors=PGO_AFTER_COLOR, radii=0.003))
    segments: Float64[ndarray, "g 2 3"] = loop_segments(map_snapshot.keyframe_t_ns, map_snapshot.keyframe_poses, map_snapshot.pose_graph_edges)
    colors: UInt8[ndarray, "g 4"] = np.array(POSE_GRAPH_COLORS, dtype=np.uint8)[map_snapshot.pose_graph_edge_kinds]
    rr.log(POSE_GRAPH_ENTITY, rr.LineStrips3D(list(segments), colors=colors, radii=0.002))
