"""The global BA's Rerun layer and trajectory (PR d2): what :meth:`slam_rs._core.Vio.finish_map` returns, drawn beside the raw VIO.

The raw VIO stays where it is (``/world/runs/slam_rs/{rig,trajectory,path}``); the GBA-corrected keyframes go to their own
entities under the same run, so its alignment transform places both in the dataset's world. The LM attempts go to
``/stats/gba`` on their own ``gba_attempt`` timeline, so the deep page can plot the convergence without a rerun.
"""

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray

from slam_rs import _core
from slam_rs.columns import ColumnLog
from slam_rs.map_loops import RUN_ENTITY
from slam_rs.trajectory import Trajectory

GBA_KEYFRAMES_ENTITY: str = f"{RUN_ENTITY}/gba/keyframes"
"""The map keyframes after the global BA, one point each."""
GBA_PATH_ENTITY: str = f"{RUN_ENTITY}/gba/path"
"""The same keyframes as one line, in time order."""
GBA_STATS_ENTITY: str = "/stats/gba"
"""The global BA's per-attempt scalars and its summary."""
GBA_TIMELINE: str = "gba_attempt"
"""Sequence timeline of the LM attempts, from 0."""
GBA_COLOR: tuple[int, int, int] = (255, 120, 220)
"""The GBA-corrected keyframes: magenta, apart from the raw estimate's green."""
GBA_KEYFRAME_RADIUS_M: float = 0.012
"""Radius of a GBA-corrected keyframe, metres."""
GBA_PATH_RADIUS_M: float = 0.004
"""Stroke of the GBA keyframes' path, metres."""


def keyframe_trajectory(result: _core.MapResult, *, after_gba: bool) -> Trajectory:
    """The map keyframes as a trajectory, on the clock ``track`` was given.

    Args:
        result: What :meth:`slam_rs._core.Vio.finish_map` returned.
        after_gba: The poses after the global BA, or the map's before it (after loops and PGO: the +LC keyframes).

    Returns:
        One pose per map keyframe, oldest first, quaternions w-first.
    """
    poses: Float64[ndarray, "k 7"] = result.keyframe_poses if after_gba else result.keyframe_poses_before_gba
    return Trajectory.from_xyzw(result.keyframe_t_ns, poses)


def log_map_result(result: _core.MapResult, columns: ColumnLog, t_ns: int) -> None:
    """Draw the GBA-corrected keyframes at ``video_time`` ``t_ns``, and the GBA's attempts on :data:`GBA_TIMELINE`.

    Nothing is drawn when the GBA did not run: the map's keyframes are already drawn by the VIO layer.

    Args:
        result: What :meth:`slam_rs._core.Vio.finish_map` returned.
        columns: The run's held rows (:attr:`slam_rs.vio_log.VioLogger.columns`), and the recording they go to.
        t_ns: ``video_time`` of the row, usually the last pose's.
    """
    gba: _core.GbaRun | None = result.gba
    if gba is None or result.keyframe_t_ns.shape[0] == 0:
        return
    recording: rr.RecordingStream | None = columns.recording
    positions: Float64[ndarray, "k 3"] = result.keyframe_poses[:, 0:3]
    columns.static_once(GBA_KEYFRAMES_ENTITY, lambda: rr.Points3D.from_fields(colors=GBA_COLOR, radii=GBA_KEYFRAME_RADIUS_M))
    columns.add_batch(GBA_KEYFRAMES_ENTITY, rr.Points3D.columns, t_ns, positions=positions)
    columns.static_once(GBA_PATH_ENTITY, lambda: rr.LineStrips3D.from_fields(colors=GBA_COLOR, radii=GBA_PATH_RADIUS_M))
    columns.add_batch(GBA_PATH_ENTITY, rr.LineStrips3D.columns, t_ns, strips=[positions])
    columns.flush()
    attempts: rr.TimeColumn = rr.TimeColumn(GBA_TIMELINE, sequence=np.arange(gba.iteration_lambda.shape[0], dtype=np.int64))
    for name, values in (
        ("cost", gba.iteration_cost),
        ("lambda", gba.iteration_lambda),
        ("cg_iterations", gba.iteration_cg.astype(np.float64)),
        ("accepted", gba.iteration_accepted.astype(np.float64)),
        ("wall_s", gba.iteration_stage_s.sum(axis=1)),
    ):
        rr.send_columns(f"{GBA_STATS_ENTITY}/{name}", indexes=[attempts], columns=rr.Scalars.columns(scalars=values), recording=recording)
    rr.log(
        GBA_STATS_ENTITY,
        rr.AnyValues(
            keyframes=gba.keyframes,
            landmarks=gba.landmarks,
            observations=gba.observations,
            pose_graph_rows=gba.pose_graph_rows,
            keyframes_without_reprojection=gba.keyframes_without_reprojection,
            initial_cost=gba.initial_cost,
            final_cost=gba.final_cost,
            termination=gba.termination,
            build_s=gba.build_s,
            solve_s=gba.solve_s,
            apply_s=gba.apply_s,
            gauge_tilt_deg=gba.gauge_tilt_deg,
            threads=gba.threads,
        ),
        static=True,
        recording=recording,
    )
