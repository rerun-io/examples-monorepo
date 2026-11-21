"""The slam-rs result layer: the final run and the raw VIO run, each under its own ``/world/runs/<name>``.

The dataset's ``gt`` layer owns ``/world/rig_00``; a result written there would
merge with the ground truth in every catalog query. The layer is still named
``slam_rs`` and holds both runs, so one registration replaces both.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from dataforge import schema
from dataforge.writing import atomic_recording
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from slam_rs.catalog_feed import RIG_ENTITY, TIMELINE
from slam_rs.follow_frame import follow_entity, follow_yaw, yaw_quaternion_xyzw
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import Trajectory, shift_clock

POSE_SOURCE: str = "slam_rs"
"""Catalog layer name, and DataForge run name of the final (optimised, non-causal) trajectory."""
VIO_SOURCE: str = "slam_rs_vio"
"""DataForge run name of the raw causal VIO trajectory, in the same layer."""
VIO_COLOR: tuple[int, int, int] = (150, 150, 165)
"""The raw VIO run is drawn grey, under the final run's start-to-end gradient."""
RIG_AXIS_LENGTH_M: float = 0.1
"""Axes drawn at each run's rig, metres."""


def write_layer(
    path: Path,
    segment_id: str,
    run: SegmentRun,
    *,
    clock_offset_ns: int,
    profile: Literal["fast", "reference"],
    backend: str,
    decoder: str = "cpu",
    animate_base_rig: bool = False,
) -> None:
    """Atomically write poses on base video time, without changing base sensor data.

    ``clock_offset_ns`` is the feed's export offset plus camera-to-IMU offset.
    Explicit initialization and terminal IMU drops are recorded; unexplained
    losses, interior gaps and non-finite runs cannot replace an existing result.
    ``animate_base_rig`` also writes the final poses on ``/world/rig_00``; only
    for a dataset without a ``gt`` layer, where nothing else moves the rig.
    Each run carries its level follow frame (:func:`slam_rs.follow_frame.follow_entity`).
    The caller decides from the catalog, so a dataset's ground truth must be
    registered as the layer named ``gt``: a base layer, or a layer by another
    name, that animates the rig would clash again without an error.
    """
    if problem := run.unexplained_loss():
        raise ValueError(problem)
    final: Trajectory = run.estimate if run.final is None else run.final
    if not np.array_equal(final.t_ns, run.estimate.t_ns):
        raise ValueError("the final trajectory must have the raw VIO's timestamps")
    positions: Float64[ndarray, "n 3"] = final.position_m
    rotations: Float64[ndarray, "n 4"] = final.xyzw()[:, 3:7]
    raw_positions: Float64[ndarray, "n 3"] = run.estimate.position_m
    raw_rotations: Float64[ndarray, "n 4"] = run.estimate.xyzw()[:, 3:7]
    times: Int64[ndarray, " n"] = shift_clock(run.estimate, -clock_offset_ns).t_ns
    finite: bool = all(np.isfinite(array).all() for array in (positions, rotations, raw_positions, raw_rotations))
    if not finite or not np.all(np.diff(times) > 0):
        raise ValueError("refusing non-finite poses or non-increasing timestamps")
    progress: Float64[ndarray, " n_edges"] = np.linspace(0.0, 1.0, len(positions) - 1)
    gradient: UInt8[ndarray, "n_edges 3"] = np.column_stack([255 * progress, 255 * (1 - progress), np.zeros_like(progress)]).astype(np.uint8)
    root: str = schema.run_path(POSE_SOURCE)
    with atomic_recording(path, application_id="dataforge", recording_id=segment_id, send_properties=False) as rec:
        rec.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        _log_run(rec, POSE_SOURCE, times, positions, rotations, gradient)
        _log_run(rec, VIO_SOURCE, times, raw_positions, raw_rotations, np.tile(np.array(VIO_COLOR, dtype=np.uint8), (len(progress), 1)))
        if animate_base_rig:
            rec.send_columns(RIG_ENTITY, indexes=[rr.TimeColumn(TIMELINE, duration=times.astype("timedelta64[ns]"))],
                             columns=rr.Transform3D.columns(translation=positions, quaternion=rotations))
        rec.log(f"{root}", rr.AnyValues(source="slam-rs catalog VIO", profile=profile, backend=backend, decoder=decoder, config_sha256=run.config_sha256,
                                      calibration_sha256=run.calibration_sha256,
                                      num_poses=len(positions), num_framesets=run.framesets, num_dropped=len(run.dropped),
                                      wall_s=run.wall_s, median_tracker_ms=run.median_tracker_ms,
                                      export_to_video_time_offset_ns=clock_offset_ns), static=True)
        if run.dropped:
            rec.log(f"{root}/dropped", rr.AnyValues(
                t_ns=[drop.t_ns - clock_offset_ns for drop in run.dropped], reasons=[drop.reason for drop in run.dropped],
            ), static=True)


def _log_run(
    rec: rr.RecordingStream,
    source: str,
    times: Int64[ndarray, " n"],
    positions: Float64[ndarray, "n 3"],
    rotations: Float64[ndarray, "n 4"],
    colors: UInt8[ndarray, "n_edges 3"],
) -> None:
    """One run under ``/world/runs/<source>``: the per-frame rig, the whole path, its trail up to the cursor and its endpoints.

    Args:
        rec: The layer's recording.
        source: DataForge run name.
        times: ``video_time`` of each pose, nanoseconds.
        positions: Rig positions in the world, metres.
        rotations: Rig rotations, Rerun's XYZW.
        colors: One colour per edge between consecutive poses.
    """
    rig: str = f"{schema.run_path(source)}/rig"
    rec.send_columns(rig, indexes=[rr.TimeColumn(TIMELINE, duration=times.astype("timedelta64[ns]"))],
                     columns=rr.Transform3D.columns(translation=positions, quaternion=rotations))
    rec.send_columns(follow_entity(schema.run_path(source)), indexes=[rr.TimeColumn(TIMELINE, duration=times.astype("timedelta64[ns]"))],
                     columns=rr.Transform3D.columns(translation=positions, quaternion=yaw_quaternion_xyzw(follow_yaw(times, positions))))
    rec.log(rig, rr.TransformAxes3D(RIG_AXIS_LENGTH_M), static=True)
    indices: Int64[ndarray, " n_overview"] = np.unique(np.append(np.arange(0, len(positions), max(1, len(positions) // 4000)), len(positions) - 1))
    overview: Float64[ndarray, "n_overview 3"] = positions[indices]
    rec.log(schema.trajectory_path(source), rr.LineStrips3D(np.stack([overview[:-1], overview[1:]], axis=1), colors=colors[indices[:-1]],
                                                            radii=rr.Radius.ui_points(1.5)), static=True)
    rec.log(f"{schema.run_path(source)}/endpoints", rr.Points3D(positions[[0, -1]], colors=[[0, 255, 0], [255, 0, 0]],
                                                                 radii=rr.Radius.ui_points(3.0), labels=["start", "end"]), static=True)
    rec.send_columns(
        schema.trail_path(source),
        indexes=[rr.TimeColumn(TIMELINE, duration=times[1:].astype("timedelta64[ns]"))],
        columns=rr.LineStrips3D.columns(strips=np.stack([positions[:-1], positions[1:]], axis=1), colors=colors),
    )
    rec.log(schema.trail_path(source), rr.LineStrips3D.from_fields(radii=0.006), static=True)
