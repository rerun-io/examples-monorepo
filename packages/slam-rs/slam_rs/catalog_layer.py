"""Pose layers in DataForge's base-rig layout, following the Basalt layer contract."""

from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from dataforge import schema
from dataforge.writing import atomic_recording
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from slam_rs.catalog_feed import RIG_ENTITY, TIMELINE
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import shift_clock

POSE_SOURCE: str = "slam_rs"
"""Catalog layer name and DataForge run name of the slam-rs pose layer."""


def write_layer(
    path: Path,
    segment_id: str,
    run: SegmentRun,
    *,
    clock_offset_ns: int,
    profile: Literal["fast", "reference"],
    backend: str,
    decoder: str = "cpu",
) -> None:
    """Atomically write poses on base video time, without changing base sensor data.

    ``clock_offset_ns`` is the feed's export offset plus camera-to-IMU offset.
    Empty, incomplete or non-finite runs cannot replace an existing result.
    """
    if len(run.estimate) < 2 or run.lost or len(run.estimate) != run.framesets:
        raise ValueError(f"refusing incomplete pose layer: {len(run.estimate)} poses / {run.framesets} framesets, {run.lost} pending")
    positions: Float64[ndarray, "n 3"] = run.estimate.position_m
    rotations: Float64[ndarray, "n 4"] = np.roll(run.estimate.quaternion_wxyz, -1, axis=1)
    times: Int64[ndarray, " n"] = shift_clock(run.estimate, -clock_offset_ns).t_ns
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all() or not np.all(np.diff(times) > 0):
        raise ValueError("refusing non-finite poses or non-increasing timestamps")
    root: str = schema.run_path(POSE_SOURCE)
    with atomic_recording(path, application_id="dataforge", recording_id=segment_id, send_properties=False) as rec:
        rec.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rec.send_columns(RIG_ENTITY, indexes=[rr.TimeColumn(TIMELINE, duration=times.astype("timedelta64[ns]"))],
                         columns=rr.Transform3D.columns(translation=positions, quaternion=rotations))
        indices: Int64[ndarray, " n_overview"] = np.unique(np.append(np.arange(0, len(positions), max(1, len(positions) // 4000)), len(positions) - 1))
        overview: Float64[ndarray, "n_overview 3"] = positions[indices]
        progress: Float64[ndarray, " n_edges"] = np.linspace(0.0, 1.0, len(positions) - 1)
        colors: UInt8[ndarray, "n_edges 3"] = np.column_stack([255 * progress, 255 * (1 - progress), np.zeros_like(progress)]).astype(np.uint8)
        rec.log(schema.trajectory_path(POSE_SOURCE), rr.LineStrips3D(np.stack([overview[:-1], overview[1:]], axis=1), colors=colors[indices[:-1]], radii=rr.Radius.ui_points(1.5)), static=True)
        rec.log(f"{root}/endpoints", rr.Points3D(positions[[0, -1]], colors=[[0, 255, 0], [255, 0, 0]], radii=rr.Radius.ui_points(3.0), labels=["start", "end"]), static=True)
        rec.send_columns(
            schema.trail_path(POSE_SOURCE),
            indexes=[rr.TimeColumn(TIMELINE, duration=times[1:].astype("timedelta64[ns]"))],
            columns=rr.LineStrips3D.columns(strips=np.stack([positions[:-1], positions[1:]], axis=1), colors=colors),
        )
        rec.log(schema.trail_path(POSE_SOURCE), rr.LineStrips3D.from_fields(radii=0.006), static=True)
        rec.log(f"{root}", rr.AnyValues(source="slam-rs catalog VIO", profile=profile, backend=backend, decoder=decoder, config_sha256=run.config_sha256,
                                      num_poses=len(positions), wall_s=run.wall_s, median_tracker_ms=run.median_tracker_ms,
                                      export_to_video_time_offset_ns=clock_offset_ns), static=True)
