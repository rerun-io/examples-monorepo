"""The slam-rs result layer: the final run and the raw VIO run, each under its own ``/world/runs/<name>``.

The dataset's ``gt`` layer owns ``/world/rig_00``; a result written there would
merge with the ground truth in every catalog query. The layer is still named
``slam_rs`` and holds both runs, so one registration replaces both.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
from dataforge import schema
from dataforge.writing import atomic_recording
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from simplecv.rerun_walk_path import ENDPOINTS, TRAIL, segments, send_endpoints, send_trail, walk_colors

from slam_rs import _core
from slam_rs.catalog_feed import RIG_ENTITY, TIMELINE, CameraCalib, SegmentFeed
from slam_rs.columns import ColumnLog
from slam_rs.config import config_text_sha256
from slam_rs.follow_frame import follow_entity, follow_yaw, yaw_quaternion_xyzw
from slam_rs.tracking import Lockstep, SegmentRun, drive
from slam_rs.trajectory import Trajectory, shift_clock
from slam_rs.vio_log import COLOR_DOWNSCALE, VioLogger, VioStage, dataset_videos, log_rig

POSE_SOURCE: str = "slam_rs"
"""Catalog layer name, and DataForge run name of the final (optimised, non-causal) trajectory."""
VIO_SOURCE: str = "slam_rs_vio"
"""DataForge run name of the raw causal VIO trajectory, in the same layer."""
RIG_AXIS_LENGTH_M: float = 0.1
"""Axes drawn at each run's rig, metres."""
LayerDetail: TypeAlias = Literal["poses", "full"]
"""What a run's layer holds: ``poses``, both runs' paths and the metadata; ``full`` adds what the run sees frameset by
frameset (keypoints on the base cameras, the window's frusta, landmarks in image colours, the map as it grows, the
counters; :class:`slam_rs.vio_log.VioLogger`)."""


@dataclass(slots=True, frozen=True)
class LayerRun:
    """What :func:`track_layer` produced: the run, and with ``full`` detail the stage that drew it."""

    run: SegmentRun
    """The tracked run, on the absolute device clock."""
    stage: VioStage | None
    """The estimator stage and its logger (stage times, keypoint counts, the bootstrap); None with ``poses`` detail."""


def track_layer(
    rec: rr.RecordingStream,
    feed: SegmentFeed,
    vio: _core.Vio,
    *,
    config_text: str,
    calibration: _core.Calibration,
    run_gba: bool,
    detail: LayerDetail,
    profile: Literal["fast", "reference"],
    animate_base_rig: bool,
    strict: bool,
    max_framesets: int | None = None,
    stop_ns: int | None = None,
) -> LayerRun:
    """Track the segment once and write its whole result layer into ``rec``: the one driver of every tool that writes one.

    The run goes through :func:`slam_rs.tracking.drive`; with ``full`` detail a :class:`VioStage` sees each pose as it
    is produced and holds its rows as columns (sent every few hundred framesets), and :func:`log_layer` draws both runs'
    paths from the finished trajectories. Nothing is tracked twice.

    Args:
        rec: The layer's recording (an atomic file for the catalog, or a viewer for a replay).
        feed: The open segment.
        vio: The estimator, built from ``calibration`` and ``config_text``.
        config_text: The exact config text the estimator was built from; its hash is recorded.
        calibration: The calibration the estimator was built from.
        run_gba: Run the global BA for the final run.
        detail: :data:`LayerDetail`.
        profile: Estimator configuration profile, recorded.
        animate_base_rig: Also move ``/world/rig_00`` with the final run (no ``gt`` layer).
        strict: Refuse an incomplete run (:func:`log_layer`).
        max_framesets: Stop after this many framesets; None tracks the whole segment.
        stop_ns: Stop after the frameset at this estimator time; None tracks the whole segment.

    Returns:
        The run, and the stage with ``full`` detail.
    """
    lockstep: Lockstep = Lockstep(vio)
    stage: VioStage | None = None
    cam_offset_ns: int = feed.imu.cam_time_offset_ns
    if detail == "full":
        # The landmarks take their colour from the frames: the estimator's own stay gray, the logger gets an RGB copy.
        feed = dataclasses.replace(feed, color_downscale=COLOR_DOWNSCALE)
        logger: VioLogger = VioLogger(
            cameras=feed.cameras,
            ground_truth=feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1])),
            tracked_resolution=tuple(vio.tracked_resolution),
            cam_time_offset_ns=cam_offset_ns,
            # The base's own lens videos: the window's frusta reference their frames, nothing is encoded again.
            video_entities=dataset_videos(feed.camera_positions),
            camera_positions=feed.camera_positions,
            base_downscale=feed.profile.downscale,
            columns=ColumnLog(recording=rec),
        )
        stage = VioStage(lockstep=lockstep, logger=logger)
    run: SegmentRun = drive(feed, lockstep, stop_ns, max_framesets, config_sha256=config_text_sha256(config_text), calibration=calibration,
                            run_gba=run_gba, observer=stage)
    log_layer(rec, run, clock_offset_ns=feed.video_time_offset_ns, profile=profile, backend=str(_core.gpu_backend) if vio.gpu else "cpu",
              decoder=feed.decode_device, detail=detail, animate_base_rig=animate_base_rig,
              lenses=tuple(zip(feed.cameras, dataset_videos(feed.camera_positions), strict=True)),
              walk_span_ns=(int(feed.frame_t_ns[0]) - cam_offset_ns, int(feed.frame_t_ns[-1]) - cam_offset_ns), strict=strict)
    if stage is not None:
        stage.finish(shift_clock(run.estimate if run.final is None else run.final, -feed.export_offset_ns), run.map_result)
    return LayerRun(run=run, stage=stage)


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
    """Atomically write a layer of the run's poses alone (:func:`log_layer`), without changing base sensor data.

    An exception (a refused run included) leaves an existing ``path`` as it was.
    """
    with atomic_recording(path, application_id="dataforge", recording_id=segment_id, send_properties=False) as rec:
        log_layer(rec, run, clock_offset_ns=clock_offset_ns, profile=profile, backend=backend, decoder=decoder, animate_base_rig=animate_base_rig)


def log_layer(
    rec: rr.RecordingStream,
    run: SegmentRun,
    *,
    clock_offset_ns: int,
    profile: Literal["fast", "reference"],
    backend: str,
    decoder: str = "cpu",
    detail: LayerDetail = "poses",
    animate_base_rig: bool = False,
    lenses: Sequence[tuple[CameraCalib, str]] = (),
    walk_span_ns: tuple[int, int] | None = None,
    strict: bool = True,
) -> None:
    """Write both runs' poses on base video time, and the run's metadata, into the layer's recording.

    This is the one writer of the two runs' rig, follow frame, trail, whole path and endpoints, for the catalog layer
    and the replay alike; what a run adds frameset by frameset (keypoints, window, landmarks, map, counters) is
    :class:`slam_rs.vio_log.VioLogger`'s, in the same recording.

    ``clock_offset_ns`` is the feed's export offset plus camera-to-IMU offset. Explicit initialization and terminal IMU
    drops are recorded. Unexplained losses, interior gaps and non-finite runs cannot replace an existing result with
    ``strict``; without it (a replay, which shows any run) they are a warning.
    ``animate_base_rig`` also writes the final poses on ``/world/rig_00``; only
    for a dataset without a ``gt`` layer, where nothing else moves the rig.
    Each run carries its level follow frame (:func:`slam_rs.follow_frame.follow_entity`).
    The caller decides from the catalog, so a dataset's ground truth must be
    registered as the layer named ``gt``: a base layer, or a layer by another
    name, that animates the rig would clash again without an error.

    Args:
        rec: The layer's recording.
        run: The tracked run, on the absolute device clock.
        clock_offset_ns: Device clock minus ``video_time``.
        profile: Estimator configuration profile, recorded.
        backend: Frontend lane, recorded.
        decoder: Video decoder, recorded.
        detail: What the layer holds (:data:`LayerDetail`), recorded.
        animate_base_rig: Also move ``/world/rig_00`` with the final run.
        lenses: Each fed camera with its base ``VideoStream``, in feed order: the raw run's rig carries a frustum a camera
            showing that lens's frame of the pose (``VideoFrameReference``). Empty: the rig carries no frusta.
        walk_span_ns: ``video_time`` the walk gradient runs from green to red over (the feed's first and last frameset);
            None: the first and last pose.
        strict: Refuse an incomplete run (above).
    """
    problem: str | None = run.unexplained_loss()
    if problem is not None and strict:
        raise ValueError(problem)
    if problem is not None:
        print(f"warning: {problem}")
    final: Trajectory = run.estimate if run.final is None else run.final
    if not np.array_equal(final.t_ns, run.estimate.t_ns):
        raise ValueError("the final trajectory must have the raw VIO's timestamps")
    if len(run.estimate) < 2:
        print(f"{len(run.estimate)} poses: no path to draw")
        return
    positions: Float64[ndarray, "n 3"] = final.position_m
    rotations: Float64[ndarray, "n 4"] = final.xyzw()[:, 3:7]
    raw_positions: Float64[ndarray, "n 3"] = run.estimate.position_m
    raw_rotations: Float64[ndarray, "n 4"] = run.estimate.xyzw()[:, 3:7]
    times: Int64[ndarray, " n"] = shift_clock(run.estimate, -clock_offset_ns).t_ns
    finite: bool = all(np.isfinite(array).all() for array in (positions, rotations, raw_positions, raw_rotations))
    if not finite or not np.all(np.diff(times) > 0):
        raise ValueError("refusing non-finite poses or non-increasing timestamps")
    span: tuple[int, int] = (int(times[0]), int(times[-1])) if walk_span_ns is None else walk_span_ns
    gradient: UInt8[ndarray, "n 3"] = walk_colors(times, span[0], span[1])
    root: str = schema.run_path(POSE_SOURCE)
    if len(run.ground_truth) == 0:
        # The runs are in the estimator's own gravity-aligned world; aligned to a ground truth they are in the
        # dataset's, whose base layer states its own axes.
        rec.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    _log_run(rec, POSE_SOURCE, times, positions, rotations, gradient)
    _log_run(rec, VIO_SOURCE, times, raw_positions, raw_rotations, gradient, lenses)
    if animate_base_rig:
        rec.send_columns(RIG_ENTITY, indexes=[rr.TimeColumn(TIMELINE, duration=times.astype("timedelta64[ns]"))],
                         columns=rr.Transform3D.columns(translation=positions, quaternion=rotations))
    rec.log(f"{root}", rr.AnyValues(source="slam-rs catalog VIO", profile=profile, backend=backend, decoder=decoder, detail=detail, config_sha256=run.config_sha256,
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
    colors: UInt8[ndarray, "n 3"],
    lenses: Sequence[tuple[CameraCalib, str]] = (),
) -> None:
    """One run under ``/world/runs/<source>`` (:mod:`simplecv.rerun_walk_path`'s names): the per-frame rig, its follow frame,
    the whole path (static here: every dataset's catalog layout draws it at any cursor), the trail up to the cursor and
    the endpoints, and with ``lenses`` the rig's frusta, each showing its lens's frame of the pose's time.

    Args:
        rec: The layer's recording.
        source: DataForge run name.
        times: ``video_time`` of each pose, nanoseconds.
        positions: Rig positions in the world, metres.
        rotations: Rig rotations, Rerun's XYZW.
        colors: One colour per pose (a segment takes its later pose's).
        lenses: Each camera of the run's rig copy (``rig/cam_NN``, feed order) with its base ``VideoStream``.
    """
    run: str = schema.run_path(source)
    rig: str = f"{run}/rig"
    index = [rr.TimeColumn(TIMELINE, duration=times.astype("timedelta64[ns]"))]
    rec.send_columns(rig, indexes=index, columns=rr.Transform3D.columns(translation=positions, quaternion=rotations))
    rec.send_columns(follow_entity(run), indexes=index, columns=rr.Transform3D.columns(translation=positions, quaternion=yaw_quaternion_xyzw(follow_yaw(times, positions))))
    rec.log(rig, rr.TransformAxes3D(RIG_AXIS_LENGTH_M), static=True)
    log_rig(tuple(camera for camera, _ in lenses), rig, recording=rec)
    for camera, video in lenses:
        lens: str = f"{rig}/cam_{camera.index:02d}/pinhole/video"
        rec.log(lens, rr.VideoFrameReference.from_fields(video_reference=video), static=True)
        rec.send_columns(lens, indexes=index, columns=rr.VideoFrameReference.columns(timestamp=times))
    indices: Int64[ndarray, " n_overview"] = np.unique(np.append(np.arange(0, len(positions), max(1, len(positions) // 4000)), len(positions) - 1))
    overview: Float64[ndarray, "n_overview 3"] = positions[indices]
    rec.log(schema.trajectory_path(source), rr.LineStrips3D(segments(overview), colors=colors[indices[1:]], radii=rr.Radius.ui_points(1.5)), static=True)
    send_trail(f"{run}/{TRAIL}", times, positions, colors, recording=rec)
    send_endpoints(f"{run}/{ENDPOINTS}", int(times[-1]), positions[0], positions[-1], (colors[0], colors[-1]), recording=rec)
