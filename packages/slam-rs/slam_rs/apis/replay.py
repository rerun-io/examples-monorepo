"""Replay one reference segment through the Rust core and log it to Rerun.

The tool is the end-to-end wiring of everything else in the package: the frozen
manifest picks the segment and the IMU noise model, the feed decodes it on the
pinned CPU ``gray8`` path, the core consumes framesets and IMU batches, and the
result is logged under paths that mirror the dataset so a run sits beside the
ground truth in one viewer.

The estimator is still a stub, so ``--stage input`` reports ``NotInitialised`` or
``NeedMoreImu`` and writes no pose. That is the point of running it now: the
plumbing, the timestamps and the ATE report are exercised before the estimator
exists, and the day it starts tracking nothing else has to change.

``--stage frontend`` is the stage that does produce something. It runs
:class:`slam_rs._core.OpticalFlow` over the same framesets and hands what it
produced to :mod:`slam_rs.frontend_log`, which draws the keypoints, their trails,
the occupancy grid and — where the C++ fork dumped its own, **for this segment** —
the two frontends' keypoints side by side. The committed dumps are the smoke
segment's, so any other ``--segment`` gets no overlay rather than the smoke
segment's keypoints over its pixels.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
from jaxtyping import Float64, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import _core
from slam_rs.catalog_feed import RIG_ENTITY, CameraCalib, Frameset, LocalSegment, SegmentFeed, open_segment
from slam_rs.frontend_log import FrontendLogger, camera_entity, frontend_blueprint
from slam_rs.reference import ReferenceManifest, ReferenceSegment, load_manifest
from slam_rs.trajectory import AteResult, Trajectory, ate, coverage, shift_clock, write_trajectory

SMOKE_SEGMENT: str = "msd-index__MIO_others__MIO10_short_2_panorama"
"""Default segment: the 7.6 s rotation-dominated panorama from the smoke tier."""
RUN_ENTITY: str = "/world/runs/slam_rs"
"""Where this run's estimate goes, beside the dataset's own ``/world/runs/gt``."""
IMAGE_DOWNSCALE: int = 2
"""Images are logged at half resolution: the viewer does not need full-resolution pixels to show what was fed."""
JPEG_QUALITY: int = 85
"""Quality of the full-resolution frames the frontend stage logs; 960x960 grayscale lands around 38 kB."""

Stage: TypeAlias = Literal["input", "frontend"]
"""How far a replay runs: the estimator's inputs only, or the optical-flow frontend over them."""


@dataclass(slots=True)
class Config:
    """Replay a reference segment through the slam-rs core."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer, save and headless behaviour."""
    stage: Stage = "input"
    """``input`` logs what the estimator is fed; ``frontend`` also runs the optical flow over it.

    The frontend stage logs full-resolution frames, because its keypoints are in
    the pixels of the frame it tracked, not of a downscaled copy of it.
    """
    segment: str = SMOKE_SEGMENT
    """Segment id from ``reference_segments.toml``; also names the IMU parameters used for ``--rrd``."""
    rrd: Path | None = None
    """Base-layer ``.rrd`` to replay instead of the manifest's, keeping ``--segment``'s IMU parameters.

    The frames are then another recording's, so the frontend stage draws no C++
    overlay on them: the path is what the dumps are matched against, and no dump
    directory names a path.
    """
    gt_rrd: Path | None = None
    """Ground-truth ``.rrd`` for ``--rrd``; the manifest's own path is used when neither is given."""
    max_framesets: int | None = None
    """Stop after this many framesets; None replays the whole segment."""
    frame_stride: int = 1
    """Replay every n-th frameset. Every frame is still decoded: decimated AV1 decode is unreliable."""
    window_s: float = 60.0
    """Longest time window fetched from the catalog in one round trip."""
    output_csv: Path | None = None
    """Where the estimated trajectory is written; defaults to ``data/<segment>/slam_rs.csv``."""


def _log_calibration(cameras: tuple[CameraCalib, ...]) -> None:
    """Log the rig's static geometry so the images sit in the right place in 3D."""
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    for camera in cameras:
        rr.log(
            f"{RIG_ENTITY}/cam_{camera.index:02d}",
            rr.Transform3D(translation=camera.imu_T_cam[:3, 3], mat3x3=camera.imu_T_cam[:3, :3]),
            static=True,
        )
        image_from_camera: Float64[ndarray, "3 3"] = np.array(
            [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        rr.log(
            camera_entity(camera.index),
            rr.Pinhole(image_from_camera=image_from_camera, resolution=[camera.width, camera.height], camera_xyz=rr.ViewCoordinates.RDF),
            static=True,
        )


@dataclass(slots=True, frozen=True)
class ReplayOutcome:
    """What one replay produced, all on the ``video_time`` clock."""

    estimate: Trajectory
    """Poses the core reported while tracking; possibly empty."""
    ground_truth: Trajectory
    """The nearest ground-truth pose per replayed frameset, row-aligned in time with the framesets."""
    imu_samples: int
    """Inertial samples pushed into the core."""
    framesets: int
    """Framesets replayed."""


def replayed_identity(rrd: Path | None, segment_id: str) -> str:
    """What the frontend's C++ overlay matches its dumps against.

    ``--rrd`` replays another recording's frames under ``--segment``'s IMU
    parameters, so the manifest's segment id would attach that segment's dumps to
    pixels the C++ never saw. The path stands in as the identity instead, and no
    dump directory can name one.

    Args:
        rrd: The ``--rrd`` override, or None when the manifest's own layer is replayed.
        segment_id: Manifest segment id.

    Returns:
        The segment id, or the overriding path as text.
    """
    return segment_id if rrd is None else str(rrd)


def _replay(feed: SegmentFeed, config: Config, segment: ReferenceSegment) -> ReplayOutcome:
    """Drive the core over the feed, logging inputs, the frontend and any tracked pose.

    Args:
        feed: Open segment feed.
        config: Parsed CLI options.
        segment: Manifest entry for the segment being replayed.

    Returns:
        The estimate, the ground truth sampled at the frameset times, and counts.
    """
    vio: _core.Vio = _core.Vio(camera_count=len(feed.cameras), min_imu_samples=1)
    frontend: _core.OpticalFlow | None = None
    logger: FrontendLogger | None = None
    if config.stage == "frontend":
        # Every frontend field of basalt's shipped configs is already the default
        # the C++ constructor sets; the image safe radius is the one exception,
        # and it is a property of the device the manifest freezes per segment.
        flow_config: _core.VioConfig = _core.VioConfig()
        flow_config.optical_flow_image_safe_radius = segment.reference.optical_flow_image_safe_radius
        frontend = _core.OpticalFlow(_core.Calibration.from_catalog(feed.cameras, feed.imu), flow_config)
        logger = FrontendLogger.create(len(feed.cameras), replayed_identity(config.rrd, segment.segment_id))
        rr.send_blueprint(frontend_blueprint(feed.cameras))
    frontend_ms: list[float] = []
    statuses: dict[str, int] = {}
    pose_t_ns: list[int] = []
    positions: list[Float64[ndarray, " 3"]] = []
    quaternions: list[Float64[ndarray, " 4"]] = []
    gt_t_ns: list[int] = []
    gt_positions: list[Float64[ndarray, " 3"]] = []
    gt_quaternions: list[Float64[ndarray, " 4"]] = []
    started: float = time.monotonic()
    replayed: int = 0
    pushed: int = 0
    frameset: Frameset
    for frameset in feed.framesets():
        if config.max_framesets is not None and replayed >= config.max_framesets:
            break
        replayed += 1

        # The feed already hands over the samples since the previous frameset,
        # running one past this frame time: a backend that integrates up to the
        # frame and blocks until it can deadlocks on the first frameset otherwise.
        if len(frameset.imu):
            vio.push_imu_batch(
                frameset.imu.t_ns,
                np.ascontiguousarray(frameset.imu.gyro_rad_s),
                np.ascontiguousarray(frameset.imu.accel_m_s2),
            )
            pushed += len(frameset.imu)
            for sample in range(len(frameset.imu)):
                rr.set_time("video_time", duration=np.timedelta64(int(frameset.imu.t_ns[sample]), "ns"))
                rr.log("/world/rig_00/imu_00/gyro", rr.Scalars(frameset.imu.gyro_rad_s[sample]))
                rr.log("/world/rig_00/imu_00/accel", rr.Scalars(frameset.imu.accel_m_s2[sample]))
        rr.set_time("video_time", duration=np.timedelta64(frameset.t_ns, "ns"))

        for camera, image in zip(feed.cameras, frameset.images, strict=True):
            entity: str = f"{camera_entity(camera.index)}/image"
            if frontend is not None:
                # Full resolution, or the keypoints would sit two pixels off the
                # corner they were computed on; JPEG keeps a whole segment small.
                rr.log(entity, rr.Image(image, color_model="L").compress(jpeg_quality=JPEG_QUALITY))
            else:
                small: UInt8[ndarray, "h w"] = np.ascontiguousarray(image[::IMAGE_DOWNSCALE, ::IMAGE_DOWNSCALE])
                rr.log(entity, rr.Image(small, color_model="L"))

        if frameset.ground_truth is not None:
            pose_wxyz: Float64[ndarray, " 7"] = frameset.ground_truth
            rr.log(
                "/world/rig_00",
                rr.Transform3D(translation=pose_wxyz[0:3], quaternion=rr.Quaternion(xyzw=np.roll(pose_wxyz[3:7], -1))),
            )
            gt_t_ns.append(frameset.t_ns)
            gt_positions.append(pose_wxyz[0:3].copy())
            gt_quaternions.append(pose_wxyz[3:7].copy())

        if frontend is not None and logger is not None:
            started_frame: float = time.monotonic()
            frame: _core.FlowFrame = frontend.process(frameset.t_ns, frameset.images)
            frontend_ms.append(1e3 * (time.monotonic() - started_frame))
            logger.log(frame, frontend_ms[-1])
            continue

        result: _core.VioResult = vio.track(frameset.t_ns, frameset.images)
        # A PyO3 enum has no ``name`` and is unhashable (see ``_core.pyi``), so the
        # repr is both the only name it has and the only thing that keys a dict.
        status_name: str = str(result.status)
        statuses[status_name] = statuses.get(status_name, 0) + 1
        if result.status == _core.VioStatus.Tracking:
            pose: Float64[ndarray, " 7"] = result.world_from_rig
            rr.log(f"{RUN_ENTITY}/rig", rr.Transform3D(translation=pose[0:3], quaternion=rr.Quaternion(xyzw=pose[3:7])))
            rr.log(f"{RUN_ENTITY}/velocity", rr.Scalars(result.velocity))
            pose_t_ns.append(result.t_ns)
            positions.append(pose[0:3].copy())
            quaternions.append(np.roll(pose[3:7], 1).copy())

    elapsed: float = time.monotonic() - started
    print(f"{replayed} framesets in {elapsed:.1f} s ({replayed / max(elapsed, 1e-9):.1f} fps), statuses: {statuses}")
    if frontend is not None and frontend_ms:
        print(
            f"frontend: {frontend.last_keypoint_id} keypoint ids handed out, "
            f"{np.mean(frontend_ms):.1f} ms per frameset (median {np.median(frontend_ms):.1f}, max {np.max(frontend_ms):.1f})"
        )
    return ReplayOutcome(
        estimate=Trajectory(
            t_ns=np.array(pose_t_ns, dtype=np.int64),
            position_m=np.array(positions, dtype=np.float64).reshape(-1, 3),
            quaternion_wxyz=np.array(quaternions, dtype=np.float64).reshape(-1, 4),
        ),
        ground_truth=Trajectory(
            t_ns=np.array(gt_t_ns, dtype=np.int64),
            position_m=np.array(gt_positions, dtype=np.float64).reshape(-1, 3),
            quaternion_wxyz=np.array(gt_quaternions, dtype=np.float64).reshape(-1, 4),
        ),
        imu_samples=pushed,
        framesets=replayed,
    )


def main(config: Config) -> None:
    """Replay one segment, log it, write the trajectory and report the error.

    Args:
        config: Parsed CLI options.
    """
    manifest: ReferenceManifest = load_manifest()
    segment: ReferenceSegment = manifest.by_id(config.segment)
    source: LocalSegment = LocalSegment(
        base_rrd=config.rrd if config.rrd is not None else segment.base_path,
        gt_rrd=config.gt_rrd if config.gt_rrd is not None else (None if config.rrd is not None else segment.gt_path),
    )
    output_csv: Path = config.output_csv if config.output_csv is not None else Path("data") / segment.segment_id / "slam_rs.csv"
    print(f"replaying {segment.segment_id} ({segment.tier} tier) from {source.base_rrd}")

    with open_segment(source, segment.imu, frame_stride=config.frame_stride, window_s=config.window_s) as feed:
        print(
            f"{len(feed.cameras)} cameras, {len(feed.frame_t_ns)} framesets, ground truth "
            f"{'attached' if feed.has_ground_truth else 'absent'}, clock offset {feed.capture_start_time_ns} ns"
        )
        _log_calibration(feed.cameras)
        outcome: ReplayOutcome = _replay(feed, config, segment)
        print(f"{outcome.imu_samples} IMU samples pushed, {len(outcome.ground_truth)} ground-truth poses sampled")

        # Exports carry the absolute device clock, the one every basalt CSV and
        # every gt.csv sidecar uses. Writing video_time here would produce a file
        # that associates with none of them.
        write_trajectory(output_csv, shift_clock(outcome.estimate, feed.capture_start_time_ns))
        print(f"{len(outcome.estimate)} tracked poses -> {output_csv} (absolute ns)")
        if len(outcome.estimate) > 0 and len(outcome.ground_truth) > 0:
            result: AteResult = ate(outcome.ground_truth, outcome.estimate)
            print(result.summary())
            print(f"coverage: {coverage(outcome.ground_truth, outcome.estimate):.1%}")
        else:
            print("no ATE: the core reported no tracked poses (expected until the estimator lands)")
