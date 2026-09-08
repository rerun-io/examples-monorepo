"""Replay one reference segment through the Rust core and log it to Rerun.

The tool is the end-to-end wiring of everything else in the package: the frozen
manifest picks the segment and the IMU noise model, the feed decodes it on the
pinned CPU ``gray8`` path, the core consumes framesets and IMU batches, and the
result is logged under paths that mirror the dataset so a run sits beside the
ground truth in one viewer.

``--stage input`` logs only what the estimator is fed — the frames, the IMU and
the ground truth — and builds no core object at all: it is the cheapest way to
look at a segment, and it is what the plumbing was first written against.

``--stage frontend`` runs :class:`slam_rs._core.OpticalFlow` over the same
framesets and hands what it produced to :mod:`slam_rs.frontend_log`, which draws
the keypoints, their trails, the occupancy grid and — on the one recording the
C++ fork dumped its own keypoints from — the two frontends' keypoints side by
side; :attr:`Config.rrd` states how that recording is recognised.

``--stage vio`` is the whole pipeline: :class:`slam_rs._core.Vio` consumes the
IMU and the framesets, :mod:`slam_rs.vio_log` draws the estimated trajectory
against both the ground truth and the basalt C++ reference, the keyframe window,
the landmarks and the per-stage timings, and the run ends with the two ATE
numbers the V2 gate is written against.
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
from slam_rs.catalog_feed import RIG_ENTITY, TIMELINE, CameraCalib, Frameset, ImuStream, LocalSegment, SegmentFeed, open_segment
from slam_rs.frontend_log import FrontendLogger, camera_entity, frontend_blueprint
from slam_rs.reference import ReferenceManifest, ReferenceSegment, flow_config, load_manifest
from slam_rs.reference_bundle import BundleFile
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import Trajectory, ate, coverage, empty_trajectory, read_trajectory, shift_clock, write_trajectory
from slam_rs.vio_log import VioLogger, log_rig, vio_blueprint

SMOKE_SEGMENT: str = "msd-index__MIO_others__MIO10_short_2_panorama"
"""Default segment: the 7.6 s rotation-dominated panorama from the smoke tier."""
IMU_ENTITY: str = f"{RIG_ENTITY}/imu_00"
"""Where the inertial input is drawn, under the rig it belongs to."""
IMAGE_DOWNSCALE: int = 2
"""Images are logged at half resolution: the viewer does not need full-resolution pixels to show what was fed."""
JPEG_QUALITY: int = 85
"""Quality of the full-resolution frames the frontend stage logs; 960x960 grayscale lands around 38 kB."""

Stage: TypeAlias = Literal["input", "frontend", "vio"]
"""How far a replay runs: the estimator's inputs, the optical-flow frontend over them, or the whole pipeline."""


@dataclass(slots=True)
class Config:
    """Replay a reference segment through the slam-rs core."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer, save and headless behaviour."""
    stage: Stage = "input"
    """``input`` logs what the estimator is fed, ``frontend`` runs the optical flow over it, ``vio`` runs the whole pipeline.

    The two stages that track log full-resolution frames, because their keypoints
    are in the pixels of the frame they tracked, not of a downscaled copy of it.
    """
    segment: str = SMOKE_SEGMENT
    """Segment id from ``reference_segments.toml``; also names the IMU parameters used for ``--rrd``."""
    rrd: Path | None = None
    """Base-layer ``.rrd`` to replay instead of the manifest's, keeping ``--segment``'s IMU parameters.

    The C++ overlay is drawn only where the replayed recording's own segment id —
    the one inside the file, not the name the file is filed under — is the one the
    dumps came from. So another recording's frames get no overlay however the
    file is spelled, and this recording's own ``.rrd`` still gets one.
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
    """Log the rig's static geometry so the images sit in the right place in 3D.

    The ``Pinhole`` goes on the ``pinhole`` child, which is the dataset's own
    layout: the images and the keypoints hang under it, so they are the pixels of
    the camera that projects them.
    """
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    log_rig(cameras, RIG_ENTITY, pinhole_child="/pinhole")


@dataclass(slots=True)
class FrontendStage:
    """The optical-flow frontend over one segment, and the Rerun layer it draws.

    One value rather than three: the frontend, its logger and its timings only
    ever exist together, on ``--stage frontend``.
    """

    flow: _core.OpticalFlow
    """The frontend every frameset goes through."""
    logger: FrontendLogger
    """Where its keypoints, trails, occupancy and counters go."""
    elapsed_ms: list[float] = field(default_factory=list)
    """Wall time each ``process`` call took, in frameset order."""

    def run(self, frameset: Frameset) -> None:
        """Track one frameset and log what it produced, at the caller's time cursor.

        Args:
            frameset: The frameset to track.
        """
        started: float = time.monotonic()
        frame: _core.FlowFrame = self.flow.process(frameset.t_ns, frameset.images)
        self.elapsed_ms.append(1e3 * (time.monotonic() - started))
        self.logger.log(frame, self.elapsed_ms[-1])

    def summary(self) -> str:
        """One line on what the stage did, for the end of a replay."""
        if not self.elapsed_ms:
            return "frontend: no frameset reached it"
        return (
            f"frontend: {self.flow.last_keypoint_id} keypoint ids handed out, "
            f"{np.mean(self.elapsed_ms):.1f} ms per frameset "
            f"(median {np.median(self.elapsed_ms):.1f}, max {np.max(self.elapsed_ms):.1f})"
        )


@dataclass(slots=True)
class VioStage:
    """The whole pipeline over one segment, and the Rerun layer it draws.

    The estimator, its logger and its timings only ever exist together, on
    ``--stage vio``.
    """

    lockstep: Lockstep
    """The estimator and the D17 hold, which the V2 gate drives the same way."""
    logger: VioLogger
    """Where the trajectories, the window, the landmarks and the counters go."""

    @property
    def elapsed_ms(self) -> list[float]:
        """Wall time each ``track`` call that tracked took, in frameset order."""
        return self.lockstep.elapsed_ms

    @property
    def pending(self) -> list[Frameset]:
        """Framesets still held for want of the inertial samples that cover them."""
        return self.lockstep.pending

    def run(self, frameset: Frameset) -> None:
        """Track the frameset and everything its samples now cover, and log each one.

        Args:
            frameset: The frameset to track, with the samples since the previous one.
        """
        for held, result in self.lockstep.push(frameset):
            # The rows belong at the frameset's own time, which is the caller's
            # cursor for all but a retried one.
            rr.set_time("video_time", duration=np.timedelta64(held.t_ns, "ns"))
            # Both are present on a frameset that tracked — the snapshot because it
            # measured, the keypoints because the frontend accepted it — so a
            # missing one is a broken invariant, not a rung to skip (D32).
            snapshot: _core.VioSnapshot | None = self.lockstep.vio.snapshot()
            frame: _core.FlowFrame | None = self.lockstep.vio.flow_frame()
            assert snapshot is not None, f"frameset {held.t_ns} tracked without a window snapshot"
            assert frame is not None, f"frameset {held.t_ns} tracked without the keypoints it tracked on"
            self.logger.log(result, snapshot, frame, self.lockstep.elapsed_ms[-1])

    def summary(self) -> str:
        """One line on what the stage did, for the end of a replay.

        The empty case is this stage's own to report: the run that tracked
        nothing is the one whose held framesets most need naming, and a caller
        that guarded the call on ``elapsed_ms`` suppressed exactly that line.
        """
        unresolved: str = ""
        if self.pending:
            unresolved = f", {len(self.pending)} FRAMESETS NEVER COVERED BY THE IMU at {[held.t_ns for held in self.pending]}"
        if not self.elapsed_ms:
            return f"vio: {self.lockstep.imu_samples} IMU samples pushed, nothing tracked{unresolved}"
        return (
            f"vio: {self.lockstep.imu_samples} IMU samples pushed, {len(self.elapsed_ms)} tracked, "
            f"{self.lockstep.retries} retries, {np.mean(self.elapsed_ms):.1f} ms per frameset "
            f"(median {np.median(self.elapsed_ms):.1f}, max {np.max(self.elapsed_ms):.1f})"
            f"{unresolved}"
        )


def _cpp_trajectory(manifest: ReferenceManifest, segment: ReferenceSegment, capture_start_time_ns: int) -> Trajectory:
    """The basalt C++ reference for one segment, moved onto the replay's ``video_time`` clock.

    Empty, with one printed line, when the trajectory is not on this machine: the
    two long-tier segments keep theirs in the reference bundle.
    """
    resolved: BundleFile = manifest.cpp_trajectory(segment)
    if not resolved.available:
        print(f"no C++ trajectory to compare against: {resolved.reason}")
        return empty_trajectory()
    return shift_clock(read_trajectory(resolved.path), -capture_start_time_ns)


def log_imu(imu: ImuStream) -> None:
    """Log one frameset's inertial samples, one column per channel.

    Two ``send_columns`` calls instead of a ``set_time`` and two ``log`` calls per
    sample — about 57 samples a frameset at 1 kHz, so 171 calls become 2, and the
    frameset's inertial rung goes from 0.446 ms to 0.037 ms. The rows are the
    same rows: each timestamp still carries its three components, which is what
    the partition says.

    Args:
        imu: The samples since the previous frameset, on the ``video_time`` clock.
    """
    if not len(imu):
        return
    times: rr.TimeColumn = rr.TimeColumn(TIMELINE, duration=imu.t_ns.astype("timedelta64[ns]"))
    components: int = imu.gyro_rad_s.shape[1]
    for entity, channel in ((f"{IMU_ENTITY}/gyro", imu.gyro_rad_s), (f"{IMU_ENTITY}/accel", imu.accel_m_s2)):
        rr.send_columns(entity, indexes=[times], columns=rr.Scalars.columns(scalars=channel.reshape(-1)).partition([components] * len(imu)))


def _replay(feed: SegmentFeed, config: Config, stage: FrontendStage | VioStage | None) -> int:
    """Log every frameset's inputs and drive whichever stage was built over them.

    Args:
        feed: Open segment feed.
        config: Parsed CLI options.
        stage: The frontend or the whole pipeline, or None for ``--stage input``.

    Returns:
        Framesets replayed. Everything a stage produced is on the stage itself.
    """
    started: float = time.monotonic()
    replayed: int = 0
    frameset: Frameset
    for frameset in feed.framesets():
        if config.max_framesets is not None and replayed >= config.max_framesets:
            break
        replayed += 1

        # The samples are what the estimator is fed, so they are logged in every
        # stage whether or not one consumes them.
        log_imu(frameset.imu)
        rr.set_time("video_time", duration=np.timedelta64(frameset.t_ns, "ns"))

        for camera, image in zip(feed.cameras, frameset.images, strict=True):
            entity: str = f"{camera_entity(camera.index)}/image"
            if config.stage == "input":
                small: UInt8[ndarray, "h w"] = np.ascontiguousarray(image[::IMAGE_DOWNSCALE, ::IMAGE_DOWNSCALE])
                rr.log(entity, rr.Image(small, color_model="L"))
            else:
                # Full resolution, or the keypoints would sit two pixels off the
                # corner they were computed on; JPEG keeps a whole segment small.
                # The encode costs 3.9 ms a frameset on the replay thread and is
                # deliberate: no lane that reports a wall time comes through here,
                # because the V2 gate drives the feed and `Vio` itself with
                # nothing logged.
                rr.log(entity, rr.Image(image, color_model="L").compress(jpeg_quality=JPEG_QUALITY))

        if frameset.ground_truth is not None:
            pose_wxyz: Float64[ndarray, " 7"] = frameset.ground_truth
            rr.log(
                "/world/rig_00",
                rr.Transform3D(translation=pose_wxyz[0:3], quaternion=rr.Quaternion(xyzw=np.roll(pose_wxyz[3:7], -1))),
            )

        if stage is not None:
            stage.run(frameset)

    elapsed: float = time.monotonic() - started
    print(f"{replayed} framesets in {elapsed:.1f} s ({replayed / max(elapsed, 1e-9):.1f} fps)")
    if stage is not None:
        print(stage.summary())
    return replayed


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
        stage: FrontendStage | VioStage | None = None
        if config.stage == "frontend":
            stage = FrontendStage(
                flow=_core.OpticalFlow(_core.Calibration.from_catalog(feed.cameras, feed.imu), flow_config(manifest, segment)),
                logger=FrontendLogger(len(feed.cameras), feed.segment_id),
            )
            rr.send_blueprint(frontend_blueprint(feed.cameras))
        elif config.stage == "vio":
            truth: Trajectory | None = feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1]))
            stage = VioStage(
                lockstep=Lockstep(vio=_core.Vio(_core.Calibration.from_catalog(feed.cameras, feed.imu), flow_config(manifest, segment))),
                logger=VioLogger(
                    cameras=feed.cameras,
                    ground_truth=truth if truth is not None else empty_trajectory(),
                    cpp=_cpp_trajectory(manifest, segment, feed.capture_start_time_ns),
                    frame_t_ns=feed.frame_t_ns,
                ),
            )
            rr.send_blueprint(vio_blueprint(feed.cameras))
        _replay(feed, config, stage)
        if not isinstance(stage, VioStage):
            return
        if stage.pending:
            # Every frameset either produced a pose or is still held; a held one
            # at the end of the segment is a lost frameset, not a count to print.
            raise SystemExit(f"{len(stage.pending)} framesets never got the inertial samples that cover them")

        # Exports carry the absolute device clock, the one every basalt CSV and
        # every gt.csv sidecar uses. Writing video_time here would produce a file
        # that associates with none of them.
        estimate: Trajectory = stage.logger.estimated()
        write_trajectory(output_csv, shift_clock(estimate, feed.capture_start_time_ns))
        print(f"{len(estimate)} tracked poses -> {output_csv} (absolute ns)")
        if len(estimate) == 0:
            print("no ATE: the estimator reported no tracked pose")
            return
        for name, reference in (("ground truth", stage.logger.ground_truth), ("basalt C++", stage.logger.cpp)):
            if len(reference) == 0:
                continue
            print(f"vs {name}, {coverage(reference, estimate):.1%} of its span covered")
            print(ate(estimate, reference).summary())
