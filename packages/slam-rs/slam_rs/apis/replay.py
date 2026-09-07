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
from slam_rs.catalog_feed import RIG_ENTITY, CameraCalib, Frameset, LocalSegment, SegmentFeed, open_segment
from slam_rs.frontend_log import FrontendLogger, camera_entity, frontend_blueprint
from slam_rs.reference import ReferenceManifest, ReferenceSegment, load_manifest
from slam_rs.reference_bundle import BundleFile
from slam_rs.trajectory import AteResult, Trajectory, ate, coverage, empty_trajectory, read_trajectory, shift_clock, write_trajectory
from slam_rs.vio_log import VioLogger, vio_blueprint

SMOKE_SEGMENT: str = "msd-index__MIO_others__MIO10_short_2_panorama"
"""Default segment: the 7.6 s rotation-dominated panorama from the smoke tier."""
RUN_ENTITY: str = "/world/runs/slam_rs"
"""Where this run's estimate goes, beside the dataset's own ``/world/runs/gt``."""
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

    vio: _core.Vio
    """The estimator every frameset and every inertial sample goes through."""
    logger: VioLogger
    """Where the trajectories, the window, the landmarks and the counters go."""
    elapsed_ms: list[float] = field(default_factory=list)
    """Wall time each ``track`` call took, in frameset order."""
    statuses: dict[str, int] = field(default_factory=dict)
    """How many times ``track`` answered each status, retries included."""
    imu_samples: int = 0
    """Inertial samples pushed so far."""
    pending: list[Frameset] = field(default_factory=list)
    """Framesets refused for want of IMU, oldest first, waiting for the samples that cover them."""

    def run(self, frameset: Frameset) -> None:
        """Push the frameset's inertial samples, then track everything they now cover.

        A frameset the estimator refuses is **held, not dropped** (D17): the next
        frameset's batch runs one sample past its own frame time and therefore
        past this one's, so the held frameset tracks then — and before the
        frameset whose samples unblocked it, because time order is the
        trajectory. Nothing moved on the refusal, so the retry produces the pose
        a run that had the samples all along would have produced.

        Args:
            frameset: The frameset to track, with the samples since the previous one.
        """
        if len(frameset.imu):
            # The feed already hands over the samples since the previous frameset,
            # running one past this frame time: a backend that integrates up to the
            # frame and blocks until it can deadlocks on the first frameset otherwise.
            self.vio.push_imu_batch(
                frameset.imu.t_ns,
                np.ascontiguousarray(frameset.imu.gyro_rad_s),
                np.ascontiguousarray(frameset.imu.accel_m_s2),
            )
            self.imu_samples += len(frameset.imu)
        self.pending.append(frameset)
        while self.pending:
            held: Frameset = self.pending[0]
            started: float = time.monotonic()
            result: _core.VioResult = self.vio.track(held.t_ns, held.images)
            elapsed_ms: float = 1e3 * (time.monotonic() - started)
            # A PyO3 enum has no ``name`` and is unhashable (see ``_core.pyi``), so the
            # repr is both the only name it has and the only thing that keys a dict.
            status_name: str = str(result.status)
            self.statuses[status_name] = self.statuses.get(status_name, 0) + 1
            if result.status != _core.VioStatus.Tracking:
                return
            self.pending.pop(0)
            self.elapsed_ms.append(elapsed_ms)
            # The rows belong at the frameset's own time, which is the caller's
            # cursor for all but a retried one.
            rr.set_time("video_time", duration=np.timedelta64(held.t_ns, "ns"))
            # Both are present on a frameset that tracked — the snapshot because it
            # measured, the keypoints because the frontend accepted it — and the
            # checks are what say so to the typechecker.
            snapshot: _core.VioSnapshot | None = self.vio.snapshot()
            frame: _core.FlowFrame | None = self.vio.flow_frame()
            if snapshot is not None and frame is not None:
                self.logger.log(result, snapshot, frame, elapsed_ms)

    def summary(self) -> str:
        """One line on what the stage did, for the end of a replay."""
        unresolved: str = ""
        if self.pending:
            unresolved = f", {len(self.pending)} FRAMESETS NEVER COVERED BY THE IMU at {[held.t_ns for held in self.pending]}"
        return (
            f"vio: {self.imu_samples} IMU samples pushed, statuses {self.statuses}, "
            f"{np.mean(self.elapsed_ms):.1f} ms per frameset "
            f"(median {np.median(self.elapsed_ms):.1f}, max {np.max(self.elapsed_ms):.1f})"
            f"{unresolved}"
        )


def _frontend_stage(feed: SegmentFeed, segment: ReferenceSegment) -> FrontendStage:
    """Build the optical-flow stage for one segment."""
    return FrontendStage(
        flow=_core.OpticalFlow(_core.Calibration.from_catalog(feed.cameras, feed.imu), _flow_config(segment)),
        logger=FrontendLogger(len(feed.cameras), feed.segment_id),
    )


def _vio_stage(feed: SegmentFeed, segment: ReferenceSegment, ground_truth: Trajectory, cpp: Trajectory) -> VioStage:
    """Build the whole pipeline for one segment, with both reference trajectories to draw against."""
    return VioStage(
        vio=_core.Vio(_core.Calibration.from_catalog(feed.cameras, feed.imu), _flow_config(segment)),
        logger=VioLogger(cameras=feed.cameras, ground_truth=ground_truth, cpp=cpp),
    )


def _flow_config(segment: ReferenceSegment) -> _core.VioConfig:
    """basalt's defaults with the one field the manifest freezes per device.

    Every other field of basalt's shipped configs is already the default the C++
    constructor sets; the image safe radius is a property of the device — 472 on
    Index, 340 on G2 — and the manifest carries it per segment.
    """
    config: _core.VioConfig = _core.VioConfig()
    config.optical_flow_image_safe_radius = segment.reference.optical_flow_image_safe_radius
    return config


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
        for sample in range(len(frameset.imu)):
            rr.set_time("video_time", duration=np.timedelta64(int(frameset.imu.t_ns[sample]), "ns"))
            rr.log("/world/rig_00/imu_00/gyro", rr.Scalars(frameset.imu.gyro_rad_s[sample]))
            rr.log("/world/rig_00/imu_00/accel", rr.Scalars(frameset.imu.accel_m_s2[sample]))
        rr.set_time("video_time", duration=np.timedelta64(frameset.t_ns, "ns"))

        for camera, image in zip(feed.cameras, frameset.images, strict=True):
            entity: str = f"{camera_entity(camera.index)}/image"
            if config.stage == "input":
                small: UInt8[ndarray, "h w"] = np.ascontiguousarray(image[::IMAGE_DOWNSCALE, ::IMAGE_DOWNSCALE])
                rr.log(entity, rr.Image(small, color_model="L"))
            else:
                # Full resolution, or the keypoints would sit two pixels off the
                # corner they were computed on; JPEG keeps a whole segment small.
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
    if stage is not None and stage.elapsed_ms:
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
            stage = _frontend_stage(feed, segment)
            rr.send_blueprint(frontend_blueprint(feed.cameras))
        elif config.stage == "vio":
            truth: Trajectory | None = feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1]))
            stage = _vio_stage(
                feed,
                segment,
                ground_truth=truth if truth is not None else empty_trajectory(),
                cpp=_cpp_trajectory(manifest, segment, feed.capture_start_time_ns),
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
        # The association is driven by the estimate, as the reference manifest's
        # own C++ numbers are: every estimate pose takes the nearest reference
        # pose within the tolerance, so a 917 Hz ground truth does not weight the
        # metric by its own density.
        for name, reference in (("ground truth", stage.logger.ground_truth), ("basalt C++", stage.logger.cpp)):
            if len(reference) == 0:
                continue
            result: AteResult = ate(estimate, reference)
            print(
                f"vs {name}: ATE rmse {result.rmse_m * 100:.2f} cm, max {result.max_m * 100:.2f} cm, "
                f"median {result.median_m * 100:.2f} cm over {result.n_associated} of {len(estimate)} poses; "
                f"{coverage(reference, estimate):.1%} of its span covered"
            )
