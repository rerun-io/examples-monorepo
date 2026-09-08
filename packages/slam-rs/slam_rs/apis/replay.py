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
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import _core
from slam_rs.catalog_feed import (
    DEFAULT_WINDOW_S,
    Frameset,
    LocalSegment,
    SegmentFeed,
    open_segment,
)
from slam_rs.frontend_log import FrontendLogger, frontend_blueprint
from slam_rs.reference import SMOKE_SEGMENTS, ReferenceManifest, ReferenceSegment, flow_config, load_manifest
from slam_rs.reference_bundle import BundleFile
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import Trajectory, ate, coverage, empty_trajectory, read_trajectory, shift_clock, write_trajectory
from slam_rs.vio_log import FrameMode, VioLogger, VioStage, log_calibration, log_frameset_inputs, vio_blueprint

SMOKE_SEGMENT: str = SMOKE_SEGMENTS[1]
"""Default segment: the 7.6 s rotation-dominated panorama from the smoke tier."""

Stage: TypeAlias = Literal["input", "frontend", "vio"]
"""How far a replay runs: the estimator's inputs, the optical-flow frontend over them, or the whole pipeline."""


@dataclass(slots=True)
class Config:
    """Replay a reference segment through the slam-rs core."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer, save and headless behaviour."""
    artifact_root: Path | None = None
    """Read every recording and sidecar from one directory per segment; see :func:`slam_rs.reference.relocate`."""
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
    window_s: float = DEFAULT_WINDOW_S
    """Longest time window fetched from the catalog in one round trip."""
    output_csv: Path | None = None
    """Where the estimated trajectory is written; defaults to ``data/<segment>/slam_rs.csv``."""
    gpu: bool = False
    """Run the frontend's pyramid, patch build and KLT tracker on the GPU through CubeCL.

    The default is the CPU port, which is what every reference number was
    produced on. A core built without the ``gpu`` cargo feature refuses this
    rather than quietly running on the CPU, and so does a host with no usable
    GPU: the run stops with one sentence naming what is absent.
    """


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


def _cpp_trajectory(manifest: ReferenceManifest, segment: ReferenceSegment, capture_start_time_ns: int, replayed_segment_id: str) -> Trajectory:
    """The basalt C++ reference for one segment, moved onto the replay's ``video_time`` clock.

    Empty, with one printed line, when the trajectory is not on this machine (the
    two long-tier segments keep theirs in the reference bundle), or when ``--rrd``
    replays another segment than the manifest entry: a C++ run of one clip says
    nothing about another, and associating the two only fails.
    """
    if replayed_segment_id != segment.segment_id:
        print(f"no C++ comparison: the recording is {replayed_segment_id}, the C++ run is {segment.segment_id}")
        return empty_trajectory()
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
    # `--max-framesets` is a count and the feed reads by time; the feed is what
    # converts one to the other, so this loop and `tracking._drive` cannot
    # disagree about which frameset a count ends on.
    stop_ns: int | None = feed.stop_ns_after(config.max_framesets)
    # The stage that tracks needs the pixels its keypoints were computed on.
    mode: FrameMode = "downscaled" if config.stage == "input" else "jpeg"
    frameset: Frameset
    for frameset in feed.framesets(stop_ns):
        if config.max_framesets is not None and replayed >= config.max_framesets:
            break
        replayed += 1
        log_frameset_inputs(feed, frameset, mode)
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
    manifest: ReferenceManifest = load_manifest(artifact_root=config.artifact_root)
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
        log_calibration(feed.cameras)
        stage: FrontendStage | VioStage | None = None
        if config.stage == "frontend":
            stage = FrontendStage(
                flow=_core.OpticalFlow(_core.Calibration.from_catalog(feed.cameras, feed.imu), flow_config(manifest, segment)),
                logger=FrontendLogger(len(feed.cameras), feed.segment_id),
            )
            rr.send_blueprint(frontend_blueprint(feed.cameras))
        elif config.stage == "vio":
            truth: Trajectory = feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1]))
            # Everything this constructor refuses is the caller's own request —
            # a configuration this port does not run, or ``--gpu`` on a host
            # with no driver, no device or no adapter — and the shim
            # (:func:`slam_rs.apis.run`) is what turns it into one sentence and a
            # non-zero exit rather than a traceback through the feed.
            vio: _core.Vio = _core.Vio(_core.Calibration.from_catalog(feed.cameras, feed.imu), flow_config(manifest, segment), gpu=config.gpu)
            stage = VioStage(
                lockstep=Lockstep(vio=vio),
                logger=VioLogger(
                    cameras=feed.cameras,
                    ground_truth=truth,
                    cpp=_cpp_trajectory(manifest, segment, feed.capture_start_time_ns, feed.segment_id),
                    frame_t_ns=feed.frame_t_ns,
                ),
            )
            rr.send_blueprint(vio_blueprint(feed.cameras))
        _replay(feed, config, stage)
        if not isinstance(stage, VioStage):
            return
        # The per-frameset segments show where the run had got to; these show
        # where it went, at every cursor and for one copy of each path.
        stage.logger.log_complete_paths()
        stage.refuse_lost_framesets()

        # Exports carry the absolute device clock, the one every basalt CSV and
        # every gt.csv sidecar uses. Writing video_time here would produce a file
        # that associates with none of them. How far that is from video_time is
        # the recording's own fact (`SegmentFeed.export_offset_ns`), which is why
        # the RoboCap probe can share this rule instead of stating its own.
        estimate: Trajectory = stage.logger.estimated()
        write_trajectory(output_csv, shift_clock(estimate, feed.export_offset_ns))
        print(f"{len(estimate)} tracked poses -> {output_csv} (absolute ns)")
        if len(estimate) == 0:
            print("no ATE: the estimator reported no tracked pose")
            return
        for name, reference in (("ground truth", stage.logger.ground_truth), ("basalt C++", stage.logger.cpp)):
            if len(reference) == 0:
                continue
            print(f"vs {name}, {coverage(reference, estimate):.1%} of its span covered")
            print(ate(estimate, reference).summary())
