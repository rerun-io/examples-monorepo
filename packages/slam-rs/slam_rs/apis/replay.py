"""Replay catalog or local recordings through the Rust core and log ground-truth comparisons."""

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
    CatalogSegment,
    Frameset,
    LocalSegment,
    SegmentFeed,
    SegmentSource,
    open_segment,
)
from slam_rs.frontend_log import FrontendLogger, frontend_blueprint
from slam_rs.reference import SMOKE_SEGMENTS, ImuParameters, ReferenceManifest, ReferenceSegment, load_manifest, resolved_flow_config
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import Trajectory, ate, coverage, shift_clock, write_trajectory
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
    stage: Stage = "input"
    """``input`` logs what the estimator is fed, ``frontend`` runs the optical flow over it, ``vio`` runs the whole pipeline.

    The two stages that track log full-resolution frames, because their keypoints
    are in the pixels of the frame they tracked, not of a downscaled copy of it.
    """
    segment: str = SMOKE_SEGMENT
    """Segment id from ``gate.toml``; also names the IMU parameters used for ``--rrd``."""
    rrd: Path | None = None
    """Local base recording; retains the selected dataset configuration and IMU model."""
    gt_rrd: Path | None = None
    """Optional local ground-truth recording; requires --rrd."""
    catalog: str | None = None
    """Catalog URL override; defaults to the manifest. Exclusive with local recording files."""
    max_framesets: int | None = None
    """Stop after this many framesets; None replays the whole segment."""
    frame_stride: int = 1
    """Replay every n-th frameset. Every frame is still decoded: decimated AV1 decode is unreliable."""
    window_s: float = DEFAULT_WINDOW_S
    """Longest time window fetched from the catalog in one round trip."""
    output_csv: Path | None = None
    """Where the estimated trajectory is written; defaults to ``data/<segment>/slam_rs.csv``."""
    profile: Literal["reference", "fast"] = "fast"
    """Config overlay applied before tracking."""
    gpu: bool = False
    """Run the frontend's pyramid, patch build and KLT tracker on the GPU through CubeCL.

    The default is the CPU port. A core built without a GPU cargo feature refuses this
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
    manifest: ReferenceManifest = load_manifest()
    listed: ReferenceSegment | None = next((s for s in manifest.segments if s.segment_id == config.segment), None)
    dataset_name: str = listed.dataset_name if listed is not None else config.segment.split("__")[0]
    vio_config: _core.VioConfig
    imu: ImuParameters
    if listed is not None:
        vio_config, _config_text = resolved_flow_config(manifest, listed, profile=config.profile)
        imu = manifest.dataset(listed.dataset_name).imu
    else:
        vio_config = _core.VioConfig.from_json(manifest.vio_config_text(dataset_name, profile=config.profile))  # refuses an unknown dataset
        imu = manifest.dataset(dataset_name).imu
    source: SegmentSource
    origin: str
    if config.rrd is not None:
        if config.catalog is not None:
            raise ValueError("--catalog and --rrd name two sources for one replay")
        source = LocalSegment(config.rrd, config.gt_rrd)
        origin = str(config.rrd)
    else:
        if config.gt_rrd is not None:
            raise ValueError("--gt-rrd requires --rrd")
        origin = config.catalog or manifest.catalog_url
        source = CatalogSegment(origin, dataset_name, config.segment)
    output_csv: Path = config.output_csv if config.output_csv is not None else Path("data") / config.segment / "slam_rs.csv"
    print(
        f"replaying {config.segment} ({f'{listed.tier} tier' if listed is not None else 'not in the reference set: ground truth only'}) from {origin}"
    )

    with open_segment(source, imu, frame_stride=config.frame_stride, window_s=config.window_s) as feed:
        print(
            f"{len(feed.cameras)} cameras, {len(feed.frame_t_ns)} framesets, ground truth "
            f"{'attached' if feed.has_ground_truth else 'absent'}, clock offset {feed.capture_start_time_ns} ns"
        )
        if not feed.has_ground_truth:
            print("ground truth absent, not scored")
        log_calibration(feed.cameras)
        stage: FrontendStage | VioStage | None = None
        if config.stage == "frontend":
            stage = FrontendStage(
                flow=_core.OpticalFlow(_core.Calibration.from_catalog(feed.cameras, feed.imu), vio_config),
                logger=FrontendLogger(len(feed.cameras)),
            )
            rr.send_blueprint(frontend_blueprint(feed.cameras))
        elif config.stage == "vio":
            truth: Trajectory = feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1]))
            # Everything this constructor refuses is the caller's own request —
            # a configuration this port does not run, or ``--gpu`` on a host
            # with no driver, no device or no adapter — and the shim
            # (:func:`slam_rs.apis.run`) is what turns it into one sentence and a
            # non-zero exit rather than a traceback through the feed.
            vio: _core.Vio = _core.Vio(_core.Calibration.from_catalog(feed.cameras, feed.imu), vio_config, gpu=config.gpu)
            stage = VioStage(
                lockstep=Lockstep(vio=vio),
                logger=VioLogger(
                    cameras=feed.cameras,
                    ground_truth=truth,
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

        estimate: Trajectory = stage.logger.estimated()
        write_trajectory(output_csv, shift_clock(estimate, feed.export_offset_ns))
        print(f"{len(estimate)} tracked poses -> {output_csv} (absolute ns)")
        if len(estimate) == 0:
            print("no ATE: the estimator reported no tracked pose")
            return
        reference: Trajectory = stage.logger.ground_truth
        if len(reference) == 0:
            return
        print(f"vs ground truth, {coverage(reference, estimate):.1%} of its span covered")
        print(ate(estimate, reference).summary())
