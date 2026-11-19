"""Replay catalog or local recordings through the Rust core and log ground-truth comparisons."""

import dataclasses
import time
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.stage_times import StageTimes

from slam_rs import BackendRequest, GbaRequest, _core
from slam_rs.catalog_feed import (
    DEFAULT_WINDOW_S,
    CatalogSegment,
    DecodeRequest,
    Frameset,
    LocalSegment,
    SegmentFeed,
    SegmentSource,
    dataset_rig,
    open_segment,
    select_decode_device,
)
from slam_rs.config import SlamConfig, load_slam_config
from slam_rs.frontend_log import FrontendLogger, frontend_blueprint
from slam_rs.gba_log import keyframe_trajectory, log_map_result
from slam_rs.reference import SMOKE_SEGMENTS, Benchmarks, ReferenceSegment, load_benchmarks, resolved_flow_config
from slam_rs.rig import TIMELINE
from slam_rs.tracking import FinalRun, Lockstep, final_trajectory
from slam_rs.trajectory import Trajectory, ate, coverage, shift_clock, write_trajectory
from slam_rs.vio_log import RUN_ENTITY, FrameMaskLog, FrameMode, VioLogger, VioStage, log_calibration, log_frameset_inputs, vio_blueprint

SMOKE_SEGMENT: str = SMOKE_SEGMENTS[1]
"""Default segment: the 7.6 s rotation-dominated panorama from the smoke tier."""

BOOTSTRAP_ENTITY: str = f"{RUN_ENTITY}/bootstrap"
"""Where the inertial bootstrap's attempts go, one row per attempt at the frameset that ran it."""

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
    """Segment id from ``benchmarks.toml``; also names the IMU parameters used for ``--rrd``."""
    rrd: Path | None = None
    """Local base recording; retains the selected dataset configuration and IMU model."""
    gt_rrd: Path | None = None
    """Optional local ground-truth recording; requires --rrd."""
    mask_entity: str | None = None
    """Segmentation masks for ``--stage vio``: the name of each camera's mask entity under its ``pinhole`` (e.g. ``people_mask``
    reads ``<camera>/pinhole/people_mask``). The frontend neither detects nor keeps tracks where a camera's latest mask
    at or before its frame is nonzero (``slam_rs.frame_masks``). None tracks every pixel."""
    mask_rrd: Path | None = None
    """Local mask layer recording for ``--mask-entity``; requires --rrd. None reads the masks from the base recording or the
    catalog segment."""
    catalog: str | None = None
    """Catalog URL override; defaults to slam.toml. Exclusive with local recording files."""
    max_framesets: int | None = None
    """Stop after this many framesets; None replays the whole segment."""
    frame_stride: int = 1
    """Replay every n-th frameset. Every frame is still decoded: decimated AV1 decode is unreliable."""
    window_s: float = DEFAULT_WINDOW_S
    """Longest time window fetched from the catalog in one round trip."""
    output_csv: Path | None = None
    """Where the final trajectory is written; defaults to ``data/<segment>/slam_rs.csv``. The raw VIO goes beside it as ``<stem>_vio.csv``."""
    profile: Literal["reference", "fast"] = "fast"
    """Config overlay applied before tracking."""
    backend: BackendRequest = "auto"
    """Where ``--stage vio`` runs the frontend's pyramid, patch build and KLT tracker.

    ``auto`` runs them on the GPU through CubeCL when this host can, otherwise
    on the CPU with one warning that names the reason. ``gpu`` stops with that
    reason instead of falling back; ``cpu`` forces the CPU port. ``--stage
    frontend`` always runs the CPU frontend.
    """
    decode_device: DecodeRequest = "auto"
    """Video decoder: NVDEC when CUDA is available (the slam-rs-cuda environment), otherwise the CPU decoder."""
    vio_bootstrap: Literal["stereo", "inertial"] | None = None
    """Override ``config.vio_bootstrap``: how the estimator gets its first state. None keeps the dataset config's."""
    blank_cameras: tuple[int, ...] = ()
    """Cameras fed black images instead of their frames: a mono-inertial proxy on a stereo recording.

    A stereo-off switch is not a valid proxy: stereo observations
    would still reach the estimator and give it metric scale. Black images give
    the frontend nothing to detect or track (``tests/test_vio_bootstrap.py::test_blanked_cameras_track_nothing``).
    """
    decode_threads: int = 1
    """PyAV decoder threads per camera stream (CPU decode); the Insta360 lenses decode at about 2.6x real time with 4."""
    image_log_every: int = 1
    """Log the camera images on every n-th frameset; 0 logs none.

    On a long clip the images dominate the recording's size, the replay's memory (a headless replay without
    ``--rr-config.save`` holds everything it logs) and its wall time. Poses, keypoints and counters are logged
    on every frameset regardless.
    """
    map: Literal["off", "offline", "realtime"] | None = None
    """``--stage vio``: the back end's persistent keyframe map, drawn under ``/world/runs/slam_rs``; None keeps the profile's.

    On by default (``backend.enabled``). ``offline`` steps it inline (tracking waits, results
    repeat); ``realtime`` runs it on its own thread (tracking never waits); ``off`` builds none. The VIO poses are the
    same in all three.
    """
    gba: bool = True
    """With the map on: run the global BA on the map after loops and PGO (``Vio.finish_map``) and re-anchor the final
    trajectory on its keyframes; draw them under ``/world/runs/slam_rs/gba`` and write them to ``<output-csv
    stem>_gba_keyframes.csv``. ``--no-gba`` re-anchors on the map as loops and PGO left it. The raw VIO is unchanged."""
    ba_backend: GbaRequest = "follow"
    """Where the global BA's PCG runs: ``follow`` takes the frontend's lane (the GPU when the back-end probe passes);
    ``cpu`` or ``gpu`` choose it outright, for benches and A/B runs."""


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
class BootstrapLog:
    """The inertial bootstrap in the result recording: each attempt when it happens, the status as a recording property.

    The attempts are logged as the run makes them, so a bootstrap that never accepts and stops the run is still in
    the recording. Nothing is logged on the stereo start.
    """

    logged: int = 0
    """Attempts already in the recording."""
    accepted: bool = False
    """Whether the recording already holds the accepted status."""

    def log_new(self, vio: _core.Vio) -> None:
        """Log the attempts made since the last call, at their own frameset time, and the status when it changed.

        Args:
            vio: The pipeline that runs the bootstrap.
        """
        report: _core.BootstrapReport | None = vio.bootstrap_report()
        if report is None:
            return
        attempts: list[_core.BootstrapAttempt] = report.attempts
        for attempt in attempts[self.logged :]:
            rr.set_time(TIMELINE, duration=np.timedelta64(attempt.last_t_ns, "ns"))
            rr.log(
                BOOTSTRAP_ENTITY,
                rr.AnyValues(
                    verdict=attempt.verdict,
                    accepted=attempt.accepted,
                    first_t_ns=attempt.first_t_ns,
                    frames=attempt.frames,
                    tracks=attempt.tracks,
                    tracks_used=attempt.tracks_used,
                ),
            )
            quality: dict[str, float | None] = {
                "window_s": (attempt.last_t_ns - attempt.first_t_ns) * 1e-9,
                "excitation_mps2": attempt.excitation_mps2,
                "residual_rms_rad": attempt.residual_rms_rad,
                "condition": attempt.condition,
                "gravity_norm_unconstrained": attempt.gravity_norm_unconstrained,
                "scale_sigma": attempt.scale_sigma,
            }
            for name, value in quality.items():
                if value is not None:
                    rr.log(f"{BOOTSTRAP_ENTITY}/{name}", rr.Scalars(value))
        accepted: bool = report.start_t_ns is not None
        if len(attempts) == self.logged and accepted == self.accepted:
            return
        self.logged = len(attempts)
        self.accepted = accepted
        # A recording property holds the latest value: on a run that stops this is the status where it stopped.
        rr.send_property(
            "bootstrap",
            rr.AnyValues(
                status="accepted" if accepted else "bootstrapping",
                attempts=len(attempts),
                start_t_ns=report.start_t_ns,
                accepted_at_t_ns=report.accepted_at_t_ns,
                replayed=report.replayed,
            ),
        )

    @staticmethod
    def summary(vio: _core.Vio, first_t_ns: int) -> str | None:
        """One line on the bootstrap for the console, or None on the stereo start.

        Args:
            vio: The pipeline that ran.
            first_t_ns: The segment's first frameset time, the origin of the printed seconds.
        """
        report: _core.BootstrapReport | None = vio.bootstrap_report()
        if report is None:
            return None
        verdicts: list[str] = [attempt.verdict for attempt in report.attempts]
        if report.start_t_ns is None or report.accepted_at_t_ns is None:
            return f"inertial bootstrap: {len(verdicts)} attempts {verdicts}; no window accepted"
        return (
            f"inertial bootstrap: {len(verdicts)} attempts {verdicts}; started at {(report.start_t_ns - first_t_ns) * 1e-9:.2f} s, "
            f"first pose at {(report.accepted_at_t_ns - first_t_ns) * 1e-9:.2f} s, {report.replayed} framesets replayed"
        )


def blank(framesets: Iterable[Frameset], cameras: tuple[int, ...]) -> Iterator[Frameset]:
    """The framesets with black images in ``cameras`` (``--blank-cameras``); unchanged when ``cameras`` is empty.

    Args:
        framesets: The feed's framesets.
        cameras: Camera indices whose images are replaced by zeros.

    Yields:
        Each frameset, its blanked cameras' images zeroed.
    """
    for frameset in framesets:
        if cameras:
            frameset = dataclasses.replace(
                frameset, images=[np.zeros_like(image) if index in cameras else image for index, image in enumerate(frameset.images)]
            )
        yield frameset


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
    # converts one to the other, so this loop and `tracking.drive` cannot
    # disagree about which frameset a count ends on.
    stop_ns: int | None = feed.stop_ns_after(config.max_framesets)
    # The stage that tracks needs the pixels its keypoints were computed on.
    mode: FrameMode = "downscaled" if config.stage == "input" else "jpeg"
    mask_log: FrameMaskLog | None = None if config.mask_entity is None else FrameMaskLog(config.mask_entity)
    times: StageTimes = stage.times if isinstance(stage, VioStage) else StageTimes()
    source: Generator[Frameset, None, None] = feed.framesets(stop_ns)
    framesets: Iterator[Frameset] = blank(source, config.blank_cameras)
    bootstrap: BootstrapLog = BootstrapLog()
    while config.max_framesets is None or replayed < config.max_framesets:
        waited: float = time.monotonic()
        frameset: Frameset | None = next(framesets, None)
        if frameset is None:
            break
        times.add("decode", 1e3 * (time.monotonic() - waited))
        log_images: bool = config.image_log_every > 0 and replayed % config.image_log_every == 0
        replayed += 1
        logged: float = time.monotonic()
        log_frameset_inputs(feed, frameset, mode if log_images else "off", mask_log)
        times.add("log_inputs", 1e3 * (time.monotonic() - logged))
        if isinstance(stage, VioStage):
            stage.run(frameset)
            bootstrap.log_new(stage.lockstep.vio)
        elif stage is not None:
            stage.run(frameset)
    source.close()  # a --max-framesets stop leaves the decode worker up to 64 framesets ahead; stop it and free them

    if isinstance(stage, VioStage):
        stage.flush()
        bootstrap.log_new(stage.lockstep.vio)
    elapsed: float = time.monotonic() - started
    print(f"{replayed} framesets in {elapsed:.1f} s ({replayed / max(elapsed, 1e-9):.1f} fps)")
    print(times.line("frameset"))
    if stage is not None:
        print(stage.summary())
    if isinstance(stage, VioStage) and (line := BootstrapLog.summary(stage.lockstep.vio, int(feed.frame_t_ns[0]))) is not None:
        print(line)
    if isinstance(stage, VioStage) and (shutter := stage.lockstep.vio.rolling_shutter_frames) is not None:
        print(f"rolling shutter: {shutter[0]} frames corrected, {shutter[1]} left raw (no gyro coverage at the frame time)")
    if isinstance(stage, VioStage) and stage.keypoints:
        print(
            f"keypoints held per frameset (all cameras): median {np.median(stage.keypoints):.0f}, "
            f"mean {np.mean(stage.keypoints):.1f}, min {np.min(stage.keypoints)}, max {np.max(stage.keypoints)}"
        )
    return replayed


def main(config: Config) -> None:
    """Replay one segment, log it, write the trajectory and report the error.

    Args:
        config: Parsed CLI options.
    """
    if config.image_log_every < 0:
        raise ValueError(f"--image-log-every {config.image_log_every}: a cadence of 0 (no images) or more")
    settings: SlamConfig = load_slam_config()
    benchmarks: Benchmarks = load_benchmarks(settings)
    listed: ReferenceSegment | None = next((s for s in benchmarks.segments if s.segment_id == config.segment), None)
    dataset_name: str = listed.dataset_name if listed is not None else config.segment.split("__")[0]
    vio_config: _core.VioConfig
    if listed is not None:
        vio_config, _config_text = resolved_flow_config(settings, listed, profile=config.profile)
    else:
        vio_config = _core.VioConfig.from_json(settings.vio_config_text(dataset_name, profile=config.profile))  # refuses an unknown dataset
    if config.vio_bootstrap is not None:
        vio_config.vio_bootstrap = config.vio_bootstrap
    if config.map is not None:
        vio_config = vio_config.with_backend(config.map)
    if config.mask_rrd is not None and config.mask_entity is None:
        raise ValueError("--mask-rrd names a layer; --mask-entity names the masks in it")
    if config.mask_entity is not None and config.stage != "vio":
        raise ValueError("--mask-entity applies to --stage vio, the one stage that tracks with masks")
    source: SegmentSource
    origin: str
    if config.rrd is not None:
        if config.catalog is not None:
            raise ValueError("--catalog and --rrd name two sources for one replay")
        source = LocalSegment(config.rrd, config.gt_rrd, config.mask_rrd)
        origin = str(config.rrd)
    else:
        if config.gt_rrd is not None or config.mask_rrd is not None:
            raise ValueError("--gt-rrd and --mask-rrd require --rrd")
        origin = config.catalog or settings.catalog_url
        source = CatalogSegment(origin, dataset_name, config.segment)
    output_csv: Path = config.output_csv if config.output_csv is not None else Path("data") / config.segment / "slam_rs.csv"
    print(
        f"replaying {config.segment} ({f'{listed.tier} tier' if listed is not None else 'not in the reference set: ground truth only'}) from {origin}"
    )

    with open_segment(
        source,
        profile=replace(dataset_rig(settings.dataset(dataset_name)), decode_threads=config.decode_threads),
        frame_stride=config.frame_stride,
        window_s=config.window_s,
        decode_device=select_decode_device(config.decode_device),
        mask_entity=config.mask_entity,
    ) as feed:
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
            # a configuration this port does not run, or ``--backend gpu`` on a
            # host with no driver, no device or no adapter — and the shim
            # (:func:`slam_rs.apis.run`) is what turns it into one sentence and a
            # non-zero exit rather than a traceback through the feed.
            vio: _core.Vio = _core.Vio(
                _core.Calibration.from_catalog(feed.cameras, feed.imu), vio_config, backend=config.backend, ba_backend=config.ba_backend
            )
            print(f"frontend backend: {vio.backend}" + (f" ({_core.gpu_backend})" if vio.gpu else "") + f", GBA lane: {vio.gba_lane}")
            unknown: list[int] = [camera for camera in config.blank_cameras if not 0 <= camera < len(feed.cameras)]
            if unknown:
                raise ValueError(f"--blank-cameras {unknown}: the rig has cameras 0..{len(feed.cameras) - 1}")
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
        # The raw VIO first, beside the final run with ``_vio`` added to the stem (the names the scorer's
        # ``--est slam_rs=... --est slam_rs_vio=...`` reads), both on the dataset's clock: nothing after it can lose a
        # tracked run.
        vio_csv: Path = output_csv.with_stem(f"{output_csv.stem}_vio")
        write_trajectory(vio_csv, shift_clock(estimate, feed.export_offset_ns))
        # The final run: the GBA on the map after loops and PGO, then the raw VIO re-anchored on its keyframes (the raw
        # VIO itself with the back end off). A failed GBA falls back to the map after loops and PGO.
        finished: FinalRun = final_trajectory(stage.lockstep.vio, estimate, run_gba=config.gba)
        final: Trajectory = finished.final
        map_result: _core.MapResult | None = finished.map_result
        if finished.gba_failure is not None:
            print(f"global BA: failed: {finished.gba_failure}")
        stage.logger.log_final_run(final)
        write_trajectory(output_csv, shift_clock(final, feed.export_offset_ns))
        print(f"{len(estimate)} tracked poses -> {output_csv} (final), {vio_csv} (raw VIO) (absolute ns)")
        if map_result is not None and map_result.gba is not None:
            # At the cursor the flush left on the last frameset, beside the final map.
            log_map_result(map_result)
            gba_csv: Path = output_csv.with_name(f"{output_csv.stem}_gba_keyframes.csv")
            write_trajectory(gba_csv, shift_clock(keyframe_trajectory(map_result, after_gba=True), feed.export_offset_ns))
            print(f"global BA: {map_result.gba!r}; keyframes -> {gba_csv} (absolute ns)")
        if len(estimate) == 0:
            print("no ATE: the estimator reported no tracked pose")
            return
        reference: Trajectory = stage.logger.ground_truth
        if len(reference) == 0:
            return
        print(f"vs ground truth, {coverage(reference, estimate):.1%} of its span covered")
        print(f"raw VIO: {ate(estimate, reference).summary()}")
        print(f"final: {ate(final, reference).summary()}")
