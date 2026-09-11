"""Replay RoboCap from the catalog and report regression agreement, without a GT gate."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import _core
from slam_rs.catalog_feed import (
    CatalogSegment,
    Frameset,
    RigProfile,
    open_segment,
)
from slam_rs.reference import ReferenceManifest, RobocapSession, load_manifest
from slam_rs.tracking import Lockstep, check_calibration_matches_recording, robocap_estimator_files
from slam_rs.trajectory import AteResult, Trajectory, ate, coverage, empty_trajectory, read_trajectory, shift_clock, write_trajectory
from slam_rs.vio_log import FrameMode, VioLogger, VioStage, log_calibration, log_frameset_inputs, vio_blueprint


@dataclass(slots=True)
class Config:
    """Config."""

    profile: Literal["reference", "fast"] = "fast"
    """Config overlay applied before tracking."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer, save and headless behaviour."""
    catalog: str | None = None
    """Catalog URL; defaults to the manifest."""
    gpu: bool = False
    """Use the GPU frontend."""
    reference_csv: Path | None = None
    """Optional regression trajectory; defaults to the manifest."""
    session: str = "s00000015"
    """RoboCap session id from ``reference_segments.toml``.

    Session 21 and not the fleet tool's 15: this lane draws a recording, and 21
    is the long one (154.9 s, 4,648 framesets) whose repeated loop is where a
    yaw offset or a scale error would show. Session 15 is the fleet default
    because it is the one with a reference wall measured on the cap, which is what a
    runtime row is read against.
    """
    seconds: float = 0.0
    """Replay this many seconds of video time from the first frameset; 0 replays the whole session.

    The span the S17 evidence was measured over — 2,700 of session 21's
    framesets, 10.40 cm against the reference — and about 550 MB of ``.rrd``. The
    whole session is a viewer recording nobody opens.
    """
    output_csv: Path | None = None
    """Where the estimated trajectory is written; defaults to ``data/robocap-<session>/slam_rs.csv``."""
    log_frames: bool = True
    """Log the four camera images. Off measures the estimator's wall time without the JPEG encode."""
    window_s: float = 30.0
    """Longest time window of encoded samples fetched in one round trip."""


def main(config: Config) -> None:
    """Replay one RoboCap session, log it, write the trajectory and report the agreement.

    Args:
        config: Parsed CLI options.
    """
    manifest: ReferenceManifest = load_manifest()
    session: RobocapSession = manifest.robocap.session(config.session)
    offset_ns: int = manifest.robocap.imu.cam_time_offset_ns
    output_csv: Path = config.output_csv if config.output_csv is not None else Path("data") / f"robocap-{session.session_id}" / "slam_rs.csv"
    calibration: _core.Calibration
    flow_config: _core.VioConfig
    calibration, flow_config, _config_text = robocap_estimator_files(manifest, profile=config.profile)
    reference_path: Path | None = config.reference_csv or (manifest.package_root / session.reference_csv if session.reference_csv else None)
    reference: Trajectory = read_trajectory(reference_path) if reference_path else empty_trajectory()
    print("ground truth absent, not scored; regression reference is reported, not gated")
    print(f"basalt calibration {manifest.robocap.calibration} at downscale {manifest.robocap.downscale}: {list(calibration.resolution)}")
    print(f"basalt config {manifest.robocap.vio_config}: safe radius {flow_config.optical_flow_image_safe_radius} px")

    with open_segment(
        CatalogSegment(config.catalog or manifest.catalog_url, "robocap", session.segment_id),
        manifest.robocap.imu,
        profile=RigProfile.from_robocap(manifest.robocap),
        window_s=config.window_s,
    ) as feed:
        check_calibration_matches_recording(calibration, feed.cameras, manifest.robocap.imu, manifest.robocap.downscale)
        first_ns: int = int(feed.frame_t_ns[0])
        last_ns: int = int(feed.frame_t_ns[-1]) if config.seconds <= 0.0 else first_ns + int(config.seconds * 1e9)
        replayed_ns: int = min(int(feed.frame_t_ns[-1]), last_ns) - first_ns
        print(
            f"{len(feed.cameras)} of the rig's {feed.rig_cameras} cameras {manifest.robocap.camera_names} at rig positions {feed.camera_positions}, "
            f"{feed.cameras[0].width}x{feed.cameras[0].height}, {len(feed.frame_t_ns)} framesets over "
            f"{(int(feed.frame_t_ns[-1]) - first_ns) / 1e9:.1f} s, replaying the first {replayed_ns / 1e9:.1f} s"
        )
        print(
            f"camera offset {offset_ns} ns applied to the frames; capture_start_time_ns {feed.capture_start_time_ns} "
            f"NOT added (video_time is the device clock), so the export shifts by {feed.export_offset_ns} ns"
        )

        log_calibration(feed.cameras)
        rr.send_blueprint(vio_blueprint(feed.cameras))
        # The same drive `replay --stage vio` runs: the D17 hold, the D32
        # invariant asserts and the rows at the held frameset's own time are one
        # contract, not two copies of one.
        stage: VioStage = VioStage(
            lockstep=Lockstep(vio=_core.Vio(calibration, flow_config, gpu=config.gpu)),
            logger=VioLogger(cameras=feed.cameras, ground_truth=empty_trajectory(), frame_t_ns=feed.frame_t_ns),
        )

        replayed: int = 0
        started: float = time.monotonic()
        mode: FrameMode = "jpeg" if config.log_frames else "off"
        frameset: Frameset
        for frameset in feed.framesets(last_ns):
            if frameset.t_ns > last_ns:
                break
            replayed += 1
            log_frameset_inputs(feed, frameset, mode)
            stage.run(frameset)
        elapsed: float = time.monotonic() - started
        stage.logger.log_complete_paths()

        lockstep: Lockstep = stage.lockstep
        tracked: int = len(lockstep.elapsed_ms)
        print(
            f"{replayed} framesets fed, {tracked} tracked, {lockstep.retries} refused and retried, "
            f"{len(lockstep.pending)} lost, {lockstep.imu_samples} inertial samples pushed"
        )
        print(
            f"wall {elapsed:.1f} s for {replayed_ns / 1e9:.1f} s of footage = {replayed_ns / 1e9 / max(elapsed, 1e-9):.2f}x realtime, "
            f"{1e3 * elapsed / max(replayed, 1):.1f} ms per frameset"
        )
        if tracked:
            print(
                f"Vio.track {np.mean(lockstep.elapsed_ms):.1f} ms per frameset "
                f"(median {np.median(lockstep.elapsed_ms):.1f}, max {np.max(lockstep.elapsed_ms):.1f}), "
                f"{1e-3 * float(np.sum(lockstep.elapsed_ms)):.1f} s of the wall"
            )
        estimate: Trajectory = stage.logger.estimated()
        # What the recording says its own clock is, not what this tool assumes:
        # zero here, `capture_start_time_ns` on MSD, and one rule for both tools.
        export_offset_ns: int = feed.export_offset_ns

    # The feed's own in-process catalog server is what the `with` holds, and
    # nothing below reads a frameset: the export and the ATE happen with it shut.
    stage.refuse_lost_framesets()
    write_trajectory(output_csv, shift_clock(estimate, export_offset_ns))
    print(f"{len(estimate)} tracked poses -> {output_csv} (absolute device clock)")
    if len(estimate) == 0:
        print("no ATE: the estimator reported no tracked pose")
        return
    if len(reference):
        print(f"vs regression reference, {coverage(reference, estimate):.1%} of its span covered (not gated)")
        agreement: AteResult = ate(shift_clock(estimate, export_offset_ns), reference)
        print(agreement.summary())
    first_pose: Float64[ndarray, " 3"] = estimate.position_m[0]
    print(f"first estimated pose at {int(estimate.t_ns[0])} ns, position {first_pose.round(4).tolist()}")
