"""Run the port on one RoboCap session and measure it against the basalt C++ trajectory.

RoboCap has no ground truth, so the only number that means anything is agreement
with the C++. The ``slam`` layer beside each base layer on the NAS **is** the C++
golden run: session 15's layer, moved onto the trajectory clock by the
camera-to-IMU offset, reproduces the checked-in ``traj_s15_vga.csv`` pose for pose
(0.000000 cm rmse, quaternions within float32 rounding), while the catalog-fed
C++ run in the reference bundle sits 9.70 cm away. So the lane to reproduce is
``basalt_vio`` on the raw session files.

Everything that lane was configured with is fed here rather than derived (C72):

* **Cameras.** Four of the six, by name, in the calibration's own order — ``left``,
  ``left_front``, ``right_front``, ``right``. The two eye cameras are excluded.
* **Pixels.** Each 1920x1080 frame is decoded and reformatted to ``gray8`` at
  640x360 in one ``swscale`` call with ``SWS_AREA``, which is the C++'s
  ``cpu_gray8_swscale_area_downscale3``. Colour and size in one filter: converting
  first and resampling afterwards is a different operation.
* **Calibration.** basalt's own file, as the fork's
  ``basalt_convert_robocap_calib.py --downscale 3`` wrote it from the device's
  Kalibr tree, read by :meth:`slam_rs._core.Calibration.from_json`. The recording's
  own statics scaled to the same downscale are asserted against it, so a
  disagreement between the C++'s calibration and the rig's is a failure, not a
  silent five-centimetre bias.
* **Config.** basalt's ``msdmo_config.json``, which ``python/robocap_vit.toml``
  names, read by :meth:`slam_rs._core.VioConfig.from_json` with nothing written
  on top.
* **Inertial pairing.** The two channels run on their own clocks (10,745 gyroscope
  against 10,751 accelerometer samples on session 15). The accelerometer is
  linearly interpolated onto the gyroscope's timestamps and a gyroscope sample
  outside the accelerometer's span is dropped rather than held at an endpoint.
* **Clock.** ``video_time`` is the inertial clock; the frames reach it by adding
  the 14,902,432 ns camera-to-IMU offset, the IMU untouched, which is what
  basalt's reader does. The frameset timestamp is then the median of its four
  frames plus that offset — basalt's own arithmetic, which puts session 15's
  first frameset on 70,258,640,500 + 14,902,432 ns, the C++ CSV's own first row.

The exported CSV carries that clock directly. ``capture_start_time_ns`` is **not**
added, because RoboCap's ``video_time`` is already the device clock the C++ wrote
its trajectory on — unlike MSD, whose ``video_time`` is relative to it. That is a
fact about the recording rather than about this tool, so the manifest states it
(``video_time_is_absolute``) and both tools export through the feed's own
:attr:`slam_rs.catalog_feed.SegmentFeed.export_offset_ns`.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import _core
from slam_rs.catalog_feed import (
    Frameset,
    LocalSegment,
    RigProfile,
    open_segment,
)
from slam_rs.reference import ReferenceManifest, RobocapSession, load_manifest
from slam_rs.tracking import Lockstep, check_calibration_matches_recording, robocap_cpp_trajectory, robocap_estimator_files
from slam_rs.trajectory import AteResult, Trajectory, ate, coverage, empty_trajectory, shift_clock, write_trajectory
from slam_rs.vio_log import FrameMode, VioLogger, VioStage, log_calibration, log_frameset_inputs, vio_blueprint


@dataclass(slots=True)
class Config:
    """Run the port on one RoboCap session against the basalt C++ trajectory beside it."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer, save and headless behaviour."""
    artifact_root: Path | None = None
    """Read every recording and sidecar from one directory per segment; see :func:`slam_rs.reference.relocate`."""
    session: str = "s00000021"
    """RoboCap session id from ``reference_segments.toml``.

    Session 21 and not the fleet tool's 15: this lane draws a recording, and 21
    is the long one (154.9 s, 4,648 framesets) whose repeated loop is where a
    yaw offset or a scale error would show. Session 15 is the fleet default
    because it is the one with a C++ wall measured on the cap, which is what a
    runtime row is read against.
    """
    seconds: float = 90.0
    """Replay this many seconds of video time from the first frameset; 0 replays the whole session.

    The span the S17 evidence was measured over — 2,700 of session 21's
    framesets, 10.40 cm against the C++ — and about 550 MB of ``.rrd``. The
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
    manifest: ReferenceManifest = load_manifest(artifact_root=config.artifact_root)
    session: RobocapSession = manifest.robocap.session(config.session)
    offset_ns: int = manifest.robocap.imu.cam_time_offset_ns
    output_csv: Path = config.output_csv if config.output_csv is not None else Path("data") / f"robocap-{session.session_id}" / "slam_rs.csv"
    calibration: _core.Calibration
    flow_config: _core.VioConfig
    calibration, flow_config = robocap_estimator_files(manifest)
    cpp: Trajectory = robocap_cpp_trajectory(manifest, session)
    print(f"{session.segment_id}: basalt C++ {len(cpp)} poses from {session.slam_path.name} (expected {session.basalt_num_poses})")
    print(f"basalt calibration {manifest.robocap.calibration} at downscale {manifest.robocap.downscale}: {list(calibration.resolution)}")
    print(f"basalt config {manifest.robocap.vio_config}: safe radius {flow_config.optical_flow_image_safe_radius} px")

    with open_segment(
        LocalSegment(base_rrd=session.base_path),
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
            lockstep=Lockstep(vio=_core.Vio(calibration, flow_config)),
            logger=VioLogger(cameras=feed.cameras, ground_truth=empty_trajectory(), cpp=cpp, frame_t_ns=feed.frame_t_ns),
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
            # The estimator alone, so the wall above can be read as decode plus
            # logging plus this. The C++ wall this lane is read against — the
            # manifest's 88.91 s for session 15 — is decode plus its own
            # estimator with basalt's Rerun logging **on**, measured on cap A.
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
    print(f"{len(estimate)} tracked poses -> {output_csv} (the device clock, as basalt's CSVs carry it)")
    if len(estimate) == 0:
        print("no ATE: the estimator reported no tracked pose")
        return
    print(f"vs basalt C++, {coverage(cpp, estimate):.1%} of its span covered")
    agreement: AteResult = ate(estimate, cpp)
    print(agreement.summary())
    first_pose: Float64[ndarray, " 3"] = estimate.position_m[0]
    print(f"first estimated pose at {int(estimate.t_ns[0])} ns, position {first_pose.round(4).tolist()}")
