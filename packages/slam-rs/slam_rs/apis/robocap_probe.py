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

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import _core
from slam_rs.apis.replay import JPEG_QUALITY, VioStage, _log_calibration, log_imu
from slam_rs.catalog_feed import (
    TIMELINE,
    CameraCalib,
    Frameset,
    LocalSegment,
    RigProfile,
    open_segment,
)
from slam_rs.frontend_log import camera_entity
from slam_rs.reference import MANIFEST_PATH, ImuParameters, ReferenceManifest, RobocapSession, load_manifest
from slam_rs.tracking import Lockstep, robocap_cpp_trajectory, robocap_estimator_files
from slam_rs.trajectory import AteResult, Trajectory, ate, coverage, empty_trajectory, shift_clock, write_trajectory
from slam_rs.vio_log import VioLogger, vio_blueprint


@dataclass(slots=True)
class Config:
    """Run the port on one RoboCap session against the basalt C++ trajectory beside it."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer, save and headless behaviour."""
    manifest: Path = MANIFEST_PATH
    """Reference manifest; a machine without the NAS runs a copy with the artifact prefix rewritten."""
    session: str = "s00000021"
    """RoboCap session id from ``reference_segments.toml``."""
    seconds: float = 90.0
    """Replay this many seconds of video time from the first frameset; 0 replays the whole session."""
    output_csv: Path | None = None
    """Where the estimated trajectory is written; defaults to ``data/robocap-<session>/slam_rs.csv``."""
    log_frames: bool = True
    """Log the four camera images. Off measures the estimator's wall time without the JPEG encode."""
    window_s: float = 30.0
    """Longest time window of encoded samples fetched in one round trip."""


def check_calibration_matches_recording(
    basalt: _core.Calibration, cameras: tuple[CameraCalib, ...], imu: ImuParameters, downscale: int
) -> None:
    """Refuse a C++ calibration that is not the rig the recording and the manifest describe.

    Everything the file carries is compared, not just the resolution: the lenses
    and the extrinsics come from the same Kalibr tree by different routes — the
    fork's converter for the file, the ``dataforge`` conversion for the recording
    — and a route that drifted would otherwise show up only as a few centimetres
    of trajectory error nobody could attribute. The recording's native statics
    are scaled here the way the converter scales them.

    The inertial half is checked against the manifest for the same reason by a
    different route: this lane configures the *estimator* from the file and the
    *feed* from the manifest's frozen Kalibr values, so a re-conversion that
    moved one and not the other would split them silently. The file's own
    ``cam_time_offset_ns`` must be zero, because the feed is what applies that
    offset (to the frames, not the IMU) and a file carrying it too would apply it
    twice.

    Args:
        basalt: The calibration read from basalt's own JSON.
        cameras: The feed's cameras, already scaled to ``downscale``.
        imu: The manifest's frozen IMU parameters, which configure the feed.
        downscale: The factor both were scaled by.

    Raises:
        ValueError: If the camera count, a resolution, an intrinsic, a distortion
            coefficient, an extrinsic, the IMU model or the clock offset disagrees.
    """
    if basalt.camera_count != len(cameras):
        raise ValueError(f"basalt's calibration has {basalt.camera_count} cameras, the feed selected {len(cameras)}")
    expected: tuple[tuple[int, int], ...] = tuple((camera.width, camera.height) for camera in cameras)
    if tuple(basalt.resolution) != expected:
        raise ValueError(f"basalt's calibration is {list(basalt.resolution)}, the feed decodes {list(expected)} at downscale {downscale}")
    # `Calibration` exposes no intrinsics accessor, so the comparison goes through
    # the round trip its own `to_json` writes, which is basalt's shape.
    written: dict[str, Any] = json.loads(basalt.to_json())["value0"]
    for camera, lens, extrinsic in zip(cameras, written["intrinsics"], written["T_imu_cam"], strict=True):
        if lens["camera_type"] != camera.model:
            raise ValueError(f"cam {camera.index}: basalt's model is {lens['camera_type']!r}, the recording gives {camera.model!r}")
        values: dict[str, float] = lens["intrinsics"]
        for name, mine in (("fx", camera.fx), ("fy", camera.fy), ("cx", camera.cx), ("cy", camera.cy)):
            # float32 statics on the recording against float64 in the JSON: a
            # thousandth of a pixel is rounding, a hundredth is a different rig.
            if abs(values[name] - mine) > 1e-2:
                raise ValueError(f"cam {camera.index}: basalt's {name} is {values[name]}, the recording gives {mine} at downscale {downscale}")
        # The distortion is resolution-invariant, so it is compared as stored and
        # to float32's own precision rather than to a pixel's.
        for number, mine in enumerate(camera.distortion.tolist(), start=1):
            if abs(values[f"k{number}"] - mine) > 1e-6:
                raise ValueError(f"cam {camera.index}: basalt's k{number} is {values[f'k{number}']}, the recording gives {mine}")
        translation: Float64[ndarray, " 3"] = np.array([extrinsic["px"], extrinsic["py"], extrinsic["pz"]], dtype=np.float64)
        offset_m: float = float(np.abs(translation - camera.imu_T_cam[:3, 3]).max())
        if offset_m > 1e-4:
            raise ValueError(f"cam {camera.index}: basalt places it {1e3 * offset_m:.3f} mm from where the recording does")
        written_rotation: Rotation = Rotation.from_quat([extrinsic["qx"], extrinsic["qy"], extrinsic["qz"], extrinsic["qw"]])
        turn_deg: float = float(np.degrees((written_rotation * Rotation.from_matrix(camera.imu_T_cam[:3, :3]).inv()).magnitude()))
        if turn_deg > 1e-2:
            raise ValueError(f"cam {camera.index}: basalt turns it {turn_deg:.4f} deg from where the recording does")
    for name, mine in (
        ("imu_update_rate", imu.rate_hz),
        ("gyro_noise_std", imu.gyro_noise_std),
        ("accel_noise_std", imu.accel_noise_std),
        ("gyro_bias_std", imu.gyro_bias_std),
        ("accel_bias_std", imu.accel_bias_std),
    ):
        # A rate is one number, a noise density is one per axis and all three are
        # the same number, which is how Kalibr writes an isotropic model.
        theirs: list[float] = written[name] if isinstance(written[name], list) else [written[name]]
        if any(abs(value - mine) > 1e-12 for value in theirs):
            raise ValueError(f"basalt's {name} is {written[name]}, the manifest gives {mine}")
    if written["cam_time_offset_ns"] != 0:
        raise ValueError(f"basalt's calibration carries cam_time_offset_ns {written['cam_time_offset_ns']}; the feed applies that offset, so the file must not")


def main(config: Config) -> None:
    """Replay one RoboCap session, log it, write the trajectory and report the agreement.

    Args:
        config: Parsed CLI options.
    """
    manifest: ReferenceManifest = load_manifest(config.manifest)
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

        _log_calibration(feed.cameras)
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
        frameset: Frameset
        for frameset in feed.framesets(last_ns):
            if frameset.t_ns > last_ns:
                break
            replayed += 1
            log_imu(frameset.imu)
            rr.set_time(TIMELINE, duration=np.timedelta64(frameset.t_ns, "ns"))
            if config.log_frames:
                for camera, image in zip(feed.cameras, frameset.images, strict=True):
                    rr.log(f"{camera_entity(camera.index)}/image", rr.Image(image, color_model="L").compress(jpeg_quality=JPEG_QUALITY))
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
            # logging plus this. The C++ wall in `run.json` is decode plus its own
            # estimator with nothing logged.
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
