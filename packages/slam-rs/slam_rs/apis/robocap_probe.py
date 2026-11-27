"""Replay RoboCap from the catalog and report regression agreement, without a GT gate."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from simplecv.catalog_video_reader import DecodeRequest, select_decode_device
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import BackendRequest, _core
from slam_rs.catalog_feed import (
    ROBOCAP_RIG,
    CatalogSegment,
    open_segment,
)
from slam_rs.catalog_layer import LayerDetail, LayerRun, track_layer
from slam_rs.config import SlamConfig, load_slam_config
from slam_rs.reference import Benchmarks, RobocapSession, load_benchmarks
from slam_rs.tracking import SegmentRun, check_calibration_matches_recording, robocap_estimator_files
from slam_rs.trajectory import AteResult, Trajectory, ate, coverage, empty_trajectory, read_trajectory, write_trajectory
from slam_rs.vio_log import vio_blueprint

OUTPUT_NAME: str = "slam_rs_vio.csv"
"""The probe's trajectory file: the raw causal VIO, under the name the scoring pair gives it (the Rust ``slam-rs-cli`` writes the final run as ``slam_rs.csv``)."""


@dataclass(slots=True)
class Config:
    """Replay RoboCap from the catalog and report regression agreement."""

    profile: Literal["reference", "fast"] = "fast"
    """Config overlay applied before tracking."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer, save and headless behaviour."""
    catalog: str | None = None
    """Catalog URL; defaults to slam.toml."""
    backend: BackendRequest = "auto"
    """Frontend: auto runs the GPU when this host can, otherwise the CPU with one warning; gpu fails instead; cpu forces it."""
    decode_device: DecodeRequest = "auto"
    """Video decoder: NVDEC when CUDA is available (the slam-rs-cuda environment), otherwise the CPU decoder."""
    reference_csv: Path | None = None
    """Optional regression trajectory; defaults to benchmarks.toml."""
    session: str = "s00000015"
    """RoboCap session id; defaults to s15."""
    seconds: float = 0.0
    """Seconds from the first frameset; zero replays the full session."""
    output_csv: Path | None = None
    """Where the raw VIO trajectory is written; defaults to ``data/robocap-<session>/slam_rs_vio.csv`` (:data:`OUTPUT_NAME`)."""
    detail: LayerDetail = "full"
    """What the layer holds (:data:`slam_rs.catalog_layer.LayerDetail`): ``poses`` measures the estimator with the paths alone."""
    window_s: float = 30.0
    """Longest time window of encoded samples fetched in one round trip."""


def main(config: Config) -> None:
    """Replay one RoboCap session, log it, write the trajectory and report the agreement.

    Args:
        config: Parsed CLI options.
    """
    settings: SlamConfig = load_slam_config()
    benchmarks: Benchmarks = load_benchmarks(settings)
    session: RobocapSession = benchmarks.robocap.session(config.session)
    if not benchmarks.robocap.is_listed(config.session):
        print(f"{config.session} is not in benchmarks.toml: replaying it from the catalog with no regression reference")
    output_csv: Path = config.output_csv if config.output_csv is not None else Path("data") / f"robocap-{session.session_id}" / OUTPUT_NAME
    calibration: _core.Calibration
    flow_config: _core.VioConfig
    config_text: str
    calibration, flow_config, config_text = robocap_estimator_files(settings, profile=config.profile)
    reference_path: Path | None = config.reference_csv or (settings.package_root / session.reference_csv if session.reference_csv else None)
    reference: Trajectory = read_trajectory(reference_path) if reference_path else empty_trajectory()
    print("ground truth absent, not scored; regression reference is reported, not gated")
    print(f"basalt calibration {settings.robocap.calibration} at downscale {ROBOCAP_RIG.downscale}: {list(calibration.resolution)}")
    print(f"basalt config {settings.robocap.vio_config}: safe radius {flow_config.optical_flow_image_safe_radius} px")

    with open_segment(
        CatalogSegment(config.catalog or settings.catalog_url, "robocap", session.segment_id),
        profile=ROBOCAP_RIG,
        window_s=config.window_s,
        decode_device=select_decode_device(config.decode_device),
    ) as feed:
        check_calibration_matches_recording(calibration, feed.cameras, feed.imu, ROBOCAP_RIG.downscale)
        first_ns: int = int(feed.frame_t_ns[0])
        last_ns: int = int(feed.frame_t_ns[-1]) if config.seconds <= 0.0 else first_ns + int(config.seconds * 1e9)
        replayed_ns: int = min(int(feed.frame_t_ns[-1]), last_ns) - first_ns
        print(
            f"{len(feed.cameras)} of the rig's {feed.rig_cameras} cameras {ROBOCAP_RIG.camera_names} at rig positions {feed.camera_positions}, "
            f"{feed.cameras[0].width}x{feed.cameras[0].height}, {len(feed.frame_t_ns)} framesets over "
            f"{(int(feed.frame_t_ns[-1]) - first_ns) / 1e9:.1f} s, replaying the first {replayed_ns / 1e9:.1f} s"
        )
        print(
            f"common estimator time shift {feed.imu.cam_time_offset_ns} ns applied to cameras and IMU; capture_start_time_ns {feed.capture_start_time_ns} "
            f"NOT added (video_time is the device clock), so the export shifts by {feed.export_offset_ns} ns"
        )

        rr.send_blueprint(vio_blueprint(feed.camera_positions))
        # The catalog layer's driver and writer: the D17 hold, the D32 invariant asserts and the rows at the held
        # frameset's own time are one contract, not two copies of one.
        vio: _core.Vio = _core.Vio(calibration, flow_config, backend=config.backend)
        recording: rr.RecordingStream | None = rr.get_global_data_recording()
        assert recording is not None, "RerunTyroConfig sets up the global recording"
        started: float = time.monotonic()
        layer: LayerRun = track_layer(
            recording, feed, vio, config_text=config_text, calibration=calibration, run_gba=False, detail=config.detail,
            profile=config.profile, animate_base_rig=False,
            strict=False, stop_ns=last_ns,
        )
        elapsed: float = time.monotonic() - started
        run: SegmentRun = layer.run
        replayed: int = run.framesets
        tracked: int = len(run.estimate)
        print(f"{replayed} framesets fed, {tracked} tracked, {replayed - tracked} lost")
        print(
            f"wall {elapsed:.1f} s for {replayed_ns / 1e9:.1f} s of footage = {replayed_ns / 1e9 / max(elapsed, 1e-9):.2f}x realtime, "
            f"{1e3 * elapsed / max(replayed, 1):.1f} ms per frameset"
        )
        print(f"Vio.track median {run.median_tracker_ms:.2f} ms")
        if layer.stage is not None:
            print(layer.stage.summary())
        # What the recording says its own clock is, not what this tool assumes:
        # zero here, `capture_start_time_ns` on MSD, and one rule for both tools.
        video_time_offset_ns: int = feed.video_time_offset_ns

    # The feed's own in-process catalog server is what the `with` holds, and
    # nothing below reads a frameset: the export and the ATE happen with it shut.
    if run.uncovered():
        raise SystemExit(f"{run.uncovered()} framesets never got the inertial samples that cover them")
    estimate: Trajectory = run.estimate
    write_trajectory(output_csv, estimate, video_time_offset_ns=video_time_offset_ns)
    print(f"{len(estimate)} tracked poses (raw VIO) -> {output_csv} (absolute device clock)")
    if len(estimate) == 0:
        print("no ATE: the estimator reported no tracked pose")
        return
    if len(reference):
        print(f"vs regression reference, {coverage(reference, estimate):.1%} of its span covered (not gated)")
        agreement: AteResult = ate(estimate, reference)
        print(agreement.summary())
    first_pose: Float64[ndarray, " 3"] = estimate.position_m[0]
    print(f"first estimated pose at {int(estimate.t_ns[0])} ns, position {first_pose.round(4).tolist()}")
