"""Replay a catalog or local segment through the slam-rs core into a viewer or a file, and report the ground-truth error.

The run and its layer are the catalog layer's (:func:`slam_rs.catalog_layer.track_layer`): one driver, one writer of
every entity. What this tool adds is where the layer goes (``RerunTyroConfig``: a viewer, a file, both), the knobs of an
experiment, the trajectory CSVs and the ATE against the ground truth.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import BackendRequest, GbaRequest, _core
from slam_rs.catalog_feed import (
    DEFAULT_WINDOW_S,
    CatalogSegment,
    DecodeRequest,
    LocalSegment,
    SegmentFeed,
    SegmentSource,
    dataset_rig,
    open_segment,
    select_decode_device,
)
from slam_rs.catalog_layer import LayerDetail, LayerRun, track_layer
from slam_rs.config import SlamConfig, load_slam_config
from slam_rs.gba_log import keyframe_trajectory
from slam_rs.reference import SMOKE_SEGMENTS, Benchmarks, ReferenceSegment, load_benchmarks, resolved_flow_config
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import Trajectory, ate, coverage, shift_clock, write_trajectory
from slam_rs.vio_log import BootstrapLog, VioStage, vio_blueprint

SMOKE_SEGMENT: str = SMOKE_SEGMENTS[1]
"""Default segment: the 7.6 s rotation-dominated panorama from the smoke tier."""


@dataclass(slots=True)
class Config:
    """Replay a reference segment through the slam-rs core."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer, save and headless behaviour."""
    segment: str = SMOKE_SEGMENT
    """Segment id from ``benchmarks.toml``; also names the IMU parameters used for ``--rrd``."""
    rrd: Path | None = None
    """Local base recording; retains the selected dataset configuration and IMU model."""
    gt_rrd: Path | None = None
    """Optional local ground-truth recording; requires --rrd."""
    mask_entity: str | None = None
    """Segmentation masks: the name of each camera's mask entity under its ``pinhole`` (e.g. ``people_mask``
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
    """Where the frontend's pyramid, patch build and KLT tracker run.

    ``auto`` runs them on the GPU through CubeCL when this host can, otherwise
    on the CPU with one warning that names the reason. ``gpu`` stops with that
    reason instead of falling back; ``cpu`` forces the CPU port.
    """
    decode_device: DecodeRequest = "auto"
    """Video decoder: NVDEC when CUDA is available (the slam-rs-cuda environment), otherwise the CPU decoder."""
    vio_bootstrap: Literal["stereo", "inertial"] | None = None
    """Override ``config.vio_bootstrap``: how the estimator gets its first state. None keeps the dataset config's."""
    blank_cameras: tuple[int, ...] = ()
    """Cameras fed black images instead of their frames: a mono-inertial proxy on a stereo recording (:attr:`SegmentFeed.blank_cameras`)."""
    decode_threads: int = 1
    """PyAV decoder threads per camera stream (CPU decode); the Insta360 lenses decode at about 2.6x real time with 4."""
    map: Literal["off", "offline", "realtime"] | None = None
    """The back end's persistent keyframe map, drawn under ``/world/runs/slam_rs``; None keeps the profile's.

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
    animate_base_rig: bool = False
    """On a segment without ground truth: also move the dataset's rig ``/world/rig_00`` with the final run, as the catalog
    layer does there, so views rooted at the rig follow it when this recording is viewed beside the base alone."""
    detail: LayerDetail = "full"
    """What the layer holds (:data:`slam_rs.catalog_layer.LayerDetail`): ``poses`` measures the estimator with the layer's
    paths alone."""


def main(config: Config) -> None:
    """Replay one segment, log its layer, write the trajectories and report the error.

    Args:
        config: Parsed CLI options.
    """
    settings: SlamConfig = load_slam_config()
    benchmarks: Benchmarks = load_benchmarks(settings)
    listed: ReferenceSegment | None = next((s for s in benchmarks.segments if s.segment_id == config.segment), None)
    dataset_name: str = listed.dataset_name if listed is not None else config.segment.split("__")[0]
    vio_config: _core.VioConfig
    config_text: str
    if listed is not None:
        vio_config, config_text = resolved_flow_config(settings, listed, profile=config.profile)
    else:
        config_text = settings.vio_config_text(dataset_name, profile=config.profile)  # refuses an unknown dataset
        vio_config = _core.VioConfig.from_json(config_text)
    if config.vio_bootstrap is not None:
        vio_config.vio_bootstrap = config.vio_bootstrap
    if config.map is not None:
        vio_config = vio_config.with_backend(config.map)
    if config.mask_rrd is not None and config.mask_entity is None:
        raise ValueError("--mask-rrd names a layer; --mask-entity names the masks in it")
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

    feed: SegmentFeed
    with open_segment(
        source,
        profile=dataclasses.replace(dataset_rig(settings.dataset(dataset_name)), decode_threads=config.decode_threads),
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
        elif config.animate_base_rig:
            raise ValueError("--animate-base-rig: this segment has ground truth, and the gt layer owns /world/rig_00")
        unknown: list[int] = [camera for camera in config.blank_cameras if not 0 <= camera < len(feed.cameras)]
        if unknown:
            raise ValueError(f"--blank-cameras {unknown}: the rig has cameras 0..{len(feed.cameras) - 1}")
        if config.blank_cameras:
            feed = dataclasses.replace(feed, blank_cameras=config.blank_cameras)
        calibration: _core.Calibration = _core.Calibration.from_catalog(feed.cameras, feed.imu)
        # Everything this constructor refuses is the caller's own request —
        # a configuration this port does not run, or ``--backend gpu`` on a
        # host with no driver, no device or no adapter — and the shim
        # (:func:`slam_rs.apis.run`) is what turns it into one sentence and a
        # non-zero exit rather than a traceback through the feed.
        vio: _core.Vio = _core.Vio(calibration, vio_config, backend=config.backend, ba_backend=config.ba_backend)
        print(f"frontend backend: {vio.backend}" + (f" ({_core.gpu_backend})" if vio.gpu else "") + f", GBA lane: {vio.gba_lane}")
        rr.send_blueprint(vio_blueprint(feed.camera_positions))
        recording: rr.RecordingStream | None = rr.get_global_data_recording()
        assert recording is not None, "RerunTyroConfig sets up the global recording"
        layer: LayerRun = track_layer(
            recording, feed, vio, config_text=config_text, calibration=calibration, run_gba=config.gba, detail=config.detail,
            profile=config.profile, animate_base_rig=config.animate_base_rig, strict=False,
            max_framesets=config.max_framesets,
        )
        export_offset_ns: int = feed.export_offset_ns
        video_time_offset_ns: int = feed.video_time_offset_ns
        first_t_ns: int = int(feed.frame_t_ns[0])

    run: SegmentRun = layer.run
    stage: VioStage | None = layer.stage
    print(f"{run.framesets} framesets in {run.wall_s:.1f} s ({run.framesets / max(run.wall_s, 1e-9):.1f} fps)")
    if stage is not None:
        print(stage.times.line("frameset"))
        print(stage.summary())
        if (line := BootstrapLog.summary(stage.lockstep.vio, first_t_ns)) is not None:
            print(line)
        if (shutter := stage.lockstep.vio.rolling_shutter_frames) is not None:
            print(f"rolling shutter: {shutter[0]} frames corrected, {shutter[1]} left raw (no gyro coverage at the frame time)")
        if stage.keypoints:
            print(
                f"keypoints held per frameset (all cameras): median {np.median(stage.keypoints):.0f}, "
                f"mean {np.mean(stage.keypoints):.1f}, min {np.min(stage.keypoints)}, max {np.max(stage.keypoints)}"
            )
    print(f"Vio.track median {run.median_tracker_ms:.2f} ms")
    if run.uncovered():
        raise SystemExit(f"{run.uncovered()} framesets never got the inertial samples that cover them")

    estimate: Trajectory = run.estimate
    # The raw VIO first, beside the final run with ``_vio`` added to the stem (the names the scorer's
    # ``--est slam_rs=... --est slam_rs_vio=...`` reads), both on the dataset's clock: nothing after it can lose a
    # tracked run.
    vio_csv: Path = output_csv.with_stem(f"{output_csv.stem}_vio")
    write_trajectory(vio_csv, estimate, video_time_offset_ns=video_time_offset_ns)
    # The final run: the GBA on the map after loops and PGO, then the raw VIO re-anchored on its keyframes (the raw
    # VIO itself with the back end off). A failed GBA falls back to the map after loops and PGO.
    final: Trajectory = estimate if run.final is None else run.final
    if run.gba_failure is not None:
        print(f"global BA: failed: {run.gba_failure}")
    write_trajectory(output_csv, final, video_time_offset_ns=video_time_offset_ns)
    print(f"{len(estimate)} tracked poses -> {output_csv} (final), {vio_csv} (raw VIO) (absolute ns)")
    if run.map_result is not None and run.map_result.gba is not None:
        gba_csv: Path = output_csv.with_name(f"{output_csv.stem}_gba_keyframes.csv")
        write_trajectory(gba_csv, shift_clock(keyframe_trajectory(run.map_result, after_gba=True), export_offset_ns), video_time_offset_ns=video_time_offset_ns)
        print(f"global BA: {run.map_result.gba!r}; keyframes -> {gba_csv} (absolute ns)")
    if len(estimate) == 0:
        print("no ATE: the estimator reported no tracked pose")
        return
    reference: Trajectory = run.ground_truth
    if len(reference) == 0:
        return
    print(f"vs ground truth, {coverage(reference, estimate):.1%} of its span covered")
    print(f"raw VIO: {ate(estimate, reference).summary()}")
    print(f"final: {ate(final, reference).summary()}")
