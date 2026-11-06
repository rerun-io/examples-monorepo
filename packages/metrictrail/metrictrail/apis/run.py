"""Run MetricTrail (ViPE ``pipeline=metrictrail``) on one input and log it to Rerun.

The pipeline is built exactly as upstream's ``run.py`` builds it: Hydra composes ``configs/default.yaml`` with the
benchmark preset's overrides (``configs/slam/README_metrictrail.md``), ViPE validates the typed config, and
``make_pipeline(...).run(stream)`` runs each stream. While it runs, :mod:`metrictrail.capture` logs the live
intermediates; afterwards :mod:`metrictrail.outputs` logs the trajectory, GT, fused cloud, errors and properties.

Inputs: a video (``--benchmark video``), a frame directory named after the sequence, or a VSLAM-LAB sequence directory
(its ``rgb_0/`` frames are linked under ``<output>/frames/<case>`` so the outputs and DPVO's seed carry upstream's eval
case name, and its ``groundtruth.csv`` is the GT).

    pixi run -e metrictrail python tools/run.py --input /mnt/nas/ssd-8tb/metrictrail/data/TUM/rgb/desk --benchmark tum \
        --gt-sequence /mnt/nas/ssd-8tb/datasets/VSLAM-LAB-Benchmark/RGBDTUM/rgbd_dataset_freiburg1_desk \
        --output /tmp/mt/desk --seed 1234 --rr-config.save /tmp/mt/desk.rrd --rr-config.headless
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import rerun as rr
import torch
from jaxtyping import UInt8
from numpy import ndarray
from rerun.chunk import RrdReader
from simplecv.rerun_log_utils import RerunTyroConfig

from metrictrail.apis.log_run import DEFAULT_DEPTH_RANGE_M, DEPTH_RANGES_M
from metrictrail.artifacts import ColoredCloud
from metrictrail.benchmarks import BENCHMARKS, VSLAMLAB_RASTER_OVERRIDES, BenchmarkName, BenchmarkPreset
from metrictrail.blueprint import log_waiting_notes, run_blueprint
from metrictrail.capture import CapturedPass, CaptureSession, rgb_normals
from metrictrail.frames import FrameSource, InputKind, input_kind, link_sequence_frames, open_frames
from metrictrail.ground_truth import GtContext, load_gt_context, protocol_for
from metrictrail.outputs import LoggedRun, LogOptions, RunMetrics, RunOutputs, log_run_outputs, make_provenance, read_run_outputs
from metrictrail.palette import RunLabel


@dataclass
class Config:
    """Run MetricTrail on one sequence or video and log it."""

    input: Path
    """A video, a frame directory (its name names the outputs and seeds DPVO) or a VSLAM-LAB sequence directory."""
    output: Path
    """Run directory: ViPE's artifacts go to ``<output>/vipe_out``, live-capture passes to ``<output>/capture``."""
    benchmark: BenchmarkName = "video"
    """Upstream benchmark preset (``configs/slam/README_metrictrail.md``); ``video`` is the README quick start."""
    seed: int | None = None
    """``pipeline.slam.seed`` (repeated runs: 1234, 1235, 1236); None keeps the preset's (upstream 1234)."""
    seed_name: str | None = None
    """``pipeline.slam.seed_name`` (the name DPVO's per-sequence seed hashes); None: the input's name."""
    gt_sequence: Path | None = None
    """VSLAM-LAB sequence directory for GT; None: the input itself when it is a VSLAM-LAB sequence, else no GT."""
    intrinsics: Path | None = None
    """Known-K file of ``kitti_calibrated`` (``pipeline.optional.intrinsics``)."""
    extra_overrides: list[str] = field(default_factory=list)
    """More Hydra overrides, appended last (e.g. ``pipeline.init.input_raster.resampler=bilinear``)."""
    vipe_config_dir: Path | None = None
    """ViPE's ``configs/`` directory; None: the installed package's (``vipe._paths.get_config_path``)."""
    label: RunLabel = "ours"
    """Which tree runs: ``ours`` (``recipes/vipe-src`` + lietorch-mojo) or ``upstream`` (an unmodified ViPE on ``sys.path``)."""
    capture: bool = True
    """Log the live intermediates (frames, depth, DINOv3 PCA, DPVO patches, BA window); adds host syncs."""
    fuse_cloud: bool = True
    """Fuse the metric cloud with upstream's ``vipe.metrictrail.pointcloud.fuse_point_cloud`` (default filter)."""
    max_cloud_points: int = 400_000
    """Display subsample of the cloud."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer / save / connect flags."""


@dataclass(frozen=True, slots=True)
class PreparedInput:
    """What ViPE reads and what the logger needs to find it again."""

    stream_path: Path
    """``streams.base_path``: the video or the frame directory."""
    name: str
    """Artifact name ViPE gives the run."""
    gt_sequence: Path | None
    """VSLAM-LAB sequence for GT, if any."""
    raster_overrides: list[str]
    """Extra raster overrides for raw VSLAM-LAB frames."""


def prepare_input(config: Config) -> PreparedInput:
    """Resolve the input; a VSLAM-LAB sequence's frames are linked under upstream's eval case name."""
    kind: InputKind = input_kind(config.input)
    if kind == "video":
        if BENCHMARKS[config.benchmark].stream != "video":
            raise ValueError(f"a video input needs --benchmark video, not {config.benchmark}")
        return PreparedInput(stream_path=config.input, name=config.input.stem, gt_sequence=config.gt_sequence, raster_overrides=[])
    if BENCHMARKS[config.benchmark].stream == "video":
        raise ValueError(f"--benchmark video reads a video file, and {config.input} is a directory")
    if kind == "frame_dir":
        return PreparedInput(stream_path=config.input, name=config.input.name, gt_sequence=config.gt_sequence, raster_overrides=[])
    case: str = protocol_for(config.benchmark).upstream_cases.get(config.input.name, config.input.name)
    frame_dir: Path = config.output / "frames" / case
    link_sequence_frames(config.input / "rgb_0", frame_dir)
    return PreparedInput(
        stream_path=frame_dir,
        name=case,
        gt_sequence=config.gt_sequence if config.gt_sequence is not None else config.input,
        raster_overrides=list(VSLAMLAB_RASTER_OVERRIDES.get(config.benchmark, [])),
    )


def hydra_overrides(config: Config, prepared: PreparedInput, vipe_out: Path) -> list[str]:
    """Upstream's command for the preset, then the raw-frame raster and the user's overrides (later wins in Hydra)."""
    preset: BenchmarkPreset = BENCHMARKS[config.benchmark]
    overrides: list[str] = preset.run_overrides(prepared.stream_path, vipe_out, config.intrinsics, config.seed, config.seed_name)
    return overrides + prepared.raster_overrides + list(config.extra_overrides)


def fuse_cloud(outputs: RunOutputs, prepared: PreparedInput, normals: dict[int, UInt8[ndarray, "h w 3"]]) -> ColoredCloud:
    """Upstream's fused metric cloud of the run in ViPE's world frame (needs the ViPE env with a lietorch provider).

    ``normals`` (the live capture's MoGe-2 normals per tracked frame, 8-bit RGB) give every point the voxel mean of its
    pixels' normals rotated into the world by the camera pose; a run without a normal for every frame gets none.
    """
    from vipe.metrictrail.pointcloud import fuse_point_cloud, fusion_settings, load_vipe_artifacts, open_rgb_source

    artifacts = load_vipe_artifacts(outputs.run_dir, outputs.name)
    raster_hw: tuple[int, int] = (int(artifacts.depth.shape[1]), int(artifacts.depth.shape[2]))
    is_video: bool = prepared.stream_path.is_file()
    rgb, _, _ = open_rgb_source(
        raster_hw,
        video=prepared.stream_path if is_video else None,
        frames=None if is_video else prepared.stream_path,
        input_raster=artifacts.input_raster,
        source_frame_ids=artifacts.source_frame_ids,
    )
    complete: bool = all(index in normals for index in range(len(artifacts.depth)))
    if normals and not complete:
        print(f"[metrictrail] {len(normals)} of {len(artifacts.depth)} frames carry MoGe-2 normals; the fused cloud gets none")
    cloud = fuse_point_cloud(
        artifacts.depth,
        artifacts.intrinsics,
        artifacts.c2w,
        rgb,
        fusion_settings("default"),
        world_frame=True,
        normals=(lambda index: rgb_normals(normals[index])) if normals and complete else None,
    )
    return ColoredCloud(
        points_xyz=cloud.points.astype("float32"),
        colors_rgb=cloud.colors.astype("uint8"),
        normals_xyz=None if cloud.normals is None else cloud.normals.astype("float32"),
    )


def main(config: Config) -> RunMetrics:
    """Run the pipeline on ``config.input``, then log the capture and the outputs."""
    from vipe._paths import get_config_path
    from vipe.config.parse import parse_typed_config
    from vipe.pipeline import make_pipeline
    from vipe.streams.base import StreamList

    config.output.mkdir(parents=True, exist_ok=True)
    vipe_out: Path = (config.output / "vipe_out").resolve()
    prepared: PreparedInput = prepare_input(config)
    overrides: list[str] = hydra_overrides(config, prepared, vipe_out)
    config_dir: Path = config.vipe_config_dir if config.vipe_config_dir is not None else get_config_path()
    typed = parse_typed_config("default", hydra_args=overrides, config_dir=config_dir)
    rec: rr.RecordingStream = config.rr_config.rec_stream
    depth_range_m: tuple[float, float] = DEPTH_RANGES_M.get(config.benchmark, DEFAULT_DEPTH_RANGE_M)
    session: CaptureSession | None = None
    if config.capture:
        session = CaptureSession(
            out_dir=config.output / "capture",
            application_id=config.rr_config.application_id,
            recording_id=str(rec.get_recording_id()),
            depth_range_m=depth_range_m,
        )
        session.install()
    # This module imports cv2 (metrictrail.frames) before torch, so OpenMP sizes cv2's thread team to every logical
    # CPU while torch keeps one thread per physical core; both teams spinning cost ~13 % tracking fps on the 32-thread
    # dl-server (15.7 vs 18.2 fps on TUM fr1_desk). Upstream's run.py imports torch first and never sees this.
    cv2.setNumThreads(torch.get_num_threads())
    started: float = time.perf_counter()
    try:
        pipeline = make_pipeline(typed.pipeline)
        streams = StreamList.make(typed.streams)
        for index in range(len(streams)):
            pipeline.run(streams[index])
    finally:
        if session is not None:
            session.uninstall()
    wall_s: float = time.perf_counter() - started

    kept: CapturedPass | None = session.kept_pass() if session is not None else None
    if kept is not None:
        rec.send_chunks(RrdReader(kept.path).stream())
    outputs: RunOutputs = read_run_outputs(vipe_out, prepared.name)
    frames: FrameSource = open_frames(prepared.stream_path)
    context: GtContext | None = None
    if prepared.gt_sequence is not None:
        context = load_gt_context(prepared.gt_sequence, protocol_for(config.benchmark), frames.names)
    cloud: ColoredCloud | None = fuse_cloud(outputs, prepared, session.kept_normals() if session is not None else {}) if config.fuse_cloud else None
    has_gt: bool = context is not None
    rec.send_recording_name(f"MetricTrail {config.label}: {prepared.name}")
    log_waiting_notes(rec, has_capture=kept is not None, has_gt=has_gt)
    import vipe

    provenance = make_provenance(
        dataset_path=prepared.gt_sequence if prepared.gt_sequence is not None else config.input,
        frames_read=prepared.stream_path,
        run_dir=vipe_out,
        producer=f"metrictrail.apis.run ({config.label}; vipe from {Path(vipe.__file__).parent}; overrides: {' '.join(overrides)}; wall {wall_s:.1f} s)",
        env=Path(sys.prefix),
    )
    options = LogOptions(
        label=config.label,
        log_frames=kept is None,
        raster_policy=BENCHMARKS[config.benchmark].raster_policy,
        depth_range_m=depth_range_m,
        max_cloud_points=config.max_cloud_points,
    )
    logged: LoggedRun = log_run_outputs(rec, outputs, options, frames=frames, context=context, cloud_world=cloud, provenance=provenance)
    blueprint = run_blueprint(config.label, has_capture=kept is not None, has_gt=has_gt, world_eye=logged.world_eye, metric_eye=logged.metric_eye)
    rec.send_blueprint(blueprint, make_active=True, make_default=True)
    rec.flush()
    print(logged.metrics)
    return logged.metrics
