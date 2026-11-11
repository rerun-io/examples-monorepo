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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import rerun as rr
import torch
from jaxtyping import Int64, UInt8
from numpy import ndarray
from rerun.chunk import RrdReader
from simplecv.rerun_log_utils import RerunTyroConfig

from metrictrail.apis.log_run import DEFAULT_DEPTH_RANGE_M, DEPTH_RANGES_M
from metrictrail.artifacts import ColoredCloud
from metrictrail.benchmarks import BENCHMARKS, VSLAMLAB_RASTER_OVERRIDES, BenchmarkName, BenchmarkPreset
from metrictrail.blueprint import log_waiting_notes, run_blueprint
from metrictrail.capture import CapturedPass, CaptureSession, rgb_normals
from metrictrail.frames import FrameSource, InputKind, input_kind, link_sequence_frames, open_frames
from metrictrail.gpu_video import GpuVideoSession, gpu_video_input, make_stream_list
from metrictrail.ground_truth import GtContext, load_gt_context, protocol_for
from metrictrail.live_surface import LiveSurface
from metrictrail.outputs import LoggedRun, LogOptions, RunMetrics, RunOutputs, log_run_outputs, make_provenance, read_run_outputs
from metrictrail.palette import RunLabel
from metrictrail.rerun_log import log_input_video
from metrictrail.surface import DEFAULT_TSDF, ColoredMesh, TsdfSettings, fuse_tsdf_mesh


@dataclass
class RunSettings:
    """What to run and how; the Rerun target is separate (:class:`Config` for the CLI, the app's own stream)."""

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
    depth_resolution_level: int = 0
    """MoGe-2 detail level, 0..9: 0 uses 1200 tokens; 9 restores the original 3600-token prior."""
    fuse_mesh: bool = True
    """Fuse a colour TSDF mesh from every tracked frame (:mod:`metrictrail.surface`, tsdf-rs): the default surface."""
    tsdf_voxel_m: float = DEFAULT_TSDF.voxel_size_m
    """TSDF voxel edge, metres (finer for close-up indoor clips; the mesh grows ~4x per halving)."""
    tsdf_max_depth_m: float = DEFAULT_TSDF.max_depth_m
    """Farthest depth the TSDF integrates, metres."""
    fuse_cloud: bool = False
    """Also fuse upstream's metric cloud (``vipe.metrictrail.pointcloud.fuse_point_cloud``, default filter; slow)."""
    max_cloud_points: int = 400_000
    """Display subsample of the cloud."""


@dataclass
class Config(RunSettings):
    """Run MetricTrail on one sequence or video and log it."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer / save / connect flags."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """What :func:`run_and_log` logged, for the caller's blueprint."""

    logged: LoggedRun
    """Metrics and the eyes that frame the trajectory."""
    has_capture: bool
    """Whether a live capture was logged (frames, patches, normals)."""
    has_gt: bool
    """Whether GT was logged."""
    has_video: bool
    """Whether the input video was logged (``input/video``, a video input)."""


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


def prepare_input(config: RunSettings) -> PreparedInput:
    """Resolve the input; a VSLAM-LAB sequence's frames are linked under upstream's eval case name."""
    kind: InputKind = input_kind(config.input)
    if kind == "video":
        if BENCHMARKS[config.benchmark].stream != "video":
            raise ValueError(f"a video input needs --benchmark video, not {config.benchmark}")
        stream_path: Path = config.input
        if config.input.suffix != ".mp4":
            # ViPE's raw_mp4_stream refuses any other suffix; it decodes with OpenCV, which reads .mov/.mkv/... as they are.
            stream_path = config.output / "input" / f"{config.input.stem}.mp4"
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            stream_path.unlink(missing_ok=True)
            stream_path.symlink_to(config.input.resolve())
        return PreparedInput(stream_path=stream_path, name=config.input.stem, gt_sequence=config.gt_sequence, raster_overrides=[])
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


def hydra_overrides(config: RunSettings, prepared: PreparedInput, vipe_out: Path) -> list[str]:
    """Upstream's command for the preset, our MoGe-2 provider (``ours`` only), the raw-frame raster, then the user's
    overrides (later wins in Hydra)."""
    preset: BenchmarkPreset = BENCHMARKS[config.benchmark]
    overrides: list[str] = preset.run_overrides(prepared.stream_path, vipe_out, config.intrinsics, config.seed, config.seed_name)
    provider: list[str] = []
    if config.label == "ours":
        from metrictrail.moge2 import PROVIDER_OVERRIDES

        provider = list(PROVIDER_OVERRIDES)
    return overrides + provider + prepared.raster_overrides + list(config.extra_overrides)


def frame_rgb_source(artifacts: Any, prepared: PreparedInput, gpu: GpuVideoSession | None) -> Callable[[int], UInt8[ndarray, "h w 3"]]:
    """Artifact frame ``i`` -> its 8-bit RGB on the depth raster: a video run's prepared clip on the GPU (tracking's
    own frames, one device-to-host copy), else upstream's decode of the input (``open_rgb_source``)."""
    from vipe.metrictrail.pointcloud import open_rgb_source

    if gpu is not None and artifacts.source_frame_ids is not None:
        rgb = gpu.rgb_source(artifacts.source_frame_ids)
        if rgb is not None:
            return rgb
    is_video: bool = prepared.stream_path.is_file()
    rgb, _, _ = open_rgb_source(
        (int(artifacts.depth.shape[1]), int(artifacts.depth.shape[2])),
        video=prepared.stream_path if is_video else None,
        frames=None if is_video else prepared.stream_path,
        input_raster=artifacts.input_raster,
        source_frame_ids=artifacts.source_frame_ids,
    )
    return rgb


def fuse_mesh(
    outputs: RunOutputs, prepared: PreparedInput, gpu: GpuVideoSession | None = None, settings: TsdfSettings = DEFAULT_TSDF
) -> ColoredMesh | None:
    """The run's colour TSDF mesh in ViPE's world frame: every tracked frame's depth artifact at its final pose."""
    from vipe.metrictrail.pointcloud import load_vipe_artifacts

    artifacts = load_vipe_artifacts(outputs.run_dir, outputs.name)
    return fuse_tsdf_mesh(artifacts.depth, artifacts.intrinsics, artifacts.c2w, frame_rgb_source(artifacts, prepared, gpu), settings)


def fuse_cloud(
    outputs: RunOutputs, prepared: PreparedInput, normals: dict[int, UInt8[ndarray, "h w 3"]], gpu: GpuVideoSession | None = None
) -> ColoredCloud:
    """Upstream's fused metric cloud of the run in ViPE's world frame (needs the ViPE env with a lietorch provider).

    ``normals`` (the live capture's MoGe-2 normals per tracked frame, 8-bit RGB) give every point the voxel mean of its
    pixels' normals rotated into the world by the camera pose; a run without a normal for every frame gets none.
    ``gpu``: see :func:`frame_rgb_source`.
    """
    from vipe.metrictrail.pointcloud import fusion_settings, load_vipe_artifacts

    from metrictrail.fusion import fuse_world_cloud

    artifacts = load_vipe_artifacts(outputs.run_dir, outputs.name)
    complete: bool = all(index in normals for index in range(len(artifacts.depth)))
    if normals and not complete:
        print(f"[metrictrail] {len(normals)} of {len(artifacts.depth)} frames carry MoGe-2 normals; the fused cloud gets none")
    return fuse_world_cloud(
        artifacts.depth,
        artifacts.intrinsics,
        artifacts.c2w,
        frame_rgb_source(artifacts, prepared, gpu),
        fusion_settings("default"),
        normals=(lambda index: rgb_normals(normals[index])) if normals and complete else None,
    )


def run_and_log(
    config: RunSettings,
    rec: rr.RecordingStream,
    *,
    live: bool = False,
    on_stage: Callable[[str], None] | None = None,
    on_cursor: Callable[[int], None] | None = None,
) -> RunResult:
    """Run the pipeline on ``config.input``, then log the capture and the outputs into ``rec`` (no blueprint).

    Args:
        config: What to run.
        rec: Target recording.
        live: The capture logs every DPVO pass straight into ``rec`` while it tracks, depth only on frames that grew
            DPVO's keyframe graph (a recording streamed to a viewer as it grows); False: one file per pass, merged afterwards.
        on_stage: Called with a short status line at each stage and after each tracked frame (the capture's hook);
            it may raise to stop the run.
        on_cursor: Called after each tracked frame with its ``video_time``, ns (the app moves the viewer's cursor there).
    """
    from vipe._paths import get_config_path
    from vipe.config.parse import parse_typed_config
    from vipe.pipeline import make_pipeline

    def stage(text: str) -> None:
        if on_stage is not None:
            on_stage(text)

    def tracked(frame: int, source: int | None, num_frames: int, keyframes: int) -> None:
        stage(f"tracking: frame {frame + 1} / {num_frames}, {keyframes} keyframes in the graph")
        if on_cursor is not None and source is not None and video_time_ns is not None and source < video_time_ns.shape[0]:
            on_cursor(int(video_time_ns[source]))

    if config.label == "ours":
        from metrictrail.moge2 import register_providers

        register_providers(resolution_level=config.depth_resolution_level)
    config.output.mkdir(parents=True, exist_ok=True)
    vipe_out: Path = (config.output / "vipe_out").resolve()
    prepared: PreparedInput = prepare_input(config)
    overrides: list[str] = hydra_overrides(config, prepared, vipe_out)
    config_dir: Path = config.vipe_config_dir if config.vipe_config_dir is not None else get_config_path()
    typed = parse_typed_config("default", hydra_args=overrides, config_dir=config_dir)
    depth_range_m: tuple[float, float] = DEPTH_RANGES_M.get(config.benchmark, DEFAULT_DEPTH_RANGE_M)
    tsdf: TsdfSettings = TsdfSettings(voxel_size_m=config.tsdf_voxel_m, max_depth_m=config.tsdf_max_depth_m)
    frames: FrameSource = open_frames(prepared.stream_path)
    context: GtContext | None = None
    if prepared.gt_sequence is not None:
        context = load_gt_context(prepared.gt_sequence, protocol_for(config.benchmark), frames.names)
    # The input video first, from its own samples: a streamed viewer plays it while the models load and calibrate.
    video_time_ns: Int64[ndarray, "s"] | None = context.frames.timestamps_ns if context is not None else None
    has_video: bool = frames.kind == "video"
    if has_video:
        stage("logging the input video")
        video_time_ns = log_input_video(rec, config.input, config.output / "viewer")
    gpu: GpuVideoSession = GpuVideoSession()
    session: CaptureSession | None = None
    live_surface: LiveSurface | None = LiveSurface(rec, tsdf) if config.capture and live and config.fuse_mesh else None
    if config.capture:
        session = CaptureSession(
            out_dir=config.output / "capture",
            application_id=str(rr.get_application_id(recording=rec)),
            recording_id=str(rec.get_recording_id()),
            depth_range_m=depth_range_m,
            live=rec if live else None,
            every_frame_depth=not live,
            video_time_ns=video_time_ns,
            source_raster=(lambda: gpu.raster) if has_video else None,
            on_frame=tracked if on_stage is not None or on_cursor is not None else None,
            live_surface=live_surface,
        )
        session.install()
    # This module imports cv2 (metrictrail.frames) before torch, so OpenMP sizes cv2's thread team to every logical
    # CPU while torch keeps one thread per physical core; both teams spinning cost ~13 % tracking fps on the 32-thread
    # dl-server (15.7 vs 18.2 fps on TUM fr1_desk). Upstream's run.py imports torch first and never sees this.
    cv2.setNumThreads(torch.get_num_threads())
    started: float = time.perf_counter()
    stage("loading the models, then SharedK calibration (MoGe-2 + DINOv3 on the whole clip)")
    try:
        with gpu_video_input(gpu):
            pipeline = make_pipeline(typed.pipeline)
            streams = make_stream_list(typed.streams)
            for index in range(len(streams)):
                pipeline.run(streams[index])
    finally:
        if session is not None:
            session.uninstall()
    wall_s: float = time.perf_counter() - started

    kept: CapturedPass | None = session.kept_pass() if session is not None else None
    if kept is not None and kept.path is not None:
        rec.send_chunks(RrdReader(kept.path).stream())
    outputs: RunOutputs = read_run_outputs(vipe_out, prepared.name)
    mesh: ColoredMesh | None = None
    if config.fuse_mesh:
        stage("fusing the TSDF mesh")
        mesh = fuse_mesh(outputs, prepared, gpu, tsdf)
    cloud: ColoredCloud | None = None
    if config.fuse_cloud:
        stage("fusing the metric point cloud")
        cloud = fuse_cloud(outputs, prepared, session.kept_normals() if session is not None else {}, gpu)
    stage("logging the trajectory and the surface")
    rec.send_recording_name(f"MetricTrail {config.label}: {prepared.name}")
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
    logged: LoggedRun = log_run_outputs(
        rec, outputs, options, frames=frames, context=context, cloud_world=cloud, provenance=provenance, mesh_world=mesh, video_time_ns=video_time_ns
    )
    return RunResult(logged=logged, has_capture=kept is not None, has_gt=context is not None, has_video=has_video)


def main(config: Config) -> RunMetrics:
    """Run the pipeline on ``config.input``, then log the capture, the outputs and the default blueprint."""
    rec: rr.RecordingStream = config.rr_config.rec_stream
    result: RunResult = run_and_log(config, rec)
    log_waiting_notes(rec, has_capture=result.has_capture, has_gt=result.has_gt)
    blueprint = run_blueprint(
        config.label,
        has_capture=result.has_capture,
        has_gt=result.has_gt,
        has_video=result.has_video,
        world_eye=result.logged.world_eye,
        metric_eye=result.logged.metric_eye,
    )
    rec.send_blueprint(blueprint, make_active=True, make_default=True)
    rec.flush()
    print(result.logged.metrics)
    return result.logged.metrics
