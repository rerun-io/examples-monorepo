"""Log a finished MetricTrail output directory (upstream ViPE's or our port's) to Rerun, with its default blueprint.

Nothing runs a model here: the trajectory, intrinsics, depth and run record are read from the artifacts, the frames
from the run's input, GT from a VSLAM-LAB sequence. The live intermediates (DINOv3 PCA, DPVO patches, BA window) exist
only while the pipeline runs (``metrictrail run``); their tiles say so.

    pixi run -e metrictrail python tools/log_run.py --run-dir <vipe_out> --name desk \
        --input /mnt/nas/ssd-8tb/metrictrail/data/TUM/rgb/desk --benchmark tum \
        --gt-sequence /mnt/nas/ssd-8tb/datasets/VSLAM-LAB-Benchmark/RGBDTUM/rgbd_dataset_freiburg1_desk \
        --label upstream --rr-config.save desk.rrd --rr-config.headless
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
from jaxtyping import Int64
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from metrictrail.benchmarks import BENCHMARKS, BenchmarkName, BenchmarkPreset
from metrictrail.blueprint import log_waiting_notes, run_blueprint, shares_input_camera
from metrictrail.frames import FrameSource, open_frames
from metrictrail.ground_truth import GtContext, load_gt_context, protocol_for
from metrictrail.outputs import (
    LoggedRun,
    LogOptions,
    RunMetrics,
    RunOutputs,
    artifact_name,
    log_run_outputs,
    make_provenance,
    read_run_outputs,
    ride_final_trajectory,
)
from metrictrail.palette import RunLabel
from metrictrail.rerun_log import log_input_video

DEPTH_RANGES_M: dict[BenchmarkName, tuple[float, float]] = {
    "tum": (0.3, 3.0),
    "7scenes": (0.3, 4.0),
    "kitti": (2.0, 40.0),
    "kitti_calibrated": (2.0, 40.0),
}
"""Turbo range of the depth images per benchmark; others use :data:`DEFAULT_DEPTH_RANGE_M`."""
DEFAULT_DEPTH_RANGE_M: tuple[float, float] = (0.2, 6.0)
"""Turbo range for in-the-wild inputs."""


@dataclass
class Config:
    """Log a finished MetricTrail run directory."""

    run_dir: Path
    """MetricTrail output directory (``pipeline.output.path``: pose/, intrinsics/, depth/, vipe/)."""
    input: Path
    """The frame directory or video the run read."""
    benchmark: BenchmarkName = "video"
    """Preset the run used (raster policy of the frames; GT protocol with ``gt_sequence``)."""
    name: str | None = None
    """Artifact name (``pose/<name>.npz``); None: the only one in ``run_dir``."""
    gt_sequence: Path | None = None
    """VSLAM-LAB sequence directory with ``rgb.csv`` and ``groundtruth.csv``; None: no GT."""
    label: RunLabel = "ours"
    """Which tree produced the run: ``ours`` (magenta) or ``upstream`` (amber)."""
    producer: str = "metrictrail log-run"
    """Free text naming the tree and commit that produced the run (the ``provenance.producer`` property)."""
    depth_every: int = 1
    """Log every k-th frame's depth (and a frame directory's frames); a video input is logged whole."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer / save / connect flags."""


def main(config: Config) -> RunMetrics:
    """Log ``config.run_dir`` to the recording ``config.rr_config`` set up."""
    preset: BenchmarkPreset = BENCHMARKS[config.benchmark]
    name: str = config.name if config.name is not None else artifact_name(config.run_dir)
    outputs: RunOutputs = read_run_outputs(config.run_dir, name)
    frames: FrameSource = open_frames(config.input)
    context: GtContext | None = None
    if config.gt_sequence is not None:
        if frames.kind != "frame_dir":
            raise ValueError("GT needs the frame directory the run read (frame names carry the VSLAM-LAB timestamps)")
        context = load_gt_context(config.gt_sequence, protocol_for(config.benchmark), frames.names)
    rec: rr.RecordingStream = config.rr_config.rec_stream
    has_gt: bool = context is not None
    rec.send_recording_name(f"MetricTrail {config.label}: {name}")
    has_video: bool = frames.kind == "video"
    video_time_ns: Int64[ndarray, "s"] | None = None
    if has_video:
        video_time_ns = log_input_video(rec, config.input)
    log_waiting_notes(rec, has_capture=False)
    options = LogOptions(
        label=config.label,
        log_frames=True,
        raster_policy=preset.raster_policy,
        depth_range_m=DEPTH_RANGES_M.get(config.benchmark, DEFAULT_DEPTH_RANGE_M),
        depth_every=config.depth_every,
    )
    provenance = make_provenance(
        dataset_path=config.gt_sequence if config.gt_sequence is not None else config.input,
        frames_read=config.input,
        run_dir=config.run_dir,
        producer=f"{config.producer} | logged by metrictrail.apis.log_run (finished artifacts, no live capture)",
        env=Path(sys.prefix),
    )
    logged: LoggedRun = log_run_outputs(
        rec, outputs, options, frames=frames, context=context, provenance=provenance, video_time_ns=video_time_ns
    )
    if shares_input_camera(has_capture=False, has_video=has_video, has_gt=has_gt):
        ride_final_trajectory(rec, outputs.trajectory, video_time_ns)
    blueprint = run_blueprint(
        config.label, has_capture=False, has_gt=has_gt, has_video=has_video, world_eye=logged.world_eye, metric_eye=logged.metric_eye
    )
    rec.send_blueprint(blueprint, make_active=True, make_default=True)
    rec.flush()
    print(logged.metrics)
    return logged.metrics
