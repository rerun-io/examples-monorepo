"""mamma capture App: streaming multi-view body capture (SPEC 5.2, graph ``graphs/capture.py``).

Input: a catalog segment, or a MAMMA-format capture directory (``video``, the public demo files; its rig and videos are
logged too). The outputs stream to the ``rr_config`` viewer; with ``output_dir`` they also become the segment's ``mamma_fit``
catalog layer. Presets (``quality``, ``fast``) are node parameters in ``profiles/<hardware>.toml``.
"""

import importlib.metadata
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from dagkit.app import HardwareClass, load_profile, run_app
from dagkit.graph import Graph
from dagkit.node import Mode, Node, PortType
from dagkit.runners import RunReport
from dagkit.sinks import LayerSink, Register, RerunSink, open_layer
from dagkit.sources import SegmentRef
from dagkit.types import Producer
from serde import serde
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.sources.catalog_video import CatalogVideoSource

from mamma.apis.blueprint import default_blueprint
from mamma.graphs.capture import capture_graph
from mamma.models.detection import Yolo12xDetector
from mamma.models.identity import OpenClipIdentity
from mamma.models.landmarks.estimator import LandmarkEstimator
from mamma.models.landmarks.tensorrt_backend import static_engine_path
from mamma.models.sam2 import Sam2Masks
from mamma.nodes.fit import FitNode
from mamma.nodes.landmarks import LandmarksNode
from mamma.nodes.track import TrackNode, TrackParams
from mamma.ops.smplx_body import SmplxBody
from mamma.ops.window_fit import FitterConfig, SlidingWindowFitter
from mamma.sources.npz import NpzSequenceSource
from mamma.types import BodyTopology, Landmarks3d, MultiViewLandmarks, MultiViewTracks, SmplxFits

PACKAGE_DIR: Path = Path(__file__).resolve().parents[3]
"""``packages/mamma``: weights, body models and the TensorRT cache live under ``data/`` and ``.trt_cache/``."""
PORTS: dict[str, PortType] = {"tracks": MultiViewTracks, "landmarks": MultiViewLandmarks, "fit": SmplxFits, "kp3d": Landmarks3d, "topology": BodyTopology}
ROOTS: dict[str, str] = {"tracks": "/world", "landmarks": "/world", "fit": "/world/mamma/people", "kp3d": "/world/mamma/people", "topology": "/world/mamma/people"}
LAYER: str = "mamma_fit"
LAYER_PORTS: tuple[str, ...] = ("tracks", "fit", "kp3d", "topology")
"""The layer keeps masks, boxes, fits and 3D landmarks (the dense 2D landmarks only stream to the viewer)."""
RING: tuple[str, ...] = ("rig_01/cam_00", "rig_03/cam_00", "rig_05/cam_00", "rig_07/cam_00")


@dataclass
class Config:
    rr_config: RerunTyroConfig
    segment: SegmentRef = field(default_factory=lambda: SegmentRef("dataforge-mamma-eval", "mamma-eval__extra__070525_BedlamLab_Eval_Solo_Dancing_00202_1"))
    video: Path | None = None
    """A MAMMA-format capture directory (``meta/*.npz`` + videos) to run instead of ``segment``."""
    cameras: tuple[str, ...] | None = None
    """Camera entities under ``/world``; None = every second camera of the mamma-eval ground ring (90 deg apart), or
    every camera of a ``video`` directory."""
    resize_hw: tuple[int, int] = (752, 1024)
    """Engine size (tracking, masks, landmark outputs): aspect kept, 16-aligned (the default suits 2056 x 1504 cameras)."""
    preset: Literal["quality", "fast"] = "quality"
    subjects: int | None = 1
    """People to track; None = as many as the bootstrap view detects. Only exactly one keeps the fitter's CUDA graphs."""
    output_dir: Path | None = None
    """Write the ``mamma_fit`` layer file here (one subdirectory per segment); None = viewer only."""
    register: Register = "new"
    """``new`` registers the layer when the segment has none; ``replace`` is the explicit production refresh."""
    hardware: HardwareClass | Literal["auto"] = "auto"
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class VideoParams:
    """Decode knobs (profile node ``video``)."""

    keep_native: bool = False
    """Decode at full size and crop MammaNet's input from it (the engine frames are an area resize)."""


def engine_plan() -> Path | None:
    """The MammaNet static-batch TensorRT plan (``apis/build_trt_engine.py``), if this machine has built it."""
    plan: Path = static_engine_path(PACKAGE_DIR / ".trt_cache")
    return plan if plan.exists() else None


def build(config: Config, recording: rr.RecordingStream) -> Graph:
    """Models load here, before the run, so ``RunReport.wall_s`` measures the stream (decoders start in the source thread)."""
    import torch

    torch.set_float32_matmul_precision("high")
    profile = load_profile(PACKAGE_DIR / "profiles", config.hardware)
    track: TrackParams = replace(profile.configure("track", TrackParams(), config.preset), expected_subjects=config.subjects)
    fit: FitterConfig = profile.configure("fit", FitterConfig(), config.preset)
    fit = replace(fit, use_cuda_graph=fit.use_cuda_graph and config.subjects == 1)  # per-person graph captures deadlock
    keep_native: bool = profile.configure("video", VideoParams(), config.preset).keep_native
    if config.video is None:
        cameras: tuple[str, ...] = config.cameras or RING
        video: CatalogVideoSource = CatalogVideoSource(config.segment, cameras, decode="nvdec_rgb", resize_hw=config.resize_hw, keep_native=keep_native)
    else:
        cameras = config.cameras or tuple(sorted(p.stem for p in (config.video / "meta").glob("*.npz") if p.stem != "global"))
        video = files = NpzSequenceSource(config.video, cameras, resize_hw=config.resize_hw, keep_native=keep_native)
        log_files(recording, files, config.segment.timeline)
    for view in range(len(cameras)):
        video.frames.times_ns(view)  # fetch the packets now, as the pre-port export did before its clock started
    data: Path = PACKAGE_DIR / "data"
    body: SmplxBody = SmplxBody.load(data / "body_models", data / "body_models/downsampled_verts/verts_512.pkl")
    nodes = (
        TrackNode(
            track, Yolo12xDetector(PACKAGE_DIR / track.yolo_checkpoint), OpenClipIdentity(),
            Sam2Masks(track.sam2_config, PACKAGE_DIR / track.sam2_checkpoint, memory_window_size=track.memory_window_size),
        ),
        LandmarksNode(LandmarkEstimator(data / "weights/ma_2d/mamma_mask_full_cvpr.safetensors", engine_path=engine_plan())),
        FitNode(fit, np.asarray(body.model.faces, dtype=np.int32), lambda K, cam_T_world: SlidingWindowFitter(body, K, cam_T_world, fit)),
    )  # fmt: skip
    sinks: dict[str, Node] = {"viewer": RerunSink(recording, PORTS, ROOTS, timeline=video.clock)}
    if config.output_dir is not None and config.video is None:
        sha: str = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True, cwd=PACKAGE_DIR).stdout.strip()
        producer = Producer("mamma", f"{importlib.metadata.version('mamma')}+{sha}", "mamma-capture", f"{profile.hardware}/{config.preset}")
        layer, partial = open_layer(config.segment, LAYER, producer, output_dir=config.output_dir)
        sinks[LAYER] = LayerSink(layer, partial, config.segment, LAYER, {p: PORTS[p] for p in LAYER_PORTS}, ROOTS, register=config.register)
    recording.send_blueprint(default_blueprint(cameras))
    return capture_graph(video, *nodes, sinks)


def log_files(recording: rr.RecordingStream, source: NpzSequenceSource, timeline: str) -> None:
    """A file capture has no base layer: log its rig, z-up world and videos (frame references on the source's clock)."""
    recording.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for log in source.rig.to_rerun("/world"):
        recording.log(log.path, log.data, static=True)
    for camera, path in zip(source.cameras, source.videos, strict=True):
        asset = rr.AssetVideo(path=path)
        recording.log(f"/world/{camera}/pinhole/video", asset, static=True)
        times_ns = asset.read_frame_timestamps_nanos()
        frames = rr.VideoFrameReference.columns_nanos(times_ns)
        recording.send_columns(f"/world/{camera}/pinhole/video", indexes=[rr.TimeColumn(timeline, duration=times_ns * 1e-9)], columns=frames)


def main(config: Config) -> RunReport:
    return run_app(config, build)
