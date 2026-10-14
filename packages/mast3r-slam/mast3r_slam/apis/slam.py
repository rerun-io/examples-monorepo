"""MASt3R-SLAM App: dense monocular SLAM of a catalog segment, a video file, an image directory or a live camera.

The graph is ``graphs/slam.py``; the CLI (``tools/apps/slam.py``) and the gradio app (``tools/apps/slam_gradio.py``) both
run ``build``. SLAM parameters come from ``profiles/<hardware>.toml`` (a preset picks an operating point); ``img_size``
and ``subsample`` override it. Lossless (default) waits for the backend after every keyframe; use realtime for a camera.
Inputs, first given wins: ``video``, ``images``, ``webcam``, then ``segment`` (default: the VERIFY.toml short case, so a
run without arguments and the form's defaults run a known-good clip). 0 means "not set" for ``max_frames`` and ``subsample``.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import HardwareClass, load_profile, run_app, with_pipeline
from dagkit.graph import Graph
from dagkit.node import Event, Mode, Node, PortType, Source
from dagkit.runners import RunReport
from dagkit.sinks import LayerSink, Register, RerunSink, open_layer
from dagkit.sources import SegmentRef
from dagkit.types import Producer
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.sources.catalog_video import CatalogVideoSource
from simplecv.sources.files import ImageDirSource, VideoFileSource
from simplecv.types.frames import MultiViewFrames

from mast3r_slam.graphs.slam import slam_graph
from mast3r_slam.nodes.nerfstudio import NerfstudioSink
from mast3r_slam.ops.image import resize_img
from mast3r_slam.sources.webcam import Webcam
from mast3r_slam.types import KeyframeCloud, MapUpdate, SlamConfig, TrackedFrame

PACKAGE_DIR: Path = Path(__file__).resolve().parents[2]
"""``packages/mast3r-slam`` (checkpoints and profiles are relative to it)."""
ROOT: str = "/mast3r_slam"
"""Every output lives under this root, never under a GT entity path (two layers must not share one)."""
LAYER: str = "mast3r_slam"


@dataclass
class Config:
    rr_config: RerunTyroConfig
    segment: SegmentRef | None = field(default_factory=lambda: SegmentRef("msd-odyssey", "msd-odyssey__MOO_others__MOO01_hand_puncher_1", window_ns=(0, 30_000_000_000)))
    """Catalog input when no video, images or webcam is given: the camera ``camera`` of this segment (``window_ns`` clips it)."""
    camera: str = "rig_00/cam_00"
    """Catalog camera entity under ``/world``."""
    undistort: bool = False
    """Catalog input: undistort the camera with its stored Brown-Conrady coefficients first (the MASt3R-SLAM paper does
    on EuRoC always and on TUM when calibrated)."""
    video: Path | None = None
    """Video file input (beats every other input); ``pixi run mast3r-slam-example-*`` downloads examples to ``data/``."""
    images: Path | None = None
    """Image directory input (natural sort order, 30 fps; used when there is no video)."""
    webcam: int | None = None
    """Live OpenCV camera index, used when no video or image directory is given (run with ``--mode realtime``)."""
    max_frames: int | None = None
    """File inputs: stop after this many frames, counted after ``subsample`` (0: all)."""
    hardware: HardwareClass | Literal["auto"] = "auto"
    preset: Literal["base", "eval", "fast", "calib"] | None = None
    """Profile operating point: ``fast`` (224 px, every 5th frame), ``eval`` (every 2nd frame), ``calib`` (calibrated)."""
    img_size: Literal[224, 512] | None = None
    """MASt3R input size (overrides the preset)."""
    subsample: int | None = None
    """Use every n-th input frame (overrides the preset; 0 keeps it)."""
    output_nerfstudio: Path | None = None
    """Export the final keyframes in Nerfstudio format into this directory."""
    layer_dir: Path | None = None
    """Catalog input: also write the segment's ``mast3r_slam`` layer here (windowed runs publish it unregistered)."""
    register: Register = "new"
    """How the layer reaches the segment's catalog (``new`` refuses to overwrite)."""
    runner: Literal["sequential", "threaded"] = "threaded"
    """``threaded`` only: the lossless frontend waits for the backend inside its own calls, where ``sequential`` cannot step it."""
    mode: Mode = "lossless"


def examples() -> list[Config]:
    """The gradio app's one-click example: the VERIFY.toml short case (the default segment's first 30 s, every 2nd frame
    at 512 px, uncalibrated, lossless), so one click reproduces the banded run."""
    return [Config(RerunTyroConfig(headless=True), subsample=2, img_size=512)]


def outputs(camera: str) -> tuple[dict[str, PortType], dict[str, str]]:
    """Ports and entity roots of the graph's outputs: final keyframe camera poses, per-frame tracked poses, the map and
    the keyframe clouds (under the map's per-keyframe poses)."""
    name: str = camera.split("/")[-1]
    ports: dict[str, PortType] = {"keyframes": TrackedFrame, "tracked": TrackedFrame, "map": MapUpdate, "cloud": KeyframeCloud}
    return ports, {"keyframes": f"{ROOT}/{name}", "tracked": f"{ROOT}/tracked/{name}", "map": f"{ROOT}/map", "cloud": f"{ROOT}/map/keyframes"}


def frame_hw(video: Source, img_size: Literal[224, 512]) -> tuple[int, int]:
    """MASt3R frame size of the source's images (decodes its first frame)."""
    events = iter(video.events())
    try:
        first: Event = next(e for e in events if e.port == "frames")
    finally:
        getattr(events, "close", lambda: None)()
    frames: MultiViewFrames = first.value  # type: ignore[assignment]
    height, width = frames.rgb.shape[-2:]
    shape = resize_img(np.zeros((height, width, 3), dtype=np.float32), img_size).rgb_tensor.shape
    return int(shape[-2]), int(shape[-1])


def video_source(config: Config, cfg: SlamConfig) -> Source:
    max_frames: int | None = config.max_frames or None
    if config.video is not None:
        return VideoFileSource(config.video, stride=cfg.subsample, max_frames=max_frames)
    if config.images is not None:
        return ImageDirSource(config.images, stride=cfg.subsample, max_frames=max_frames)
    if config.webcam is not None:
        return Webcam(config.webcam, max_frames)
    if config.segment is None:
        raise ValueError("no input: give a video, an image directory, a webcam or a segment")
    return CatalogVideoSource(config.segment, (config.camera,), decode="cpu_rgb", stride=cfg.subsample, undistort=config.undistort)


def slam_config(config: Config) -> SlamConfig:
    cfg: SlamConfig = load_profile(PACKAGE_DIR / "profiles", config.hardware).configure("slam", SlamConfig(), config.preset)
    overrides: dict[str, int] = {k: v for k, v in (("img_size", config.img_size), ("subsample", config.subsample)) if v}
    return replace(cfg, **overrides)  # type: ignore[arg-type]


def build(config: Config, recording: rr.RecordingStream) -> Graph:
    if config.runner != "threaded":
        raise ValueError("mast3r-slam runs on the threaded runner only (the sequential runner never steps the backend during the frontend's wait)")
    cfg: SlamConfig = slam_config(config)
    video: Source = video_source(config, cfg)
    ports, roots = outputs(config.camera)
    recording.log(ROOT, rr.ViewCoordinates.RDF, static=True)
    debug: list[rrb.View] = [  # the "pipeline (debug)" tab's output views, beside the graph view's guide
        rrb.Spatial3DView(
            origin=ROOT, contents=[f"+ {roots['tracked']}", f"+ {roots['map']}/keyframes", f"+ {roots['map']}/edges"],
            overrides={roots["tracked"]: rr.TransformAxes3D(0.1)}, name="2 frontend · tracked camera",
        ),
        rrb.DataframeView(origin=roots["tracked"], name="2 frontend · tracked poses"),
        rrb.Spatial3DView(origin=ROOT, contents=f"+ {roots['map']}/**", name="3 backend · map"),
    ]  # fmt: skip
    normal = rrb.Blueprint(rrb.Spatial3DView(origin=ROOT, name="MASt3R-SLAM"), rrb.TimePanel(state="collapsed", timeline=video.clock), collapse_panels=True)
    recording.send_blueprint(with_pipeline(normal, debug))  # the outputs' clock, not the run health's wall_time
    sinks: dict[str, Node] = {"viewer": RerunSink(recording, ports, roots)}
    if isinstance(video, CatalogVideoSource) and config.layer_dir is not None:
        producer = Producer("mast3r-slam", "0.0.1", "mast3r-slam", config.preset or "base")
        layer, partial = open_layer(video.segment, LAYER, producer, output_dir=config.layer_dir)
        layer_ports: dict[str, PortType] = {p: ports[p] for p in ("keyframes", "tracked", "map")}  # clouds stay out of the catalog
        sinks["layer"] = LayerSink(layer, partial, video.segment, LAYER, layer_ports, roots, register=config.register)
    nerfstudio = NerfstudioSink(config.output_nerfstudio) if config.output_nerfstudio is not None else None
    return slam_graph(video, cfg, frame_hw(video, cfg.img_size), sinks, live=isinstance(video, Webcam), nerfstudio=nerfstudio)


def main(config: Config) -> RunReport:
    return run_app(config, build)
