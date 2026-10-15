"""MASt3R-SLAM App: dense monocular SLAM of a catalog segment, a video file, an image directory or a live camera.

The graph is ``graphs/slam.py``; the CLI (``tools/apps/slam.py``) and the gradio app (``tools/apps/slam_gradio.py``) both
run ``build``. SLAM parameters come from ``profiles/<hardware>.toml`` (a preset picks an operating point); ``img_size``
and ``subsample`` override it. Lossless (default) waits for the backend after every keyframe; use realtime for a camera.
Inputs, first given wins: ``video``, ``images``, ``webcam``, then ``segment`` (default: the VERIFY.toml short case, so a
run without arguments and the form's defaults run a known-good clip).
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import HardwareClass, load_profile, run_app, with_pipeline
from dagkit.graph import Graph
from dagkit.node import Mode, Node, PortType, Source
from dagkit.runners import RunReport
from dagkit.sinks import LayerSink, Register
from dagkit.sources import SegmentRef
from dagkit.types import Producer
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from simplecv.sources.catalog_video import VideoSource, log_segment_videos
from simplecv.sources.files import ImageDirSource, VideoFileSource, WebcamSource

from mast3r_slam.graphs.slam import slam_graph
from mast3r_slam.nodes.nerfstudio import NerfstudioSink
from mast3r_slam.nodes.viewer import SlamViewer
from mast3r_slam.types import FrameView, KeyframeView, LogLine, MapUpdate, SlamConfig, TrackedFrame

PACKAGE_DIR: Path = Path(__file__).resolve().parents[2]
"""``packages/mast3r-slam`` (checkpoints and profiles are relative to it)."""
ROOT: str = "/mast3r_slam"
"""Every output lives under this root, never under a GT entity path (two layers must not share one)."""
LAYER: str = "mast3r_slam"


DEFAULT_CLIP: SegmentRef = SegmentRef("msd-odyssey", "msd-odyssey__MOO_others__MOO01_hand_puncher_1", window_ns=(0, 30_000_000_000))
"""The default input: VERIFY.toml's short case, the first 30 s of MOO01 on its own timeline."""


@dataclass
class Config:
    rr_config: RerunTyroConfig
    segment: SegmentRef | None = field(default_factory=lambda: DEFAULT_CLIP)
    """Catalog input when no video, images or webcam is given: the camera named by camera in this segment, inside
    window_ns [start, stop) on the segment's timeline. Only the default clip has a window: a segment you name runs whole
    unless you give it its own window."""
    camera: str = "rig_00/cam_00"
    """Catalog camera entity under /world."""
    undistort: bool = False
    """Catalog input: undistort the camera with its stored Brown-Conrady coefficients first (the MASt3R-SLAM paper does
    on EuRoC always and on TUM when calibrated)."""
    video: Path | None = None
    """Video file input (beats every other input); pixi run mast3r-slam-example-* downloads examples to data/."""
    images: Path | None = field(default=None, metadata={"form": "dir"})
    """Image directory input (natural sort order, 30 fps; used when there is no video)."""
    webcam: int | None = None
    """Live OpenCV camera index, used when no video or image directory is given (run with --mode realtime)."""
    max_frames: int | None = None
    """File inputs: stop after this many frames, counted after subsample."""
    hardware: HardwareClass | Literal["auto"] = "auto"
    preset: Literal["base", "eval", "fast", "calib"] | None = None
    """Profile operating point: fast (224 px, every 5th frame), eval (every 2nd frame), calib (calibrated)."""
    img_size: Literal[224, 512] | None = None
    """MASt3R input size (overrides the preset)."""
    subsample: int | None = None
    """Use every n-th input frame (overrides the preset)."""
    draw: Literal["all", "poses"] = "all"
    """What the viewer draws: all (the CLI's view: the current and keyframe cameras with their images and point maps, the
    path, the map, the log; the input video) or poses (the final keyframe and tracked camera poses only: a small
    recording, what verify keeps)."""
    output_nerfstudio: Path | None = field(default=None, metadata={"form": "save"})
    """Export the final keyframes in Nerfstudio format into this directory."""
    layer_dir: Path | None = field(default=None, metadata={"form": "dir"})
    """Catalog input: also write the segment's mast3r_slam layer here (windowed runs publish it unregistered)."""
    register: Register = "new"
    """How the layer reaches the segment's catalog (new refuses to overwrite)."""
    runner: Literal["sequential", "threaded"] = "threaded"
    """Threaded only: the lossless frontend waits for the backend inside its own calls, where sequential cannot step it."""
    mode: Mode = "lossless"

    def __post_init__(self) -> None:
        segment: SegmentRef | None = self.segment
        if segment is not None and segment.window_ns == DEFAULT_CLIP.window_ns and (segment.dataset, segment.segment) != (DEFAULT_CLIP.dataset, DEFAULT_CLIP.segment):
            self.segment = replace(segment, window_ns=None)  # tyro fills a named segment's window from the default clip's


def outputs(camera: str) -> tuple[dict[str, PortType], dict[str, str]]:
    """Ports and entity roots of the graph's outputs: final keyframe camera poses, per-frame tracked poses, the map, and
    for the viewer the current frame, each new keyframe and the backend's log lines (the CLI's entity names)."""
    name: str = camera.split("/")[-1]
    ports: dict[str, PortType] = {"keyframes": TrackedFrame, "tracked": TrackedFrame, "map": MapUpdate, "view": FrameView, "keyframe": KeyframeView, "log": LogLine}
    roots: dict[str, str] = {"keyframes": f"{ROOT}/{name}", "tracked": f"{ROOT}/tracked/{name}", "map": f"{ROOT}/map", "view": ROOT, "keyframe": f"{ROOT}/keyframes"}
    return ports, roots | {"log": f"{ROOT}/logs"}


def cli_blueprint(n_keyframes: int, roots: dict[str, str], timeline: str) -> rrb.Blueprint:
    """The CLI's view: the 3D scene (without the images a pinhole would back-project and the graph's own outputs) beside the
    current camera's video / image, the last keyframe's image and point map, and the log."""
    current = rrb.Tabs(
        rrb.Spatial2DView(origin=f"{ROOT}/current_camera/pinhole/video", name="Video"),
        rrb.Spatial2DView(origin=f"{ROOT}/current_camera/pinhole/image", name="Image"),
        active_tab=1, name="Current",
    )
    last = rrb.Horizontal(rrb.Spatial2DView(origin=f"{ROOT}/last_keyframe", name="Image"), rrb.Spatial2DView(origin=f"{ROOT}/last_keyframe/pointmap", name="Pointmap"), name="Last Keyframe")
    hidden: list[str] = [f"{ROOT}/current_camera/pinhole/video/**", f"{ROOT}/current_camera/pinhole/pointmap_depth", f"{ROOT}/last_keyframe/pointmap_depth"]
    hidden += [f"{ROOT}/keyframes/keyframe-{i}/pinhole/pointmap_depth" for i in range(n_keyframes)]
    hidden += [f"{roots['map']}/**", f"{roots['tracked']}/**", roots["keyframes"]]
    scene = rrb.Spatial3DView(origin="/", contents=[f"+ {ROOT}/**", *(f"- {path}" for path in hidden)])
    side = rrb.Tabs(rrb.Vertical(current, last, name="Views"), rrb.TextLogView(origin=f"{ROOT}/logs", name="Logs"), active_tab=0)
    return rrb.Blueprint(rrb.Horizontal(scene, side, column_shares=(3, 2)), rrb.TimePanel(timeline=timeline, state="collapsed"), collapse_panels=True)  # fmt: skip


def video_source(config: Config, cfg: SlamConfig) -> Source:
    if config.video is not None:
        return VideoFileSource(config.video, stride=cfg.subsample, max_frames=config.max_frames)
    if config.images is not None:
        return ImageDirSource(config.images, stride=cfg.subsample, max_frames=config.max_frames)
    if config.webcam is not None:
        return WebcamSource(config.webcam, config.max_frames)
    if config.segment is None:
        raise ValueError("no input: give a video, an image directory, a webcam or a segment")
    return VideoSource.catalog(config.segment, (config.camera,), decode="cpu_rgb", stride=cfg.subsample, undistort=config.undistort)


def slam_config(config: Config) -> SlamConfig:
    cfg: SlamConfig = load_profile(PACKAGE_DIR / "profiles", config.hardware).configure("slam", SlamConfig(), config.preset)
    overrides: dict[str, int] = {k: v for k, v in (("img_size", config.img_size), ("subsample", config.subsample)) if v is not None}
    return replace(cfg, **overrides)  # type: ignore[arg-type]


def build(config: Config, recording: rr.RecordingStream) -> Graph:
    if config.runner != "threaded":
        raise ValueError("mast3r-slam runs on the threaded runner only (the sequential runner never steps the backend during the frontend's wait)")
    cfg: SlamConfig = slam_config(config)
    video: Source = video_source(config, cfg)
    ports, roots = outputs(config.camera)
    recording.log(ROOT, rr.ViewCoordinates.RDF, static=True)

    def blueprint(n_keyframes: int) -> rrb.Blueprint:
        debug: list[rrb.View] = [  # the "pipeline (debug)" tab's output views, beside the graph view's guide
            rrb.Spatial3DView(
                origin=ROOT, contents=[f"+ {roots['tracked']}", f"+ {roots['map']}/keyframes", f"+ {roots['map']}/edges"],
                overrides={roots["tracked"]: rr.TransformAxes3D(0.1)}, name="2 frontend · tracked camera",
            ),
            rrb.DataframeView(origin=roots["tracked"], name="2 frontend · tracked poses"),
            rrb.Spatial3DView(origin=ROOT, contents=f"+ {roots['map']}/**", name="3 backend · map"),
        ]  # fmt: skip
        return with_pipeline(cli_blueprint(n_keyframes, roots, video.clock), debug)  # the outputs' clock, not the run health's wall_time

    recording.send_blueprint(blueprint(0))
    drawn: dict[str, PortType] = ports if config.draw == "all" else {p: ports[p] for p in ("keyframes", "tracked")}
    entity: str = f"{ROOT}/current_camera/pinhole/video"

    def input_video(rec: rr.RecordingStream) -> None:
        """The input under the current camera as the CLI logged it: a video file as it is, a catalog camera's encoded
        video (an image directory or a camera has none)."""
        if config.video is not None:
            log_video(config.video, Path(entity), "video_time", recording=rec)
        elif isinstance(video, VideoSource) and config.segment is not None:
            log_segment_videos(rec, config.segment, {config.camera: entity})

    segment: SegmentRef | None = config.segment
    source: str = str(config.video or config.images or (f"webcam {config.webcam}" if config.webcam is not None else f"{segment.dataset}/{segment.segment}" if segment else ""))
    about: list[str] = [f"Dataset: {source}", f"Config: {cfg}"]
    viewer = SlamViewer(recording, drawn, roots, root=ROOT, blueprint=blueprint, about=about, video=input_video if config.draw == "all" else None)
    sinks: dict[str, Node] = {"viewer": viewer}
    if isinstance(video, VideoSource) and config.segment is not None and config.layer_dir is not None:  # a catalog input
        hardware: str = load_profile(PACKAGE_DIR / "profiles", config.hardware).hardware
        producer: Producer = Producer.of("mast3r-slam", "mast3r-slam", f"{hardware}/{config.preset or 'base'}")
        layer_ports: dict[str, PortType] = {p: ports[p] for p in ("keyframes", "tracked", "map")}  # the viewer's values stay out of the catalog
        sinks["layer"] = LayerSink.open(config.segment, LAYER, producer, layer_ports, roots, output_dir=config.layer_dir, register=config.register)
    nerfstudio = NerfstudioSink(config.output_nerfstudio) if config.output_nerfstudio is not None else None
    return slam_graph(video, cfg, sinks, live=isinstance(video, WebcamSource), nerfstudio=nerfstudio)


def main(config: Config) -> RunReport:
    return run_app(config, build)
