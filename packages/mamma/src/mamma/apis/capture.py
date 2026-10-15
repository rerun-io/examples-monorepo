"""mamma capture App: streaming multi-view body capture (SPEC 5.2, graph ``graphs/capture.py``).

Input: a catalog segment, or a MAMMA-format capture directory (``video``, the public demo files). The viewer shows the rig
with each camera's video under its overlays, the shaded meshes, joints and plots; with ``output_dir`` the outputs also become
the segment's ``mamma_fit`` catalog layer. Presets (``quality``, ``fast``) are node parameters in
``profiles/<hardware>.toml``. The CLI (``tools/apps/capture.py``) and the generated gradio app (``tools/apps/capture_gradio.py``)
run the same ``build``.
"""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from dagkit.app import HardwareClass, load_profile, run_app
from dagkit.graph import Graph
from dagkit.node import Emit, Mode, Node, PortType, RunContext, Tick
from dagkit.runners import RunReport
from dagkit.sinks import LayerSink, Register, RerunSink
from dagkit.sources import SegmentRef
from dagkit.types import Producer
from jaxtyping import Float64
from numpy import ndarray
from serde import serde
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.sources.catalog_video import VideoSource, log_segment_videos
from simplecv.types.rig import RigCameras
from simplecv.video_encoder import ffmpeg_transcode, pick_ffmpeg_encoder

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
from mamma.sources.npz import npz_sequence_source, video_paths
from mamma.types import BodyTopology, FitHealth, FitParams, FitView, Landmarks3d, MultiViewLandmarks, MultiViewTracks, SmplxFits, person_classes

PACKAGE_DIR: Path = Path(__file__).resolve().parents[3]
"""``packages/mamma``: weights, body models and the TensorRT cache live under ``data/`` and ``.trt_cache/``."""
PORTS: dict[str, PortType] = {
    "rig": RigCameras, "tracks": MultiViewTracks, "landmarks": MultiViewLandmarks, "fit": SmplxFits, "kp3d": Landmarks3d, "topology": BodyTopology,
    "view": FitView, "params": FitParams, "health": FitHealth,
}  # fmt: skip
PEOPLE: str = "/world/mamma/people"
ROOTS: dict[str, str] = {
    "rig": "/world", "tracks": "/world", "landmarks": "/world", "fit": PEOPLE, "kp3d": PEOPLE, "topology": PEOPLE, "view": PEOPLE,
    "params": "/metrics/params", "health": "/metrics/fit",
}  # fmt: skip
LAYER: str = "mamma_fit"
LAYER_PORTS: tuple[str, ...] = ("tracks", "fit", "kp3d", "topology")
"""The layer keeps masks, boxes, fits and 3D landmarks (the dense 2D landmarks only stream to the viewer)."""
VIEWER_PORTS: tuple[str, ...] = ("rig", "tracks", "landmarks", "kp3d", "topology", "view", "params", "health")
"""The viewer draws each fit's mesh once, through ``view`` (the layer's ``fit`` with normals and joints)."""
RING: tuple[str, ...] = ("rig_01/cam_00", "rig_03/cam_00", "rig_05/cam_00", "rig_07/cam_00")


@dataclass
class Config:
    rr_config: RerunTyroConfig
    segment: SegmentRef = field(default_factory=lambda: SegmentRef("dataforge-mamma-eval", "mamma-eval__extra__070525_BedlamLab_Eval_Solo_Dancing_00202_1"))
    """The whole segment by default (the VERIFY ``full`` case); ``--segment.window-ns 0 10000000000`` is the ``short`` case."""
    video: Path | None = field(default=None, metadata={"form": "dir"})
    """A MAMMA-format capture directory (meta/*.npz + videos) to run instead of the segment."""
    cameras: tuple[str, ...] | None = None
    """Camera entities under ``/world``; None = every second camera of the mamma-eval ground ring (90 deg apart), or
    every camera of a ``video`` directory."""
    resize_hw: tuple[int, int] = (752, 1024)
    """Engine size (tracking, masks, landmark outputs): aspect kept, 16-aligned (the default suits 2056 x 1504 cameras)."""
    preset: Literal["quality", "fast"] = "quality"
    """Operating point: quality or fast (node parameters in profiles/<hardware>.toml)."""
    subjects: int | None = 1
    """People to track; None = as many as the bootstrap view detects. Only exactly one keeps the fitter's CUDA graphs."""
    output_dir: Path | None = field(default=None, metadata={"form": "save"})
    """Write the mamma_fit layer file here (one subdirectory per segment); None = viewer only."""
    register: Register = "new"
    """new registers the layer when the segment has none; replace is the explicit production refresh; off never registers."""
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
        video: VideoSource = VideoSource.catalog(config.segment, cameras, decode="nvdec_rgb", resize_hw=config.resize_hw, keep_native=keep_native)
    else:
        cameras = config.cameras or tuple(sorted(p.stem for p in (config.video / "meta").glob("*.npz") if p.stem != "global"))
        video = npz_sequence_source(config.video, cameras, resize_hw=config.resize_hw, keep_native=keep_native)
    video.prefetch()
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
    return app_graph(config, recording, video, nodes, f"{profile.hardware}/{config.preset}")


def app_graph(config: Config, recording: rr.RecordingStream, video: VideoSource, nodes: tuple[TrackNode, LandmarksNode, FitNode], profile: str) -> Graph:
    """The capture graph with the App's sinks (the viewer; with ``output_dir`` the ``mamma_fit`` layer) and its blueprint.

    ``build`` passes the loaded models; the unit tests pass fakes. ``profile`` is ``<hardware>/<preset>`` for the producer.
    """
    sinks: dict[str, Node] = {"viewer": CaptureViewer(recording, video, config.video or config.segment)}
    if config.output_dir is not None and config.video is None:
        producer: Producer = Producer.of("mamma", "mamma-capture", profile)
        sinks[LAYER] = LayerSink.open(config.segment, LAYER, producer, {p: PORTS[p] for p in LAYER_PORTS}, ROOTS, output_dir=config.output_dir, register=config.register)
    recording.send_blueprint(default_blueprint(video.cameras))
    return capture_graph(video, *nodes, sinks)


class CaptureViewer(RerunSink):
    """Draw the rig, its videos and the bodies.

    Every input through its Type's Rerun mapping, plus the static scene: a z-up world, the person classes and a ground quad
    under the rig. When the run starts it logs the cameras' videos (a segment's are copied from the catalog, a capture
    directory's files are referenced), so the camera views play under the overlays.
    """

    def __init__(self, recording: rr.RecordingStream, video: VideoSource, videos: SegmentRef | Path) -> None:
        super().__init__(recording, {port: PORTS[port] for port in VIEWER_PORTS}, ROOTS, timeline=video.clock)
        self.cameras, self.videos = video.cameras, videos

    def reset(self, ctx: RunContext) -> None:
        self.recording.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, person_classes(), static=True)
        if isinstance(self.videos, SegmentRef):
            log_segment_videos(self.recording, self.videos, {camera: f"/world/{camera}/pinhole/video" for camera in self.cameras})
        else:
            log_file_videos(self.recording, self.videos, self.cameras, self.timeline)

    def step(self, tick: Tick, out: Emit) -> None:
        super().step(tick, out)
        if isinstance(tick.value, RigCameras):  # the ground (z = 0) under the rig's centre, 2 m past every camera
            rig: RigCameras = tick.value
            centres: Float64[ndarray, "v 3"] = np.einsum("vji,vj->vi", -rig.cam_T_world[:, :3, :3], rig.cam_T_world[:, :3, 3])
            middle: Float64[ndarray, "2"] = centres[:, :2].mean(axis=0)
            half: float = float(np.abs(centres[:, :2] - middle).max()) + 2.0
            corners: Float64[ndarray, "4 3"] = np.c_[middle + half * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]), np.zeros(4)]
            ground = rr.Mesh3D(vertex_positions=corners, triangle_indices=[[0, 1, 2], [0, 2, 3]], vertex_colors=[[70, 80, 90, 110]] * 4)
            self.recording.log("/world/ground", ground, static=True)


def log_file_videos(recording: rr.RecordingStream, root: Path, cameras: tuple[str, ...], timeline: str) -> None:
    """A capture directory's videos, re-encoded to 4:2:0 with a keyframe every 30 frames and no B-frames (the demo files are
    4:4:4 HEVC with B-frames and a keyframe every 250 frames, which the viewer stops decoding past the second); frame i at
    i / fps, the source's clock."""
    fps: int = int(np.load(root / "meta" / "global.npz")["fps"])
    encoder: str = pick_ffmpeg_encoder()
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(len(cameras)) as encodes:
        outputs: list[Path] = [Path(tmp) / f"{i}.mp4" for i in range(len(cameras))]
        list(encodes.map(lambda src, dst: ffmpeg_transcode(src, dst, input_format="mov", fps=fps, encoder=encoder), video_paths(root, cameras), outputs))
        for camera, path in zip(cameras, outputs, strict=True):
            asset = rr.AssetVideo(path=path)
            recording.log(f"/world/{camera}/pinhole/video", asset, static=True)
            times_ns = asset.read_frame_timestamps_nanos()
            frames = rr.VideoFrameReference.columns_nanos(times_ns)
            recording.send_columns(f"/world/{camera}/pinhole/video", indexes=[rr.TimeColumn(timeline, duration=times_ns * 1e-9)], columns=frames)


def main(config: Config) -> RunReport:
    return run_app(config, build)
