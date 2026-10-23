"""exo-calib App: the ``init`` graph once, then the ``refine`` graph twice; the second pass also runs the report.

Pass 1 tracks person boxes through the Stage A rig, pass 2 through the pass-1 refined rig (handed over in memory); both
passes refine against the ORIGINAL Stage A rig. Outputs stream to the ``rr_config`` viewer over the segment as its catalog
view shows it (every camera's video and the dataset's cameras); with ``output_dir`` they also become the catalog layers
``exocalib_{init,kp2d,refined,align}`` (pass 1: unregistered ``*_pass1`` files) plus ``eval.json``.
``--resume-from init|kp2d`` swaps the first refine graph's seed source: the stored layers (after a Producer check) instead
of the previous graph's values in memory.
``build`` yields the graphs in order; the CLI (``main`` = ``run_app``) and the gradio app (``tools/apps/calibrate_gradio.py``)
both run it, with run health, so the viewer's "pipeline (debug)" tab shows every graph live.
"""

import functools
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
import torch
from dagkit.app import HardwareClass, detect_hardware, load_profile, run_app, with_pipeline
from dagkit.catalog import read_producer
from dagkit.graph import Graph
from dagkit.node import Mode, Node, PortType, Source
from dagkit.runners import RunReport
from dagkit.sinks import Latest, LayerSink, Register, RerunSink
from dagkit.sources import CatalogSegmentSource, Column, SegmentRef
from dagkit.types import Producer
from serde import serde
from serde.json import to_json
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.sources.catalog_video import SegmentFrames
from simplecv.types.rig import RigCameras

from exo_calib.apis.blueprint import exoego_view
from exo_calib.graphs.calibrate import KP2D_ROOT, init_graph, refine_graph, rig_seed
from exo_calib.nodes.body import BodyNode
from exo_calib.nodes.init import InitNode
from exo_calib.nodes.kp2d import Kp2dNode
from exo_calib.nodes.refine import RefineNode
from exo_calib.nodes.report import ReportNode
from exo_calib.sources.layout import discover_rig_layout
from exo_calib.types import FISHEYE, OVERLAY_COLOR, PINHOLE, RAW_COLOR, VARIANT_COLOR, VARIANTS, EvalReport, MultiViewKeypoints, RigLayout

PROFILES: Path = Path(__file__).resolve().parents[2] / "profiles"
PORT_TYPES: dict[str, PortType] = {**InitNode.outputs, **Kp2dNode.outputs, **RefineNode.outputs, **BodyNode.outputs, **ReportNode.outputs}
ROOTS: dict[str, str] = {
    "init": "/world/exocalib/init", "metric_scale": "/world/exocalib/init", "kp2d": KP2D_ROOT, "kp2d_overlay": KP2D_ROOT,
    "refined": "/world/exocalib/refined", "diag": "/world/exocalib/refined", "kp3d": "/world/exocalib/kp3d", "base_rig": "/world", "align": "/world",
    "status": "/world/exocalib_report", "body_shape": "/world/exocalib/body", "body": "/world/exocalib/body",
}  # fmt: skip
INIT_LAYERS: dict[str, tuple[str, ...]] = {"exocalib_init": ("init", "metric_scale")}
LAYERS: dict[str, tuple[str, ...]] = {
    "exocalib_kp2d": ("kp2d",), "exocalib_refined": ("refined", "base_rig", "diag", "kp3d", "kp2d_overlay"), "exocalib_align": ("align", "status"),
    "exocalib_body": ("body_shape", "body"),
}  # fmt: skip
PASS1_LAYERS: dict[str, tuple[str, ...]] = {"exocalib_kp2d_pass1": ("kp2d",), "exocalib_refined_pass1": ("refined", "diag", "kp3d", "kp2d_overlay")}


@dataclass
class Config:
    rr_config: RerunTyroConfig
    """Live viewer of every graph's outputs."""
    segment: SegmentRef = field(default_factory=lambda: SegmentRef("dataforge-assembly101", "assembly101__nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724"))
    frame_index: int | None = None
    """The first camera's video frame whose synchronized frameset Stage A calibrates from (the rig is static); None: the first
    frameset where every camera's frame is a keyframe, else the first every camera has a frame for. A frame not every camera has
    is an error."""
    frame_budget: int = 200
    """Frames Stage B processes per pass, spread evenly over the capture; shorter captures use every frame. 200 matched 400
    on HOCap and Assembly101 (4 segments each, within 0.05 deg) at 1.5x the speed; 100 lost 0.4 deg on Assembly101."""
    exo_rigs: tuple[str, ...] | None = None
    """Explicit exo rig names such as ``rig_00 rig_01``; ``None`` discovers them from catalog metadata."""
    lens: Literal["pinhole", "fisheye"] = "pinhole"
    """Lens of an exo camera whose segment stores none (a stored lens always wins); ``fisheye`` is equidistant."""
    output_dir: Path | None = field(default=None, metadata={"form": "save"})
    """Layer rrds (one subdirectory per segment, e.g. ``data/outputs``) and ``eval.json``; ``None``: the viewer only."""
    register: Register = "new"
    """``new`` registers layers the segment does not have yet; ``replace`` is the explicit production refresh."""
    resume_from: Literal["init", "kp2d"] | None = None
    hardware: HardwareClass | Literal["auto"] = "auto"
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Kp2dParams:
    """Stage B knobs that depend on the hardware class (``profiles/<hardware>.toml``, node ``kp2d``)."""

    batch_size: int = 32
    """Largest inference batch the TensorRT engines are built for (at least the number of exo views)."""
    detection_score: float = 0.1


def model_factories(segment: SegmentRef, cameras: tuple[str, ...], params: Kp2dParams) -> dict[str, Callable[[], object]]:
    from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
    from monopriors.models.multiview.multiview_predictor import MultiviewPredictor, MultiviewPredictorConfig
    from posekit.models.rtmpose import RtmPoseConfig
    from posekit.models.yolox import YoloxDetectorConfig
    from posekit.runtimes import TensorRtBackendConfig

    from exo_calib.ops.body import SmplxFitter

    backend = TensorRtBackendConfig(max_batch_size=params.batch_size)
    return {
        "g3t": lambda: MultiviewPredictor(MultiviewPredictorConfig(model_name="g3t")),
        "moge": lambda: MoGeV2MetricPredictor(device="cuda"),
        "detector": lambda: YoloxDetectorConfig(score_thr=params.detection_score, backend=backend).setup(),
        "pose": lambda: RtmPoseConfig(variant="rtmw-x-coco133", backend=backend).setup(),
        "frames": lambda: SegmentFrames(segment, cameras),
        "smplx": lambda: SmplxFitter("cuda"),
    }


def log_style(recording: rr.RecordingStream, cameras: tuple[str, ...]) -> None:
    """Static presentation the Types do not carry per value: frustum colours and the COCO-133 annotation contexts."""
    keypoints: list[rr.AnnotationInfo] = [rr.AnnotationInfo(id=i, label=n) for i, n in COCO_133_ID2NAME.items()]

    def context(class_id: int, label: str, color: tuple[int, int, int], links: bool) -> rr.ClassDescription:
        info = rr.AnnotationInfo(id=class_id, label=label, color=color)
        return rr.ClassDescription(info=info, keypoint_annotations=keypoints, keypoint_connections=COCO_133_LINKS if links else [])

    # A frame's track holds only the joints that triangulated (mostly hands); Rerun skips a link with a missing end.
    kp3d = [context(k + 1, f"kp3d {v}", VARIANT_COLOR[v], True) for k, v in enumerate(VARIANTS)]
    recording.log("/world/exocalib", rr.AnnotationContext(kp3d), static=True)
    for camera in cameras:
        for variant in VARIANTS:
            recording.log(f"/world/exocalib/{variant}/{camera}/pinhole", rr.Pinhole.from_fields(color=VARIANT_COLOR[variant]), static=True)
        overlay = f"/world/{camera}/pinhole/exocalib_kp2d"
        recording.log(overlay, rr.AnnotationContext([context(0, "rtmw-x kp2d", OVERLAY_COLOR, True)]), static=True)
        recording.log(f"{overlay}_raw", rr.AnnotationContext([context(0, "rtmw-x kp2d raw", RAW_COLOR, False)]), static=True)
        recording.log(f"{overlay}_rejected", rr.AnnotationContext([context(0, "rtmw-x kp2d rejected", RAW_COLOR, False)]), static=True)


def check_producer(segment: SegmentRef, layer: str, producer: Producer) -> None:
    """Refuse to resume from a layer another graph, profile or parameter set wrote."""
    stored: Producer | None = read_producer(segment.entry(), segment.segment, layer)
    if stored is None or (stored.graph, stored.profile, stored.params_sha) != (producer.graph, producer.profile, producer.params_sha):
        raise ValueError(f"cannot resume from {layer}: it was written by {stored}, this run is {producer}")


def blueprint(layout: RigLayout, graphs: Sequence[str]) -> rrb.Blueprint:
    """The catalog view's exoego layout (``tools/apps/blueprint.py``: the 3D scene, the ego cameras beside it, the exo cameras
    below, panels collapsed), then the "pipeline (debug)" tab: the ``graphs`` and their guides beside three output views the
    guide's links outline."""
    camera: str = layout.exo_camera_names[0]
    outputs: list[rrb.View] = [
        rrb.Spatial2DView(origin=f"/world/{camera}/pinhole", name=f"3 kp2d · keypoints {camera}"),
        rrb.Spatial3DView(origin="/world", contents="/world/exocalib/**", name="4 refine · rigs + 3D keypoints"),
        rrb.Spatial3DView(origin="/world", contents=["/world/exocalib_error/**", "/world/exocalib/refined/**"], name="6 report · error to GT"),
    ]
    return with_pipeline(rrb.Blueprint(exoego_view(layout), collapse_panels=True), outputs, graphs=graphs)


def build(config: Config, recording: rr.RecordingStream) -> Iterator[Graph]:
    """The App's graphs in run order; the CLI, the gradio app and verify run each one before asking for the next.

    ``init`` once, then ``refine`` pass 1 and pass 2: a ``Latest`` sink hands each graph's result to the next, and a stopped
    run ends the App. Outputs stream to ``recording``; with ``config.output_dir`` they also go to layer files and ``eval.json``.
    """
    torch.cuda.empty_cache()  # PyTorch's cache of a previous run (gradio) back to the driver, where TensorRT engines allocate
    segment: SegmentRef = config.segment
    layout: RigLayout = discover_rig_layout(segment, exo_rigs=config.exo_rigs, lens=FISHEYE if config.lens == "fisheye" else PINHOLE)
    cameras: tuple[str, ...] = layout.exo_camera_names
    hardware: HardwareClass = detect_hardware() if config.hardware == "auto" else config.hardware
    params: Kp2dParams = load_profile(PROFILES, hardware).configure("kp2d", Kp2dParams())
    models: dict[str, Callable[[], object]] = model_factories(segment, cameras, params)
    models |= {name: functools.cache(models[name]) for name in ("moge", "detector", "pose", "frames")}  # loaded (fetched) once: every graph shares them
    lenses: list[tuple[str, tuple[float, ...]]] = [(lens.model, lens.coefficients) for lens in map(layout.lens, cameras)]
    resolved: str = json.dumps([config.frame_index, config.frame_budget, cameras, lenses, params.batch_size, params.detection_score, segment.window_ns])
    print(f"segment {segment.segment}: exo cameras {list(cameras)}; profile {hardware}")

    def producer(graph: str) -> Producer:
        return Producer.of("exo-calib", graph, hardware, params_sha=hashlib.sha256(resolved.encode()).hexdigest()[:12])

    def sinks(graph: str, layers: Mapping[str, tuple[str, ...]], suffix: str = "", register: Register = config.register) -> dict[str, Node]:
        ports: list[str] = [port for layer_ports in layers.values() for port in layer_ports]
        out: dict[str, Node] = {}
        if not suffix:
            out["viewer"] = RerunSink(recording, {p: PORT_TYPES[p] for p in ports}, {p: ROOTS[p] for p in ports}, timeline=segment.timeline)
        if config.output_dir is None:
            return out
        for layer, layer_ports in layers.items():
            roots: dict[str, str] = {p: ROOTS[p] + suffix for p in layer_ports}
            out[layer] = sink = LayerSink.open(segment, layer, producer(graph), {p: PORT_TYPES[p] for p in layer_ports}, roots, output_dir=config.output_dir, register=register)
            log_style(sink.recording, cameras)
        return out

    skipped: int = {None: 0, "init": 1, "kp2d": 2}[config.resume_from]  # resume from init skips Stage A, from kp2d also pass 1
    recording.send_blueprint(blueprint(layout, ["exocalib-init", "exocalib-refine-pass1", "exocalib-refine"][skipped:]))
    log_style(recording, cameras)
    seed: Source
    if config.resume_from is None:
        keep = Latest({"init": RigCameras})
        yield init_graph(segment, cameras, config.frame_index, models, {"keep": keep, **sinks("exocalib-init", INIT_LAYERS)})
        if "init" not in keep.values:
            return
        torch.cuda.empty_cache()  # Stage A's G3T (freed with its graph) back to the driver, for the refine passes' engines
        seed = rig_seed(segment, init=keep.values["init"], tracking_rig=keep.values["init"])
    else:  # the seed reads the stored layers back when its graph runs; the Producer check refuses a foreign one now
        check_producer(segment, "exocalib_init", producer("exocalib-init"))
        init = Column(RigCameras, ROOTS["init"], static=True)
        stored: dict[str, Column] = {"init": init, "tracking_rig": init}  # pass 1 tracks through the Stage A rig
        if config.resume_from == "kp2d":  # past Stage B: its stored records, and the GT rig the video source would carry
            check_producer(segment, "exocalib_kp2d", producer("exocalib-refine"))
            stored = {"init": init, "kp2d": Column(MultiViewKeypoints, KP2D_ROOT), "rig": Column(RigCameras, "/world", static=True)}
        seed = CatalogSegmentSource(segment, stored)
    if config.resume_from != "kp2d":  # pass 1 only tracks: its outputs go to unregistered *_pass1 files
        keep = Latest({"init": RigCameras, "refined": RigCameras})
        yield refine_graph(segment, layout, seed, config.frame_budget, models, {"keep": keep, **sinks("exocalib-refine", PASS1_LAYERS, "_pass1", "off")}, report=False)
        if "refined" not in keep.values:
            return
        seed = rig_seed(segment, init=keep.values["init"], tracking_rig=keep.values["refined"])
    keep = Latest({"eval": EvalReport})
    layers: dict[str, tuple[str, ...]] = {k: v for k, v in LAYERS.items() if not (config.resume_from == "kp2d" and k == "exocalib_kp2d")}
    yield refine_graph(segment, layout, seed, config.frame_budget, models, {"keep": keep, **sinks("exocalib-refine", layers)}, report=True)
    report = keep.values.get("eval")
    if config.output_dir is None or report is None:
        return
    assert isinstance(report, EvalReport)
    path: Path = config.output_dir / segment.segment / "eval.json"
    path.write_text(to_json(report))
    print(f"wrote {path}")


def shown_segment(config: Config) -> SegmentRef:
    """The catalog segment the viewer opens (CLI and form): its videos and cameras, which the outputs join."""
    return config.segment


def main(config: Config) -> RunReport:
    return run_app(config, build, segment=shown_segment)
