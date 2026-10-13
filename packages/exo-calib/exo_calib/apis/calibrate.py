"""exo-calib App: the ``init`` graph once, then the ``refine`` graph twice; the second pass also runs the report.

Pass 1 tracks person boxes through the Stage A rig, pass 2 through the pass-1 refined rig (handed over in memory); both
passes refine against the ORIGINAL Stage A rig. Outputs become the catalog layers ``exocalib_{init,kp2d,refined,align}``
(pass 1: unregistered ``*_pass1`` files) plus ``eval.json``, and stream to the ``rr_config`` viewer.
``--resume-from init|kp2d`` reads those layers back (after a Producer check) instead of recomputing them.
"""

import hashlib
import importlib.metadata
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import rerun as rr
from dagkit.app import RUNNERS, HardwareClass, detect_hardware, load_profile
from dagkit.graph import Graph
from dagkit.node import Emit, Mode, Node, PortType, RunContext, Tick
from dagkit.sinks import LayerSink, Register, RerunSink, open_layer
from dagkit.sources import SegmentRef
from dagkit.types import Producer, read_column
from serde import serde
from serde.json import to_json
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.sources.catalog_video import SegmentFrames
from simplecv.types.rig import RigCameras

from exo_calib.graphs.calibrate import KP2D_ROOT, init_graph, refine_graph
from exo_calib.nodes.init import InitNode
from exo_calib.nodes.kp2d import Kp2dNode
from exo_calib.nodes.refine import RefineNode
from exo_calib.nodes.report import ReportNode
from exo_calib.sources.layout import discover_rig_layout
from exo_calib.types import OVERLAY_COLOR, RAW_COLOR, VARIANT_COLOR, VARIANTS, EvalReport, RigLayout

PROFILES: Path = Path(__file__).resolve().parents[2] / "profiles"
PORT_TYPES: dict[str, PortType] = {**InitNode.outputs, **Kp2dNode.outputs, **RefineNode.outputs, **ReportNode.outputs}
ROOTS: dict[str, str] = {
    "init": "/world/exocalib/init", "kp2d": KP2D_ROOT, "kp2d_overlay": KP2D_ROOT, "refined": "/world/exocalib/refined",
    "diag": "/world/exocalib/refined", "kp3d": "/world/exocalib/kp3d", "base_rig": "/world", "align": "/world",
}  # fmt: skip
LAYERS: dict[str, tuple[str, ...]] = {
    "exocalib_kp2d": ("kp2d",), "exocalib_refined": ("refined", "base_rig", "diag", "kp3d", "kp2d_overlay"), "exocalib_align": ("align",),
}  # fmt: skip
PASS1_LAYERS: dict[str, tuple[str, ...]] = {"exocalib_kp2d_pass1": ("kp2d",), "exocalib_refined_pass1": ("refined", "diag", "kp3d", "kp2d_overlay")}


@dataclass
class Config:
    rr_config: RerunTyroConfig
    """Live viewer of every graph's outputs (the catalog layers are written regardless)."""
    segment: SegmentRef = field(default_factory=lambda: SegmentRef("dataforge-assembly101", "assembly101__nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724"))
    frame_index: int = 0
    """Sample index of the synchronized frameset Stage A calibrates from (the rig is static)."""
    frame_budget: int = 1000
    """Frames Stage B processes per pass, spread evenly over the capture; shorter captures use every frame."""
    exo_rigs: tuple[str, ...] | None = None
    """Explicit exo rig names such as ``rig_00 rig_01``; ``None`` discovers them from catalog metadata."""
    output_dir: Path = Path("data/outputs")
    """Layer rrds (one subdirectory per segment) and ``eval.json``."""
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


class Keep:
    """Sink that keeps the last value of each input in memory (hands pass-1 rigs and the report to the App)."""

    outputs: Mapping[str, PortType] = {}

    def __init__(self, *ports: str) -> None:
        self.inputs: dict[str, PortType] = {port: PORT_TYPES[port] for port in ports}
        self.values: dict[str, object] = {}

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        self.values[tick.port] = tick.value

    def flush(self, out: Emit) -> None:
        pass


def model_factories(segment: SegmentRef, cameras: tuple[str, ...], params: Kp2dParams) -> dict[str, Callable[[], object]]:
    from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
    from monopriors.models.multiview.multiview_predictor import MultiviewPredictor, MultiviewPredictorConfig
    from posekit.models.rtmpose import RtmPoseConfig
    from posekit.models.yolox import YoloxDetectorConfig
    from posekit.runtimes import TensorRtBackendConfig

    backend = TensorRtBackendConfig(max_batch_size=params.batch_size)
    return {
        "g3t": lambda: MultiviewPredictor(MultiviewPredictorConfig(model_name="g3t")),
        "moge": lambda: MoGeV2MetricPredictor(device="cuda"),
        "detector": lambda: YoloxDetectorConfig(score_thr=params.detection_score, backend=backend).setup(),
        "pose": lambda: RtmPoseConfig(variant="rtmw-x-coco133", backend=backend).setup(),
        "frames": lambda: SegmentFrames(segment, cameras),
    }


def log_style(recording: rr.RecordingStream, cameras: tuple[str, ...]) -> None:
    """Static presentation the Types do not carry per value: frustum colours and the COCO-133 annotation contexts."""
    keypoints: list[rr.AnnotationInfo] = [rr.AnnotationInfo(id=i, label=n) for i, n in COCO_133_ID2NAME.items()]

    def context(class_id: int, label: str, color: tuple[int, int, int], links: bool) -> rr.ClassDescription:
        info = rr.AnnotationInfo(id=class_id, label=label, color=color)
        return rr.ClassDescription(info=info, keypoint_annotations=keypoints, keypoint_connections=COCO_133_LINKS if links else [])

    # No skeleton edges in 3D: a frame's track holds only the joints that triangulated, so most links would dangle.
    kp3d = [context(k + 1, f"kp3d {v}", VARIANT_COLOR[v], False) for k, v in enumerate(VARIANTS)]
    recording.log("/world/exocalib", rr.AnnotationContext(kp3d), static=True)
    for camera in cameras:
        for variant in VARIANTS:
            recording.log(f"/world/exocalib/{variant}/{camera}/pinhole", rr.Pinhole.from_fields(color=VARIANT_COLOR[variant]), static=True)
        overlay = f"/world/{camera}/pinhole/exocalib_kp2d"
        recording.log(overlay, rr.AnnotationContext([context(0, "rtmw-x kp2d", OVERLAY_COLOR, True)]), static=True)
        recording.log(f"{overlay}_rejected", rr.AnnotationContext([context(0, "rtmw-x kp2d rejected", RAW_COLOR, False)]), static=True)


def check_producer(segment: SegmentRef, layer: str, producer: Producer) -> None:
    """Refuse to resume from a layer another graph, profile or parameter set wrote."""
    dataset = segment.entry()
    table = dataset.filter_segments([segment.segment]).filter_contents([f"/producers/{layer}"]).reader(index=None).to_arrow_table()
    stored: dict[str, object] = {c.rsplit(":", 1)[-1]: table.column(c)[0].as_py() for c in table.column_names if c.startswith(f"/producers/{layer}:")}
    expected: dict[str, object] = {"graph": [producer.graph], "profile": [producer.profile], "params_sha": [producer.params_sha]}
    if any(stored.get(key) != value for key, value in expected.items()):
        raise ValueError(f"cannot resume from {layer}: it was written by {stored}, this run is {expected}")


def calibrate(config: Config, golden: rr.RecordingStream | None = None) -> EvalReport:
    """Run the passes; ``golden`` (verify) receives every final output instead of layer files and the viewer."""
    segment: SegmentRef = config.segment
    layout: RigLayout = discover_rig_layout(segment, exo_rigs=config.exo_rigs)
    cameras: tuple[str, ...] = layout.exo_camera_names
    hardware: HardwareClass = detect_hardware() if config.hardware == "auto" else config.hardware
    params: Kp2dParams = load_profile(PROFILES, hardware).configure("kp2d", Kp2dParams())
    models: dict[str, Callable[[], object]] = model_factories(segment, cameras, params)
    run = RUNNERS[config.runner]
    resolved: str = json.dumps([config.frame_index, config.frame_budget, cameras, params.batch_size, params.detection_score, segment.window_ns])
    sha: str = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True, cwd=PROFILES.parent).stdout.strip()
    version: str = f"{importlib.metadata.version('exo-calib')}+{sha}"
    print(f"segment {segment.segment}: exo cameras {list(cameras)}; profile {hardware}")
    output_recording: rr.RecordingStream = golden if golden is not None else config.rr_config.rec_stream

    def producer(graph: str) -> Producer:
        return Producer("exo-calib", version, graph, hardware, params_sha=hashlib.sha256(resolved.encode()).hexdigest()[:12])

    def sinks(graph: str, layers: Mapping[str, tuple[str, ...]], suffix: str = "", register: Register = config.register) -> dict[str, Node]:
        ports: list[str] = [port for layer_ports in layers.values() for port in layer_ports]
        out: dict[str, Node] = {}
        if not suffix:
            name: str = "golden" if golden is not None else "viewer"
            out[name] = RerunSink(output_recording, {p: PORT_TYPES[p] for p in ports}, {p: ROOTS[p] for p in ports}, timeline=segment.timeline)
        if golden is not None:
            return out
        for layer, layer_ports in layers.items():
            recording, partial = open_layer(segment, layer, producer(graph), output_dir=config.output_dir)
            log_style(recording, cameras)
            roots: dict[str, str] = {p: ROOTS[p] + suffix for p in layer_ports}
            out[layer] = LayerSink(recording, partial, segment, layer, {p: PORT_TYPES[p] for p in layer_ports}, roots, register=register)
        return out

    log_style(output_recording, cameras)
    if config.resume_from is None:
        keep = Keep("init")
        run(init_graph(segment, cameras, config.frame_index, models, {"keep": keep, **sinks("exocalib-init", {"exocalib_init": ("init",)})}), mode=config.mode)
        init = keep.values["init"]
    else:
        check_producer(segment, "exocalib_init", producer("exocalib-init"))
        table = read_column(segment.entry(), segment.segment, RigCameras, ROOTS["init"], segment.timeline, static=True)
        _, init = next(RigCameras.from_layer(table, ROOTS["init"], segment.timeline))
    assert isinstance(init, RigCameras)
    tracking: RigCameras = init
    if config.resume_from != "kp2d":  # pass 1 only tracks: its outputs go to unregistered *_pass1 files
        keep = Keep("refined")
        pass1: Graph = refine_graph(segment, layout, (init, init), config.frame_budget, models, {"keep": keep, **sinks("exocalib-refine", PASS1_LAYERS, "_pass1", "off")}, report=False)
        run(pass1, mode=config.mode)
        tracking = keep.values["refined"]  # type: ignore[assignment]
    resume_kp2d: bool = config.resume_from == "kp2d"
    if resume_kp2d:
        check_producer(segment, "exocalib_kp2d", producer("exocalib-refine"))
    keep = Keep("eval")
    layers: dict[str, tuple[str, ...]] = {k: v for k, v in LAYERS.items() if not (resume_kp2d and k == "exocalib_kp2d")}
    pass2: Graph = refine_graph(segment, layout, (init, tracking), config.frame_budget, models, {"keep": keep, **sinks("exocalib-refine", layers)}, report=True, resume_kp2d=resume_kp2d)
    run(pass2, mode=config.mode)
    report = keep.values["eval"]
    assert isinstance(report, EvalReport)
    if golden is None:
        path: Path = config.output_dir / segment.segment / "eval.json"
        path.write_text(to_json(report))
        print(f"wrote {path}")
    return report


def main(config: Config) -> None:
    calibrate(config)
