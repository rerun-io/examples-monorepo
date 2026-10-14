"""Run slam-rs VIO over one robocap catalog segment, then track its hands on those poses with robocap-live's Rust pipeline.

Two graphs joined by the ``slam_rs`` layer they checkpoint through (``slam`` ``catalog`` skips VIO and reads the registered layer).
Each layer is published under ``output_dir/<segment>/`` (``register``: ``new`` only when the segment has none, ``replace`` for a
production refresh, ``off``). A windowed run (VIO from the segment's start to the window's end, hands inside it) never registers
and copies its hands into the viewer, which opens on the run health; the "pipeline (debug)" tab shows each graph and its guide.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import run_app, with_pipeline
from dagkit.graph import AsOf, Graph
from dagkit.node import Mode, Node, Source
from dagkit.runners import RunReport
from dagkit.sinks import LayerFileSink, Register, RerunSink
from dagkit.sources import SegmentRef
from dagkit.verify import Case, load_card
from serde import from_dict
from simplecv.rerun_log_utils import RerunTyroConfig
from slam_rs.apis.catalog_layer import vio_publish
from slam_rs.nodes.vio import VioParams

from robocap_live.graphs.hands_layer import GROUP_TOLERANCE_NS, PosesCheckpoint, hands_layer_graph, segment_source
from robocap_live.nodes.hands import LAYER, HandsNode, HandsParams
from robocap_live.types import LayerRun

PACKAGE_DIR: Path = Path(__file__).resolve().parents[2]
"""``packages/robocap-live``: case paths (``models_dir``) are relative to it."""


@dataclass
class Config:
    rr_config: RerunTyroConfig
    segment: SegmentRef
    """A robocap catalog segment (``robocap__<device>__s<session>``) with a ``base`` layer (and ``slam_rs`` for ``slam`` catalog)."""
    output_dir: Path
    """The dataset's layer directory, visible at the same path to this host and to the catalog server."""
    hands: HandsParams
    slam: Literal["vio", "catalog"] = "vio"
    """vio: run slam-rs VIO first and track the hands on its poses; catalog: read the segment's registered slam_rs layer."""
    vio: VioParams = VioParams(decode_device="cpu")
    register: Register = "new"
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"


def blueprint(graphs: Sequence[Graph]) -> rrb.Blueprint:
    """The run health, one plot per series (``wall_time``), then the "pipeline (debug)" tab with the hands and their step's health."""
    series: list[str] = [
        f"{e.dst.node}/{what}/{e.dst.port}" for g in graphs for e in g.edges if not isinstance(e.kind, AsOf) for what in ("drops", "queue_depth")
    ]
    series += [f"{name}/step_ms" for g in graphs for name, spec in g.nodes.items() if spec.kind == "node"]
    unique: list[str] = list(dict.fromkeys(series))  # both graphs name their sinks publish and viewer
    normal = rrb.Blueprint(rrb.Grid(*(rrb.TimeSeriesView(origin=f"/dagkit/health/{s}", name=s) for s in unique)), rrb.TimePanel(timeline="wall_time"))
    outputs: list[rrb.View] = [
        rrb.Spatial3DView(origin="/world", contents="+ /world/hands/**", name="2 hands · hands in 3D"),
        rrb.Spatial2DView(origin="/world/rig_00/cam_00/pinhole", name="2 hands · cam_00 overlays"),
        rrb.TimeSeriesView(origin="/dagkit/health/hands", name="2 hands · step ms and queue"),
    ]
    return with_pipeline(normal, outputs, graphs=[g.name for g in graphs] if len(graphs) > 1 else ())


def build(config: Config, recording: rr.RecordingStream) -> Iterator[Graph]:
    """The VIO graph (``slam`` vio), then the hands-layer graph on its published poses; a windowed run's hands go to ``recording``
    too (a whole segment's are ~640 MB)."""
    publish = LayerFileSink(config.segment, LAYER, output_dir=config.output_dir, register=config.register)
    sinks: dict[str, Node] = {
        "publish": publish,
        "viewer": RerunSink(recording, {"run": LayerRun}, {"run": "/hands_layer/run"}, timeline=config.segment.timeline),
    }
    graphs: list[Graph] = []
    poses: Source | None = None
    if config.slam == "vio":
        vio, slam = vio_publish(config.segment, config.vio, config.output_dir, config.register, recording)
        graphs.append(vio)
        poses = PosesCheckpoint(slam, config.segment.timeline)
    source = segment_source(config.segment, catalog_poses=poses is None)
    graphs.append(hands_layer_graph(source, HandsNode(config.segment, config.hands, config.output_dir), sinks, poses))
    recording.send_blueprint(blueprint(graphs))
    yield from graphs
    if config.segment.window_ns is not None and publish.published is not None:
        recording.send_chunks(rr.chunk.RrdReader(publish.published).stream())


def case_segment(case: Case) -> SegmentRef:
    """The case's catalog segment, windowed when the case has a window."""
    assert case.dataset is not None and case.segment is not None, "robocap-live cases are catalog segments"
    window: tuple[int, int] | None = None if case.window_s is None else (round(case.window_s[0] * 1e9), round(case.window_s[1] * 1e9))
    return SegmentRef(case.dataset, case.segment, window_ns=window)


def case_config(case: Case, output_dir: Path, slam: Literal["vio", "catalog"]) -> Config:
    """The App's Config for a case (``models_dir`` relative to the package; VIO params of a ``robocap`` case), unregistered."""
    assert int(case.params["group_tolerance_ns"]) == GROUP_TOLERANCE_NS, f"{case.name}: the graph groups framesets within {GROUP_TOLERANCE_NS} ns"
    hands: HandsParams = from_dict(HandsParams, {"models_dir": str(PACKAGE_DIR / case.params["models_dir"]), "device": case.params["device"],
                                                 "ort_threads": int(case.params["ort_threads"]), "overlays": case.params["overlays"]})  # fmt: skip
    vio: VioParams = from_dict(VioParams, {k: case.params[k] for k in ("profile", "backend", "decode_device")}) if slam == "vio" else VioParams()
    return Config(RerunTyroConfig(headless=True), case_segment(case), output_dir, hands, slam, vio, register="off")


def examples() -> list[Config]:
    """The gradio examples, known-good inputs: VERIFY.toml's short cases (5 s of the README sample on CPU ONNX Runtime, VIO from the
    segment's start for ``robocap``, the registered poses for ``hands``), unregistered."""
    blocks = {block.name: block for block in load_card(PACKAGE_DIR / "VERIFY.toml").graph}
    return [
        case_config(next(c for c in blocks[name].cases if c.name == "short"), PACKAGE_DIR / "data" / "hands-layer", slam)
        for name, slam in (("robocap", "vio"), ("hands", "catalog"))
    ]


def main(config: Config) -> RunReport:
    return run_app(config, build)
