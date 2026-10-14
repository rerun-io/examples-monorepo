"""Track one robocap catalog segment's hands with robocap-live's Rust pipeline and publish its ``hands`` layer (SPEC 5.3).

The finished file is published under ``output_dir/<segment>/`` and registered as the segment's ``hands`` layer (``register``:
``new`` only when it has none, ``replace`` for a production refresh, ``off``). A windowed run (``--segment.window-ns``) is a
smoke run, never registered; its layer is also copied into the viewer. The run's counts go to the viewer at ``/hands_layer/run``.
The viewer opens on the run health as before; the "pipeline (debug)" tab shows the graph, its guide and the hands.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import run_app, with_pipeline
from dagkit.graph import AsOf, Graph
from dagkit.node import Mode, Node
from dagkit.runners import RunReport
from dagkit.sinks import LayerFileSink, Register, RerunSink
from dagkit.sources import SegmentRef
from dagkit.verify import load_card
from simplecv.rerun_log_utils import RerunTyroConfig

from robocap_live.graphs.hands_layer import hands_layer_graph, segment_source
from robocap_live.nodes.hands import LAYER, HandsNode, HandsParams
from robocap_live.types import LayerRun
from robocap_live.verify import PACKAGE_DIR, case_hands, case_segment


@dataclass
class Config:
    rr_config: RerunTyroConfig
    segment: SegmentRef
    """A robocap catalog segment (``robocap__<device>__s<session>``) with ``base`` and ``slam_rs`` layers."""
    output_dir: Path
    """The dataset's hands-layer directory, visible at the same path to this host and to the catalog server."""
    hands: HandsParams
    register: Register = "new"
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"


def blueprint(graph: Graph) -> rrb.Blueprint:
    """The run health, one plot per series as the viewer laid it out before the graph view (on ``wall_time``), then the
    "pipeline (debug)" tab: the hands a windowed run copies in (``video_time``) and the hands step's own health."""
    series: list[str] = [
        f"{e.dst.node}/{what}/{e.dst.port}" for e in graph.edges if not isinstance(e.kind, AsOf) for what in ("drops", "queue_depth")
    ]
    series += [f"{name}/step_ms" for name, spec in graph.nodes.items() if spec.kind == "node"]
    normal = rrb.Blueprint(rrb.Grid(*(rrb.TimeSeriesView(origin=f"/dagkit/health/{s}", name=s) for s in series)), rrb.TimePanel(timeline="wall_time"))
    outputs: list[rrb.View] = [
        rrb.Spatial3DView(origin="/world", contents="+ /world/hands/**", name="2 hands · hands in 3D"),
        rrb.Spatial2DView(origin="/world/rig_00/cam_00/pinhole", name="2 hands · cam_00 overlays"),
        rrb.TimeSeriesView(origin="/dagkit/health/hands", name="2 hands · step ms and queue"),
    ]
    return with_pipeline(normal, outputs)


def build(config: Config, recording: rr.RecordingStream) -> Iterator[Graph]:
    """The hands-layer graph; once it ran, a windowed run's layer goes to ``recording`` too (a whole segment's is ~640 MB)."""
    publish = LayerFileSink(config.segment, LAYER, output_dir=config.output_dir, register=config.register)
    sinks: dict[str, Node] = {
        "publish": publish,
        "viewer": RerunSink(recording, {"run": LayerRun}, {"run": "/hands_layer/run"}, timeline=config.segment.timeline),
    }
    graph: Graph = hands_layer_graph(segment_source(config.segment), HandsNode(config.segment, config.hands, config.output_dir), sinks)
    recording.send_blueprint(blueprint(graph))
    yield graph
    if config.segment.window_ns is not None and publish.published is not None:
        recording.send_chunks(rr.chunk.RrdReader(publish.published).stream())


def examples() -> list[Config]:
    """The gradio example, a known-good input: VERIFY.toml's short case (5 s of the README sample on CPU ONNX Runtime), unregistered."""
    case = next(c for c in load_card(PACKAGE_DIR / "VERIFY.toml").graph[0].cases if c.name == "short")
    return [Config(RerunTyroConfig(headless=True), case_segment(case), PACKAGE_DIR / "data" / "hands-layer", case_hands(case), register="off")]


def main(config: Config) -> RunReport:
    return run_app(config, build)
