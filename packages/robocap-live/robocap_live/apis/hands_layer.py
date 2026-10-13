"""Track one robocap catalog segment's hands with robocap-live's Rust pipeline and publish its ``hands`` layer (SPEC 5.3).

The finished file is published under ``output_dir/<segment>/`` and registered as the segment's ``hands`` layer (``register``:
``new`` only when it has none, ``replace`` for a production refresh, ``off``). A windowed run (``--segment.window-ns``) is a
smoke run, never registered. The run's counts go to the viewer at ``/hands_layer/run``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import rerun as rr
from dagkit.app import run_app
from dagkit.graph import Graph
from dagkit.node import Mode, Node
from dagkit.runners import RunReport
from dagkit.sinks import LayerFileSink, Register, RerunSink
from dagkit.sources import SegmentRef
from simplecv.rerun_log_utils import RerunTyroConfig

from robocap_live.graphs.hands_layer import hands_layer_graph, segment_source
from robocap_live.nodes.hands import LAYER, HandsNode, HandsParams
from robocap_live.types import LayerRun


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


def build(config: Config, recording: rr.RecordingStream) -> Graph:
    sinks: dict[str, Node] = {
        "publish": LayerFileSink(config.segment, LAYER, output_dir=config.output_dir, register=config.register),
        "viewer": RerunSink(recording, {"run": LayerRun}, {"run": "/hands_layer/run"}, timeline=config.segment.timeline),
    }
    return hands_layer_graph(segment_source(config.segment), HandsNode(config.segment, config.hands, config.output_dir), sinks)


def main(config: Config) -> RunReport:
    return run_app(config, build)
