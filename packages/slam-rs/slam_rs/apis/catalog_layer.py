"""Run slam-rs VIO over one catalog segment and publish its ``slam_rs`` pose layer under ``output_dir/<segment>/`` (``register``:
``new`` only when the segment has none, ``replace`` for a production refresh, ``off``; a windowed run never registers). The
viewer follows the rig pose live and gets the layer's trajectory at the end; the "pipeline (debug)" tab shows the graph."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import run_app, with_pipeline
from dagkit.doc import HEALTH_ROOT
from dagkit.graph import Graph
from dagkit.node import Mode
from dagkit.runners import RunReport
from dagkit.sinks import LayerFileSink, Register, RerunSink
from dagkit.sources import SegmentRef
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs.catalog_layer import POSE_SOURCE
from slam_rs.graphs.vio import VioSource, vio_graph
from slam_rs.nodes.vio import VioNode, VioParams
from slam_rs.rig import RIG_ENTITY
from slam_rs.types import RigPose


@dataclass
class Config:
    rr_config: RerunTyroConfig
    segment: SegmentRef
    """A catalog segment of robocap, msd-index, msd-odyssey or msd-g2."""
    output_dir: Path
    """The dataset's slam_rs-layer directory, visible at the same path to this host and to the catalog server."""
    vio: VioParams = VioParams()
    register: Register = "new"
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"


def vio_publish(
    segment: SegmentRef, params: VioParams, output_dir: Path, register: Register, recording: rr.RecordingStream
) -> tuple[Graph, LayerFileSink]:
    """The VIO graph with its sinks: the ``slam_rs`` layer publisher (returned too: its ``published`` file) and the live pose."""
    publish = LayerFileSink(segment, POSE_SOURCE, output_dir=output_dir, register=register)
    viewer = RerunSink(recording, {"pose": RigPose}, {"pose": RIG_ENTITY}, timeline=segment.timeline)
    return vio_graph(VioSource(segment, params), VioNode(segment, params, output_dir), {"publish": publish, "viewer": viewer}), publish


def build(config: Config, recording: rr.RecordingStream) -> Iterator[Graph]:
    """The VIO graph; once it ran, its layer (a few MB) goes to ``recording`` too."""
    graph, publish = vio_publish(config.segment, config.vio, config.output_dir, config.register, recording)
    normal = rrb.Blueprint(rrb.Spatial3DView(origin="/world", name="trajectory"), rrb.TimePanel(timeline=config.segment.timeline))
    recording.send_blueprint(with_pipeline(normal, [rrb.Spatial3DView(origin="/world", name="2 vio · rig pose"),
                                                    rrb.TimeSeriesView(origin=HEALTH_ROOT + "vio", name="2 vio · step ms and queue")]))  # fmt: skip
    yield graph
    if publish.published is not None:
        recording.send_chunks(rr.chunk.RrdReader(publish.published).stream())
        if config.register != "off" and config.segment.window_ns is None:  # where to look at the registered layer, as the old tool ended
            print(config.segment.entry().segment_url(config.segment.segment))


def main(config: Config) -> RunReport:
    return run_app(config, build)
