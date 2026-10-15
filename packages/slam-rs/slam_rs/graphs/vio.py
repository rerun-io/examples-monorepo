"""The VIO graph (lossless); its picture and guide: ``GRAPH.md`` (tests/test_vio_graph.py) and every run's "pipeline (debug)" tab."""

from collections.abc import Generator, Mapping
from dataclasses import replace
from typing import Literal

from dagkit.graph import AsOf, Block, Graph
from dagkit.node import Event, Node, Source
from dagkit.sources import IterSource, SegmentRef
from dagkit.time import STATIC

from slam_rs.catalog_feed import CatalogSegment, Frameset, ImuStream, RigProfile, SegmentFeed, open_segment
from slam_rs.nodes.vio import VioNode
from slam_rs.types import VioRig


class VioSource(IterSource):
    """Decode the VIO cameras, with the IMU.

    STATIC ``rig``, then per frameset its IMU batch (``imu``: the samples since the previous frameset through the first one past
    this one) and its gray8 images (``frames``), both at the frameset's ``video_time``. VIO initialises from rest, so the stream
    starts at the segment's first frameset; a window only ends it (frameset times up to its stop)."""

    def __init__(self, segment: SegmentRef, rig: RigProfile, decoder: Literal["cpu", "cuda"]) -> None:
        def events() -> Generator[Event, None, None]:
            source = CatalogSegment(segment.catalog_url, segment.dataset, segment.segment)
            feed: SegmentFeed
            with open_segment(source, profile=rig, cache_video=True, decode_device=decoder, include_ground_truth=False) as feed:
                offset_ns: int = feed.imu.cam_time_offset_ns  # video_time -> the estimator's clock
                yield Event(STATIC, "rig", VioRig(feed.cameras, feed.imu, feed.export_offset_ns))
                stop_ns: int | None = None if segment.window_ns is None else segment.window_ns[1] + offset_ns
                for f in feed.framesets(stop_ns):
                    if stop_ns is not None and f.t_ns > stop_ns:
                        return
                    if len(f.imu):
                        yield Event(f.t_ns - offset_ns, "imu", f.imu)
                    yield Event(f.t_ns - offset_ns, "frames", replace(f, imu=f.imu.between(f.t_ns, f.t_ns)))

        super().__init__({"rig": VioRig, "imu": ImuStream, "frames": Frameset}, events, clock=segment.timeline)


def vio_graph(source: Source, vio: VioNode, sinks: Mapping[str, Node]) -> Graph:
    graph = Graph("slam-rs-vio", about="One catalog segment's cameras and IMU, on one time-ordered stream, through slam-rs's visual-inertial "
                  "odometry into the segment's `slam_rs` pose layer: a file the publish step moves into place and registers when asked.")  # fmt: skip
    segment = graph.source("segment", source)
    node = graph.node("vio", vio)
    for port in ("imu", "frames"):
        graph.edge(segment[port], node[port], Block(4))
    graph.edge(segment["rig"], node["rig"], AsOf())
    graph.sinks({port: node[port] for port in vio.outputs}, sinks)
    return graph
