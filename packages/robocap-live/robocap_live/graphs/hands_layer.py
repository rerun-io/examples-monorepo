"""The hands-layer graph (SPEC 5.3, lossless): one catalog segment's luma framesets through the ``hands`` node into the sinks.

``segment`` (``CatalogVideoSource``, ``cpu_luma``) -> ``frames`` (Block 4) -> ``hands`` <- ``cap_rig``, ``poses`` (AsOf, STATIC);
``hands`` -> ``layer`` / ``run`` -> every sink that takes that port (the ``LayerFileSink`` publishes the layer at flush).
"""

from collections.abc import Mapping

from dagkit.graph import AsOf, Block, Graph
from dagkit.node import Node, Source
from dagkit.sources import Column, SegmentRef
from simplecv.sources.catalog_video import CatalogVideoSource

from robocap_live.nodes.hands import HandsNode
from robocap_live.types import RIG_ROOT, CapRig, CatalogPoses

CAMERAS: tuple[str, ...] = tuple(f"rig_00/cam_{i:02d}" for i in range(6))  # under /world, in the core's index order
GROUP_TOLERANCE_NS: int = 3_000_000  # a frameset holds each camera's next frame within 3 ms of its opening, as live


def segment_source(segment: SegmentRef, group_tolerance_ns: int = GROUP_TOLERANCE_NS) -> CatalogVideoSource:
    """The segment's raw luma framesets, its base-layer rig and its whole ``slam_rs`` pose table."""
    statics: dict[str, Column] = {"cap_rig": Column(CapRig, RIG_ROOT, static=True), "poses": Column(CatalogPoses, RIG_ROOT)}
    return CatalogVideoSource(segment, CAMERAS, decode="cpu_luma", group_tolerance_ns=group_tolerance_ns, statics=statics)


def hands_layer_graph(source: Source, hands: HandsNode, sinks: Mapping[str, Node]) -> Graph:
    graph = Graph("robocap-hands-layer")
    segment = graph.source("segment", source)
    node = graph.node("hands", hands)
    graph.edge(segment["frames"], node["frames"], Block(4))
    for port in ("cap_rig", "poses"):
        graph.edge(segment[port], node[port], AsOf())
    for name, sink in sinks.items():
        ref = graph.sink(name, sink)
        for port in sink.inputs:
            graph.edge(node[port], ref[port])
    return graph
