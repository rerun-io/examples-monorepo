"""The two exo-calib graphs (SPEC 5.1): ``init`` (Stage A, run once) and ``refine`` (Stage B + C/D, plus Stage E on the last pass).

Models are graph resources built from the factories the App passes in, so tests inject fakes. A refine pass reads its
frames (the metric rescale) through its video source's own ``SegmentFrames``: each exo video is fetched once per pass.
"""

from collections.abc import Callable, Mapping

from dagkit.graph import AsOf, Block, Graph, PortRef
from dagkit.node import Event, Node
from dagkit.sources import CatalogSegmentSource, Column, IterSource, SegmentRef
from dagkit.time import STATIC
from simplecv.sources.catalog_video import VideoSource
from simplecv.types.rig import RigCameras

from exo_calib.nodes.init import InitNode
from exo_calib.nodes.kp2d import Kp2dNode
from exo_calib.nodes.refine import RefineNode
from exo_calib.nodes.report import ReportNode
from exo_calib.types import MultiViewKeypoints, RigLayout

Factories = Mapping[str, Callable[[], object]]
KP2D_ROOT: str = "/world/exocalib/kp2d"


def init_graph(segment: SegmentRef, cameras: tuple[str, ...], frame_index: int, models: Factories, sinks: Mapping[str, Node]) -> Graph:
    graph = Graph("exocalib-init", about="Stage A, run once: metric, gravity-aligned exo cameras from one synchronized frameset of the exo videos. The refine graph starts from this rig.")
    for name in ("g3t", "moge"):
        graph.resource(name, models[name])
    video = graph.source("video", VideoSource.catalog(segment, cameras, decode="nvdec_rgb", sample=(frame_index,)))
    init = graph.node("init", InitNode(), needs=("g3t", "moge"))
    graph.edge(video["frames"], init["frames"])
    graph.sinks({"init": init["init"]}, sinks)
    return graph


def refine_graph(
    segment: SegmentRef,
    layout: RigLayout,
    seed: tuple[RigCameras, RigCameras],
    frame_budget: int,
    models: Factories,
    sinks: Mapping[str, Node],
    *,
    report: bool,
    resume_kp2d: bool = False,
) -> Graph:
    """``seed`` = (Stage A rig, tracking rig); both passes refine against Stage A, only the tracking rig changes.

    ``resume_kp2d`` reads the Stage B records (and the GT rig) back from the catalog instead of decoding and detecting.
    The pass without the report (pass 1) is its own graph, ``exocalib-refine-pass1``: its sinks differ, and the viewer
    keeps each graph's picture and run health apart by name.
    """
    graph = Graph(
        "exocalib-refine" if report else "exocalib-refine-pass1",
        about="Stages B-E, run twice: 2D keypoints tracked through the rig in every view, then bundle adjustment of the Stage A rig on them; the second pass tracks through the first pass's rig and scores both rigs against GT.",
    )
    for name in ("detector", "pose", "moge"):
        graph.resource(name, models[name])
    statics: dict[str, object] = {"init": seed[0], "tracking_rig": seed[1], "rig_layout": layout}
    seed_ports: dict[str, type] = {"init": RigCameras, "tracking_rig": RigCameras, "rig_layout": RigLayout}
    seed_ref = graph.source("seed", IterSource(seed_ports, lambda: (Event(STATIC, p, v) for p, v in statics.items()), clock=segment.timeline))
    if resume_kp2d:
        catalog = graph.source("catalog", CatalogSegmentSource(segment, {"kp2d": Column(MultiViewKeypoints, KP2D_ROOT), "rig": Column(RigCameras, "/world", True)}))
        records, gt = catalog["kp2d"], catalog["rig"]
        graph.resource("frames", models["frames"])
    else:
        source = VideoSource.catalog(segment, layout.exo_camera_names, decode="nvdec_rgb", sample=frame_budget)
        graph.resource("frames", lambda: source.frames)
        video = graph.source("video", source)
        kp2d = graph.node("kp2d", Kp2dNode(), needs=("detector", "pose"))
        graph.edge(video["frames"], kp2d["frames"], Block(2))
        graph.edge(seed_ref["tracking_rig"], kp2d["tracking_rig"], AsOf())
        records, gt = kp2d["kp2d"], video["rig"]
    refine = graph.node("refine", RefineNode(), needs=("frames", "moge"))
    graph.edge(records, refine["kp2d"])
    for port in ("init", "rig_layout"):
        graph.edge(seed_ref[port], refine[port], AsOf())
    ports: dict[str, PortRef] = {"kp2d": records} | {port: refine[port] for port in RefineNode.outputs}
    if report:
        rep = graph.node("report", ReportNode(segment))
        for port in ("refined", "diag"):
            graph.edge(refine[port], rep[port])
        for src, port in ((seed_ref["init"], "init"), (gt, "gt"), (seed_ref["rig_layout"], "rig_layout")):
            graph.edge(src, rep[port], AsOf())
        ports |= {"align": rep["align"], "eval": rep["eval"]}
    graph.sinks(ports, sinks)
    return graph
