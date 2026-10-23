"""The two exo-calib graphs (SPEC 5.1): ``init`` (Stage A, run once) and ``refine`` (Stage B + C/D, plus the body (F) and the
report (E) on the last pass).

Models are graph resources built from the factories the App passes in, so tests inject fakes. Every video source and the
metric rescale read the App's one ``SegmentFrames`` (``models["frames"]``): each exo video is fetched once per App run.
"""

from collections.abc import Callable, Mapping
from typing import cast

from dagkit.graph import AsOf, Block, Graph, PortRef
from dagkit.node import Event, Node, Source
from dagkit.sources import IterSource, SegmentRef
from dagkit.time import STATIC
from simplecv.sources.catalog_video import FrameProvider, VideoSource
from simplecv.types.rig import RigCameras

from exo_calib.nodes.body import BodyNode
from exo_calib.nodes.init import InitNode
from exo_calib.nodes.kp2d import Kp2dNode
from exo_calib.nodes.refine import RefineNode
from exo_calib.nodes.report import ReportNode
from exo_calib.types import RigLayout

Factories = Mapping[str, Callable[[], object]]
KP2D_ROOT: str = "/world/exocalib/kp2d"


def init_graph(segment: SegmentRef, cameras: tuple[str, ...], frame_index: int | None, models: Factories, sinks: Mapping[str, Node]) -> Graph:
    graph = Graph("exocalib-init", about="Stage A, run once: metric, gravity-aligned exo cameras from one synchronized frameset of the exo videos. The refine graph starts from this rig.")
    for name in ("g3t", "moge"):
        graph.resource(name, models[name])
    video = graph.source("video", VideoSource.catalog(segment, cameras, decode="nvdec_rgb", sample=1 if frame_index is None else (frame_index,), frames=cast(FrameProvider, models["frames"]())))
    init = graph.node("init", InitNode(), needs=("g3t", "moge"))
    graph.edge(video["frames"], init["frames"])
    graph.sinks({port: init[port] for port in InitNode.outputs}, sinks)
    return graph


def rig_seed(segment: SegmentRef, *, init: object, tracking_rig: object) -> IterSource:
    """The refine graph's seed from memory: the Stage A rig and the rig Stage B tracks through, at STATIC (the graph checks
    that both are ``RigCameras``)."""
    rigs: dict[str, object] = {"init": init, "tracking_rig": tracking_rig}
    return IterSource(dict.fromkeys(rigs, RigCameras), lambda: (Event(STATIC, port, rig) for port, rig in rigs.items()), clock=segment.timeline)


def refine_graph(
    segment: SegmentRef, layout: RigLayout, seed: Source, frame_budget: int, models: Factories, sinks: Mapping[str, Node], *, report: bool
) -> Graph:
    """``seed`` hands in the STATIC Stage A rig ``init`` (both passes refine against it) and the ``tracking_rig`` Stage B
    tracks through: from memory (``rig_seed``) or, on a resume, read back from their stored layers. A seed that carries the
    stored Stage B records ``kp2d`` and the segment's GT ``rig`` instead resumes past Stage B: the graph then decodes and
    detects nothing, and declares no detector and no pose engine.

    The pass without the report (pass 1) is its own graph, ``exocalib-refine-pass1``: its sinks differ, and the viewer
    keeps each graph's picture and run health apart by name. Its sinks may also take ``init``, to hand it to pass 2.
    """
    graph = Graph(
        "exocalib-refine" if report else "exocalib-refine-pass1",
        about="Stages B-F, run twice: 2D keypoints tracked through the rig in every view, then bundle adjustment of the Stage A rig on them; the second pass tracks through the first pass's rig, fits the person's SMPL-X body through it and scores both rigs against GT.",
    )
    resumed: bool = "kp2d" in seed.outputs
    for name in ("moge",) if resumed else ("detector", "pose", "moge"):
        graph.resource(name, models[name])
    graph.resource("frames", models["frames"])  # the App's one reader: the video source fetched its packets, the rescale decodes from them
    seed_ref = graph.source("seed", seed)
    if resumed:
        records, gt = seed_ref["kp2d"], seed_ref["rig"]
    else:
        video = graph.source("video", VideoSource.catalog(segment, layout.exo_camera_names, decode="nvdec_rgb", sample=frame_budget, frames=cast(FrameProvider, models["frames"]())))
        kp2d = graph.node("kp2d", Kp2dNode(layout), needs=("detector", "pose"))
        graph.edge(video["frames"], kp2d["frames"], Block(2))
        graph.edge(seed_ref["tracking_rig"], kp2d["tracking_rig"], AsOf())
        records, gt = kp2d["kp2d"], video["rig"]
    refine = graph.node("refine", RefineNode(layout), needs=("frames", "moge"))
    graph.edge(records, refine["kp2d"])
    graph.edge(seed_ref["init"], refine["init"], AsOf())
    ports: dict[str, PortRef] = {"kp2d": records, "init": seed_ref["init"]} | {port: refine[port] for port in RefineNode.outputs}
    if report:
        graph.resource("smplx", models["smplx"])
        body = graph.node("body", BodyNode(layout), needs=("smplx", "frames"))
        graph.edge(records, body["kp2d"])
        graph.edge(refine["refined"], body["rig"])
        ports |= {port: body[port] for port in BodyNode.outputs}
        rep = graph.node("report", ReportNode(segment, layout))
        for port in ("refined", "diag"):
            graph.edge(refine[port], rep[port])
        for src, port in ((seed_ref["init"], "init"), (gt, "gt")):
            graph.edge(src, rep[port], AsOf())
        ports |= {port: rep[port] for port in ReportNode.outputs}
    graph.sinks(ports, sinks)
    return graph
