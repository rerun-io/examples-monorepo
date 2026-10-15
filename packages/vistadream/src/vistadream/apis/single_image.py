"""VistaDream single-image App: one photo to a 3D Gaussian scene and a camera path through it (graph ``graphs/single_image.py``).

Stages: ``no-outpaint`` (MoGe depth of the image, 100 optimisation steps), ``outpaint`` (FLUX.1 Fill grows the canvas first)
and ``coarse`` (then adds inpainted views along a spiral, 500 steps each). The scene, the training views and the rendered
path stream to the ``rr_config`` viewer; ``output_ply`` also saves the scene. The gradio app is generated from ``Config``;
its example is the ``VERIFY.toml`` short case (``short_case``). The viewer opens on the normal view; the "pipeline (debug)"
tab shows the graph, its guide and four of its outputs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import run_app, with_pipeline
from dagkit.doc import describe
from dagkit.graph import Graph
from dagkit.node import Mode, Node, PortType
from dagkit.runners import RunReport
from dagkit.sinks import RerunSink
from dagkit.verify import Case, load_card
from simplecv.rerun_log_utils import RerunTyroConfig

from vistadream.graphs.single_image import single_image_graph
from vistadream.models.flux import FluxConfig
from vistadream.nodes.prepare import DepthNode, ImageSource, OutpaintNode
from vistadream.nodes.splats import ExpandNode, InitNode, PlySink, RenderNode
from vistadream.types import CameraView, Scene, Stage

PACKAGE_DIR: Path = Path(__file__).resolve().parents[3]
"""``packages/vistadream``."""
PORTS: dict[str, PortType] = {"scene": Scene, "view": CameraView}
ROOTS: dict[str, str] = {"scene": "/world", "view": "/world/render"}
"""The scene (splats at ``/world/splats``, training views at ``/world/input``, ``/world/outpaint``, ``/world/view_<i>``) and the
render path."""


@dataclass
class Config:
    rr_config: RerunTyroConfig
    image_path: Path = PACKAGE_DIR / "data/office/IMG_4029.jpg"
    """The photo (default: the package's sample)."""
    stage: Stage = "no-outpaint"
    """The outpaint and coarse stages load FLUX.1-Fill-dev (about 34 GB on first use)."""
    max_resolution: Literal[512, 1024, 1920] = 512
    """Long side of the working image (both sides become multiples of 32)."""
    expansion_percent: float = 0.3
    """Outpainting: how much the canvas grows (0.3 = 15 % of the image on every side)."""
    n_frames: int = 10
    """Coarse stage: training views at most (input and outpainted canvas included)."""
    output_ply: Path | None = field(default=None, metadata={"form": "save"})
    """Save the final scene as a 3DGS PLY here."""
    seed: int = 0
    """Seed of the order in which optimisation steps visit the training views."""
    flux: FluxConfig = field(default_factory=FluxConfig)
    runner: Literal["sequential", "threaded"] = "threaded"
    """dagkit runner; both give the same outputs (lossless)."""
    mode: Mode = "lossless"
    """lossless: every step sees every item; realtime: busy steps drop old items."""


def case_config(case: Case, rr_config: RerunTyroConfig) -> Config:
    """A ``VERIFY.toml`` case as a Config (params the case leaves out keep their default)."""
    assert case.input is not None, case
    return Config(rr_config, PACKAGE_DIR / case.input, **case.params)


def short_case(rr_config: RerunTyroConfig) -> Config:
    """The ``VERIFY.toml`` short case as a Config: the gradio example, one click runs a known-good input."""
    [case] = [c for block in load_card(PACKAGE_DIR / "VERIFY.toml").graph for c in block.cases if c.clip == "short"]
    return case_config(case, rr_config)


def blueprint(config: Config, graph: Graph) -> rrb.Blueprint:
    """The normal view, 3D: splats and camera frusta (explicit image exclusions: filters have no mid-path wildcards) | 2D:
    input, depth, canvas, path; then the "pipeline (debug)" tab, whose output views carry the graph's step numbers."""
    cameras: list[str] = ["input", "outpaint", "render", *(f"view_{i:02d}" for i in range(2, config.n_frames))]
    images: list[str] = [f"- /world/{camera}/pinhole/{image}" for camera in cameras for image in ("rgb", "depth", "valid", "supervise")]
    number: dict[str, int] = {n.name: n.number for n in describe(graph).nodes}
    final: str = "expand" if "expand" in graph.nodes else "init"
    node, what, origin = ("image", "input", "/world/input/pinhole/rgb") if config.stage == "no-outpaint" else ("outpaint", "canvas", "/world/outpaint/pinhole/rgb")
    outputs: list[rrb.View] = [
        rrb.Spatial2DView(origin=origin, name=f"{number[node]} {node} · {what}"),
        rrb.Spatial2DView(origin="/world/input/pinhole/depth", name=f"{number['depth']} depth · MoGe depth"),
        rrb.Spatial3DView(origin="/world", name=f"{number[final]} {final} · scene", contents=["+ /world/**", *images]),
        rrb.Spatial2DView(origin="/world/render/pinhole/rgb", name=f"{number['render']} render · path"),
    ]
    view3d = rrb.Spatial3DView(origin="/world", name="Gaussian splats", contents=["+ /world/**", *images])
    grid = rrb.Grid(
        rrb.Spatial2DView(origin="/world/input/pinhole/rgb", name="input"),
        rrb.Spatial2DView(origin="/world/input/pinhole/depth", name="MoGe depth"),
        rrb.Spatial2DView(origin="/world/outpaint/pinhole/rgb", name="outpainted canvas"),
        rrb.Spatial2DView(origin="/world/render/pinhole/rgb", name="render path"),
        grid_columns=2,
    )
    return with_pipeline(rrb.Blueprint(rrb.Horizontal(view3d, grid), collapse_panels=True), outputs)


def build(config: Config, recording: rr.RecordingStream) -> Graph:
    """Models load here, before the run (MoGe v1; FLUX.1 Fill unless ``no-outpaint``); the blueprint names the graph's steps,
    so it follows the graph."""
    from monopriors.models.relative_depth.moge_v1 import MoGeV1Predictor

    from vistadream.models.flux import FluxInpainting

    recording.log("/world", rr.ViewCoordinates.RDF, static=True)
    sinks: dict[str, Node] = {"viewer": RerunSink(recording, PORTS, ROOTS)}
    if config.output_ply is not None:
        sinks["ply"] = PlySink(config.output_ply)
    moge = MoGeV1Predictor(device="cuda")
    inpaint: FluxInpainting | None = None if config.stage == "no-outpaint" else FluxInpainting(config.flux)
    expand: ExpandNode | None = ExpandNode(inpaint, moge, config.n_frames, seed=config.seed) if config.stage == "coarse" and inpaint is not None else None
    source = ImageSource(config.image_path, config.max_resolution)
    graph: Graph = single_image_graph(source, OutpaintNode(inpaint, config.expansion_percent), DepthNode(moge), InitNode(seed=config.seed), expand, RenderNode(), sinks)
    recording.send_blueprint(blueprint(config, graph))
    return graph


def main(config: Config) -> RunReport:
    return run_app(config, build)
