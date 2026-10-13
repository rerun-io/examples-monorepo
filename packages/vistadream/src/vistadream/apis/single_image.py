"""VistaDream single-image App: one photo to a 3D Gaussian scene and a camera path through it (graph ``graphs/single_image.py``).

Stages: ``no-outpaint`` (MoGe depth of the image, 100 optimisation steps), ``outpaint`` (FLUX.1 Fill grows the canvas first)
and ``coarse`` (then adds inpainted views along a spiral, 500 steps each). The scene, the training views and the rendered
path stream to the ``rr_config`` viewer; ``output_ply`` also saves the scene. The gradio app is generated from ``Config``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import run_app
from dagkit.graph import Graph
from dagkit.node import Mode, Node, PortType
from dagkit.runners import RunReport
from dagkit.sinks import RerunSink
from simplecv.rerun_log_utils import RerunTyroConfig

from vistadream.graphs.single_image import single_image_graph
from vistadream.models.flux import FluxConfig
from vistadream.nodes.prepare import DepthNode, OutpaintNode, image_source
from vistadream.nodes.splats import ExpandNode, InitNode, PlySink, RenderNode
from vistadream.types import CameraView, SceneViews, Splats, Stage

PACKAGE_DIR: Path = Path(__file__).resolve().parents[3]
"""``packages/vistadream``."""
PORTS: dict[str, PortType] = {"scene": Splats, "views": SceneViews, "view": CameraView}
ROOTS: dict[str, str] = {"scene": "/world/splats", "views": "/world", "view": "/world/render"}
"""The scene's splats, its training views (``/world/input``, ``/world/outpaint``, ``/world/view_<i>``) and the render path."""


@dataclass
class Config:
    rr_config: RerunTyroConfig
    image_path: Path = PACKAGE_DIR / "data/office/IMG_4029.jpg"
    """The photo (default: the package's sample)."""
    stage: Stage = "no-outpaint"
    """``outpaint`` and ``coarse`` load FLUX.1-Fill-dev (about 34 GB on first use)."""
    max_resolution: Literal[512, 1024, 1920] = 512
    """Long side of the working image (both sides become multiples of 32)."""
    expansion_percent: float = 0.3
    """Outpainting: how much the canvas grows (0.3 = 15 % of the image on every side)."""
    n_frames: int = 10
    """Coarse stage: training views at most (input and outpainted canvas included)."""
    output_ply: Path | None = None
    """Save the final scene as a 3DGS PLY here."""
    seed: int = 0
    """Seed of the order in which optimisation steps visit the training views."""
    flux: FluxConfig = field(default_factory=FluxConfig)
    runner: Literal["sequential", "threaded"] = "threaded"
    """dagkit runner; both give the same outputs (lossless)."""
    mode: Mode = "lossless"


def nodes(stage: Stage, expansion_percent: float, n_frames: int, seed: int, flux: FluxConfig) -> tuple[OutpaintNode, DepthNode, InitNode, ExpandNode | None, RenderNode]:
    """The graph's nodes with their models (MoGe v1; FLUX.1 Fill unless ``no-outpaint``), shared by ``build`` and ``verify``."""
    from monopriors.models.relative_depth.moge_v1 import MoGeV1Predictor

    from vistadream.models.flux import FluxInpainting

    moge = MoGeV1Predictor(device="cuda")
    inpaint: FluxInpainting | None = None if stage == "no-outpaint" else FluxInpainting(flux)
    expand: ExpandNode | None = ExpandNode(inpaint, moge, n_frames, seed=seed) if stage == "coarse" and inpaint is not None else None
    return OutpaintNode(inpaint, expansion_percent), DepthNode(moge), InitNode(seed=seed), expand, RenderNode()


def blueprint(n_frames: int) -> rrb.Blueprint:
    """3D: splats and camera frusta (explicit image exclusions: filters have no mid-path wildcards) | 2D: input, depth, canvas, path."""
    cameras: list[str] = ["input", "outpaint", "render", *(f"view_{i:02d}" for i in range(2, n_frames))]
    images: list[str] = [f"- /world/{camera}/pinhole/{image}" for camera in cameras for image in ("rgb", "depth", "valid", "supervise")]
    view3d = rrb.Spatial3DView(origin="/world", name="Gaussian splats", contents=["+ /world/**", *images])
    grid = rrb.Grid(
        rrb.Spatial2DView(origin="/world/input/pinhole/rgb", name="input"),
        rrb.Spatial2DView(origin="/world/input/pinhole/depth", name="MoGe depth"),
        rrb.Spatial2DView(origin="/world/outpaint/pinhole/rgb", name="outpainted canvas"),
        rrb.Spatial2DView(origin="/world/render/pinhole/rgb", name="render path"),
        grid_columns=2,
    )
    return rrb.Blueprint(rrb.Horizontal(view3d, grid), collapse_panels=True)


def build(config: Config, recording: rr.RecordingStream) -> Graph:
    """Models load here, before the run."""
    recording.log("/world", rr.ViewCoordinates.RDF, static=True)
    recording.send_blueprint(blueprint(config.n_frames))
    sinks: dict[str, Node] = {"viewer": RerunSink(recording, PORTS, ROOTS)}
    if config.output_ply is not None:
        sinks["ply"] = PlySink(config.output_ply)
    built = nodes(config.stage, config.expansion_percent, config.n_frames, config.seed, config.flux)
    return single_image_graph(image_source(config.image_path, config.max_resolution), *built, sinks)


def main(config: Config) -> RunReport:
    return run_app(config, build)
