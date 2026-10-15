"""Graph-tier check of vistadream's single-image graph, driven by ``VERIFY.toml`` through ``python -m dagkit.verify``.

No depth GT is registered on the catalog, so the reference is the PRE-PORT golden of each case. The pre-port pipeline was
broken at import (``@dataclass`` on the wrong definition, a removed ``monopriors`` factory); ``pre``/``pre2`` ran the OLD
``SingleImagePipeline`` with the smallest fix applied and never committed (see ``VERIFY.toml``). ``run`` drives the App's graph
(``apis/single_image.build``); ``score`` reads only golden rrds, never the pipeline.

Golden contract (``recording_id`` = case name; clock ``video_time``: the image at t = 0, path step i at i / 30 s):

- ``/world/splats``: the final scene as ``rr.GaussianSplats3D`` (linear scales, xyzw quaternions, RGBA = base colour and
  opacity), t = 0.
- ``/world/input`` (``Transform3D``, identity) and ``/world/input/pinhole`` (``Pinhole``): the input camera, t = 0, with
  ``rgb`` (``Image`` uint8 RGB, the processed input), ``depth`` (``DepthImage`` float32, MoGe), ``valid`` (``Image`` L
  uint8 0/255, MoGe validity), and the final scene rendered at that camera: ``render_rgb`` (``Image`` uint8 RGB) and
  ``render_depth`` (``DepthImage`` float32).
- ``/world/render`` (``Transform3D``, camera from world) and ``/world/render/pinhole`` (``Pinhole``) with ``rgb``
  (``Image`` uint8 RGB): one render per trajectory step.
- ``/verify/single_image/run``: static ``AnyValues`` (``wall_s``, ``n_splats``, ``stage``, ``code``).
"""

import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from dagkit.catalog import served
from dagkit.graph import Graph
from dagkit.node import Emit
from dagkit.runners import run_sequential
from dagkit.sinks import Latest
from dagkit.types import cells
from dagkit.verify import Case, Phase, pre_golden
from jaxtyping import Bool
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from vistadream.apis.single_image import Config, build, case_config
from vistadream.models.flux import FluxConfig
from vistadream.ops.splats import render
from vistadream.types import SceneViews, Splats, images

TIMELINE: str = "video_time"
INPUT: str = "/world/input/pinhole"
RENDER: str = "/world/render/pinhole"


class InputRender(Latest):
    """Verify-only sink: the final scene rendered at the input camera, beside the input image (``render_rgb``, ``render_depth``)."""

    def __init__(self, recording: rr.RecordingStream) -> None:
        super().__init__({"scene": Splats, "views": SceneViews})
        self.recording: rr.RecordingStream = recording
        self.n_splats: int = 0

    def flush(self, out: Emit) -> None:
        splats, views = self.values.pop("scene", None), self.values.pop("views", None)
        if not isinstance(splats, Splats) or not isinstance(views, SceneViews):
            return
        height, width = views.input.rgb.shape[:2]
        with torch.no_grad():
            rgb, depth, _ = render(splats, views.input.K, views.input.cam_T_world, width, height)
        self.recording.set_time(TIMELINE, duration=np.timedelta64(0, "ns"))
        self.recording.log(f"{INPUT}/render_rgb", rr.Image((rgb.clamp(0, 1) * 255.0).round().byte().cpu().numpy(), color_model="RGB"))
        self.recording.log(f"{INPUT}/render_depth", rr.DepthImage(depth.float().cpu().numpy(), meter=1.0))
        self.n_splats = len(splats.centers)


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """The App's graph on the case's image (sequential, lossless, seeded RNGs) plus ``InputRender``, fed what the viewer is
    fed; every output goes to ``recording``."""
    config: Config = replace(case_config(case, RerunTyroConfig(headless=True)), flux=FluxConfig(local_files_only=True))  # FLUX.1-Fill-dev is ~34 GB: a verify run never downloads it
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    started: float = time.perf_counter()
    graph: Graph = build(config, recording)
    input_render = InputRender(recording)
    graph.sinks({edge.dst.port: edge.src for edge in graph.edges if edge.dst.node == "viewer"}, {"input_render": input_render})
    report = run_sequential(graph)
    wall_s: float = time.perf_counter() - started
    recording.log("/verify/single_image/run", rr.AnyValues(wall_s=wall_s, n_splats=float(input_render.n_splats), stage=config.stage, code="graph"), static=True)
    print(f"{case.name} {phase}: {report.status}, {input_render.n_splats} splats in {wall_s:.1f} s")


def _read(golden: Path) -> pa.Table:
    with served(golden) as dataset:
        return dataset.filter_contents([f"{INPUT}/**", f"{RENDER}/**", "/world/splats"]).reader(index=TIMELINE).to_arrow_table()


def score(case: Case, golden: Path) -> dict[str, float]:
    """Banded: ``input_psnr_db`` (the scene rendered at the input camera vs the processed input image, all pixels),
    ``input_depth_absrel`` (that render's depth vs MoGe's on MoGe-valid pixels) and ``novel_rgb_mae`` (trajectory renders
    vs the case's ``pre`` golden at the same steps, mean |difference| in 0-255 levels; ``pre`` itself scores 0).
    Reported, not banded: ``n_splats`` and ``novel_frames``."""
    table: pa.Table = _read(golden)
    [rgb] = images(table, f"{INPUT}/rgb", "Image", TIMELINE, np.uint8).values()
    [rendered] = images(table, f"{INPUT}/render_rgb", "Image", TIMELINE, np.uint8).values()
    [depth] = images(table, f"{INPUT}/depth", "DepthImage", TIMELINE, np.float32).values()
    [render_depth] = images(table, f"{INPUT}/render_depth", "DepthImage", TIMELINE, np.float32).values()
    [valid_u8] = images(table, f"{INPUT}/valid", "Image", TIMELINE, np.uint8).values()
    mse: float = float(np.mean((rendered.astype(np.float64) - rgb.astype(np.float64)) ** 2))
    valid: Bool[ndarray, "h w"] = (valid_u8 > 0) & np.isfinite(depth) & (depth > 0)  # a non-finite render depth there: NaN, fails
    absrel: float = float(np.mean(np.abs(render_depth[valid] - depth[valid]) / depth[valid])) if valid.any() else float("nan")

    reference: Path = pre_golden(case, golden)
    ours: dict[int, ndarray] = images(table, f"{RENDER}/rgb", "Image", TIMELINE, np.uint8)
    theirs: dict[int, ndarray] = ours if reference == golden else images(_read(reference), f"{RENDER}/rgb", "Image", TIMELINE, np.uint8)
    common: list[int] = sorted(ours.keys() & theirs.keys())
    errors: list[float] = [
        float(np.mean(np.abs(ours[t].astype(np.float64) - theirs[t].astype(np.float64)))) for t in common if ours[t].shape == theirs[t].shape
    ]
    n_splats: int = sum(len(c.values) for c in cells(table, "/world/splats:GaussianSplats3D:centers", TIMELINE).values())
    return {
        "input_psnr_db": 10.0 * float(np.log10(255.0**2 / mse)) if mse > 0 else float("inf"),
        "input_depth_absrel": absrel,
        "novel_rgb_mae": float(np.mean(errors)) if errors and len(errors) == len(theirs) == len(ours) else float("nan"),
        "n_splats": float(n_splats),
        "novel_frames": float(len(ours)),
    }
