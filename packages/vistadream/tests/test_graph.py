"""The single-image graph with fakes (unit lane, CPU): it validates, both runners deliver the same outputs, the PLY sink writes,
the graph view shows the declared graph, the docs are complete (GRAPH.md) and the debug tab follows the normal view."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
from dagkit.graph import Graph
from dagkit.node import Event
from dagkit.runners import run_sequential, run_threaded
from dagkit.sinks import RerunSink
from dagkit.sources import IterSource
from dagkit.testing import Collect, check_graph_docs, check_picture
from plyfile import PlyData
from simplecv.rerun_log_utils import RerunTyroConfig
from vistadream_fakes import FakePredictor, fake_inpaint, fake_rasterize

from vistadream.apis.single_image import PACKAGE_DIR, PORTS, ROOTS, Config, blueprint
from vistadream.graphs.single_image import single_image_graph
from vistadream.nodes.prepare import DepthNode, ImageSource, OutpaintNode
from vistadream.nodes.splats import ExpandNode, InitNode, PlySink, RenderNode
from vistadream.types import RgbImage, Stage

IMAGE = RgbImage(np.random.default_rng(1).integers(0, 256, (64, 96, 3), dtype=np.uint8))


def graph(coarse: bool, sinks: dict, source: IterSource | None = None) -> Graph:
    source = source or IterSource({"image": RgbImage}, lambda: iter([Event(0, "image", IMAGE)]))
    expand = ExpandNode(fake_inpaint, FakePredictor(), n_frames=3, iters=2, rasterize=fake_rasterize) if coarse else None
    nodes = (
        OutpaintNode(fake_inpaint if coarse else None),
        DepthNode(FakePredictor()),
        InitNode(iters=3, device="cpu", rasterize=fake_rasterize),
        expand,
        RenderNode(rasterize=fake_rasterize),
    )
    return single_image_graph(source, *nodes, sinks)


@pytest.mark.parametrize("coarse", [False, True])
def test_graph_validates_and_runners_agree(coarse: bool) -> None:
    runs: list[list] = []
    for runner in (run_sequential, lambda g: run_threaded(g, mode="lossless")):
        keep = Collect(PORTS)
        built: Graph = graph(coarse, {"keep": keep})
        built.validate()
        assert ("expand" in built.nodes) == coarse
        assert runner(built).status == "complete"
        runs.append(sorted(keep.items, key=lambda item: (item[1], item[0])))
    assert [(p, t) for p, t, _ in runs[0]] == [(p, t) for p, t, _ in runs[1]]
    assert len([p for p, _, _ in runs[0] if p == "view"]) == (75 if coarse else 150)  # 25 per view with 3 views, else 150
    np.testing.assert_equal([v for _, _, v in runs[0]], [v for _, _, v in runs[1]])


def test_ply_sink_writes_a_3dgs_vertex_table(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scene.ply"
    run_sequential(graph(False, {"ply": PlySink(path)}))
    vertices = PlyData.read(str(path))["vertex"]
    names: list[str] = [p.name for p in vertices.properties]
    assert names[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] and len(names) == 62
    assert names[-8:] == ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    assert vertices.count > 0 and np.isfinite(np.asarray(vertices["x"])).all()


def app_sinks(directory: Path) -> dict:
    return {"viewer": RerunSink(rr.RecordingStream("vistadream-test"), PORTS, ROOTS), "ply": PlySink(directory / "scene.ply")}


SAMPLE = ImageSource(PACKAGE_DIR / "data/office/IMG_4029.jpg", 512)
"""The App's source (described, never run here)."""


@pytest.mark.parametrize("coarse", [False, True])
def test_app_graph_picture_matches_its_declaration(coarse: bool, tmp_path: Path) -> None:
    """The graph view the runners log, written to an rrd and read back, draws every declared node and edge once."""
    built: Graph = graph(coarse, app_sinks(tmp_path), SAMPLE)
    built.validate()
    check_picture(built, tmp_path)


def test_app_graph_docs_are_complete_and_graph_md_is_current(tmp_path: Path) -> None:
    """Titles, Type docstrings and the about; GRAPH.md is the coarse shape (every node: no-outpaint only drops ``expand``)."""
    check_graph_docs(graph(True, app_sinks(tmp_path), SAMPLE), snapshot=PACKAGE_DIR / "GRAPH.md")


@pytest.mark.parametrize(
    ("stage", "outputs"),
    [
        ("no-outpaint", ["1 image · input", "3 depth · MoGe depth", "4 init · scene", "5 render · path"]),
        ("coarse", ["2 outpaint · canvas", "3 depth · MoGe depth", "5 expand · scene", "6 render · path"]),
    ],
)
def test_debug_tab_follows_the_normal_view(stage: Stage, outputs: list[str], tmp_path: Path) -> None:
    """The normal view stays the first, active tab; the debug tab's output views carry the guide's step numbers."""
    tabs = blueprint(Config(RerunTyroConfig(headless=True), stage=stage), graph(stage == "coarse", app_sinks(tmp_path), SAMPLE)).root_container
    assert isinstance(tabs, rrb.Tabs) and tabs.active_tab == 0
    normal, debug = tabs.contents
    assert isinstance(normal, rrb.Horizontal) and [v.name for v in normal.contents][0] == "Gaussian splats"
    assert isinstance(debug, rrb.Horizontal) and debug.name == "pipeline (debug)"
    left = list(debug.contents)[0]
    assert isinstance(left, rrb.Vertical)
    row = list(left.contents)[1]
    assert isinstance(row, rrb.Horizontal) and [v.name for v in row.contents] == outputs
