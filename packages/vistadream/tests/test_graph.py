"""The single-image graph with fakes (unit lane, CPU): it validates, both runners deliver the same outputs, the PLY sink writes."""

import numpy as np
import pytest
from dagkit.graph import Graph
from dagkit.node import Event
from dagkit.runners import run_sequential, run_threaded
from dagkit.sources import IterSource
from plyfile import PlyData
from vistadream_fakes import FakePredictor, Keep, fake_inpaint, fake_rasterize

from vistadream.apis.single_image import PORTS
from vistadream.graphs.single_image import single_image_graph
from vistadream.nodes.prepare import DepthNode, OutpaintNode
from vistadream.nodes.splats import ExpandNode, InitNode, PlySink, RenderNode
from vistadream.types import RgbImage

IMAGE = RgbImage(np.random.default_rng(1).integers(0, 256, (64, 96, 3), dtype=np.uint8))


def graph(coarse: bool, sinks: dict) -> Graph:
    source = IterSource({"image": RgbImage}, lambda: iter([Event(0, "image", IMAGE)]))
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
        keep = Keep(PORTS)
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
