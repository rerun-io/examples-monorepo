"""The refine graph: it validates, and the sequential and threaded (lossless) runners agree (fakes, no GPU, no catalog)."""

from dataclasses import asdict, replace

import numpy as np
from conftest import FakeScene
from dagkit.graph import Graph
from dagkit.node import Event, Fenced
from dagkit.runners import run_sequential, run_threaded
from dagkit.sources import IterSource
from dagkit.testing import collect_edges
from dagkit.time import STATIC

from exo_calib.apis.calibrate import Keep
from exo_calib.graphs.calibrate import init_graph, refine_graph
from exo_calib.nodes.refine import RefineNode
from exo_calib.types import RigLayout


def synthetic_refine_graph(scene: FakeScene, keep: Keep) -> Graph:
    """The real refine graph with the catalog video swapped for synthetic framesets (same ports) and the focal stage on CPU."""
    factories = {name: (lambda value=value: value) for name, value in scene.resources().items()}
    layout = RigLayout(scene.rig.names, (), scene.rig.names)
    graph = refine_graph(scene.segment, layout, (scene.rig, scene.rig), 12, factories, {"keep": keep}, report=True)
    frames = [scene.frames(i) for i in range(0, 48, 4)]
    video = IterSource(graph.nodes["video"].outputs, lambda: iter([Event(STATIC, "rig", scene.rig), *(Event(f.t_ns, "frames", Fenced(f, None)) for f in frames)]))
    graph.nodes["video"] = replace(graph.nodes["video"], obj=video)
    graph.nodes["refine"] = replace(graph.nodes["refine"], obj=RefineNode(device="cpu"))
    return graph


def test_both_graphs_validate() -> None:
    scene = FakeScene()
    synthetic_refine_graph(scene, Keep("refined")).validate()
    init_graph(scene.segment, scene.rig.names, 0, {"g3t": object, "moge": object}, {"keep": Keep("init")}).validate()


def test_sequential_equals_threaded_lossless() -> None:
    scene = FakeScene()
    keeps: list[Keep] = [Keep("refined", "eval", "kp2d"), Keep("refined", "eval", "kp2d")]
    seen = [
        collect_edges(synthetic_refine_graph(scene, keeps[0]), run_sequential),
        collect_edges(synthetic_refine_graph(scene, keeps[1]), lambda graph: run_threaded(graph, mode="lossless")),
    ]
    records = [[(e.t_ns, asdict(e.value)) for e in s["refine"][0]["kp2d"]] for s in seen]  # the refine node's whole input
    np.testing.assert_equal(records[1], records[0])
    assert len(records[0]) == 12
    np.testing.assert_equal(asdict(keeps[1].values["refined"]), asdict(keeps[0].values["refined"]))  # type: ignore[call-overload]
    assert keeps[1].values["eval"] == keeps[0].values["eval"]


def test_refine_graph_runs_on_the_segment_clock() -> None:
    scene = FakeScene()
    layout = RigLayout(scene.rig.names, (), scene.rig.names)
    factories = {name: object for name in ("detector", "pose", "moge", "frames")}
    graph = refine_graph(replace(scene.segment, timeline="capture_time"), layout, (scene.rig, scene.rig), 12, factories, {}, report=True)
    graph.validate()
    assert {node.obj.clock for node in graph.nodes.values() if hasattr(node.obj, "clock")} == {"capture_time"}
