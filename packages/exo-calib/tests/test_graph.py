"""The App's graphs: they validate, draw what they declare and document themselves, a stopped graph ends the App, and the
sequential and threaded (lossless) runners agree (fakes, no GPU, no catalog)."""

from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from conftest import FakeScene
from dagkit.graph import Graph
from dagkit.node import Control, Event, Fenced, RunContext, Tick
from dagkit.runners import run_sequential, run_threaded
from dagkit.sources import IterSource
from dagkit.testing import Recorder, check_graph_docs, check_picture, collect_edges
from dagkit.time import STATIC
from simplecv.rerun_log_utils import RerunTyroConfig

from exo_calib.apis import calibrate
from exo_calib.apis.calibrate import Config, Keep, build
from exo_calib.graphs.calibrate import init_graph, refine_graph
from exo_calib.nodes.refine import RefineNode
from exo_calib.types import RigLayout

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]


def app_graphs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[FakeScene, Iterator[Graph]]:
    """``build`` with the catalog layout and the models faked; nothing runs until the caller fills each graph's Keep."""
    scene = FakeScene()
    monkeypatch.setattr(calibrate, "discover_rig_layout", lambda segment, exo_rigs: RigLayout(scene.rig.names, ("ego/cam_00",), scene.rig.names))
    monkeypatch.setattr(calibrate, "model_factories", lambda *args: {name: (lambda value=value: value) for name, value in scene.resources().items()})
    config = Config(RerunTyroConfig(headless=True), scene.segment, output_dir=tmp_path / "outputs", register="off", hardware="x86_64-cuda")  # no catalog
    return scene, build(config, rr.RecordingStream("exo-calib-test", send_properties=False))


def test_the_app_graphs_validate_draw_what_they_declare_and_document_themselves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scene, steps = app_graphs(monkeypatch, tmp_path)
    init: Graph = next(steps)
    init.nodes["keep"].obj.values["init"] = scene.rig  # type: ignore[attr-defined]  # as the run would: the Stage A rig ...
    pass1: Graph = next(steps)
    pass1.nodes["keep"].obj.values["refined"] = scene.rig  # type: ignore[attr-defined]  # ... and the pass-1 rig, handed on
    pass2: Graph = next(steps)
    graphs: list[Graph] = [init, pass1, pass2]
    assert [g.name for g in graphs] == ["exocalib-init", "exocalib-refine", "exocalib-refine"]
    assert {"viewer", "report", "exocalib_kp2d", "exocalib_refined", "exocalib_align"} <= set(pass2.nodes) and "viewer" not in pass1.nodes
    for k, graph in enumerate(graphs):
        graph.validate()
        (tmp_path / str(k)).mkdir()
        check_picture(graph, tmp_path / str(k))  # the graph view as the runner logs it, read back from an rrd == the declaration
    check_graph_docs(init, pass2, snapshot=PACKAGE_DIR / "GRAPH.md")  # the two graphs the debug tab shows


def test_a_stopped_graph_hands_nothing_on_and_ends_the_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    keep, control = Keep("init"), Control()
    keep.reset(RunContext("exocalib-init", "keep", "lossless", control=control))
    keep.step(Tick(0, "init", FakeScene().rig, {}), Recorder())
    control.stop()
    keep.flush(Recorder())
    assert keep.values == {}
    _, steps = app_graphs(monkeypatch, tmp_path)
    assert next(steps).name == "exocalib-init"
    assert next(steps, None) is None  # its Keep stayed empty: no refine pass is built


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
