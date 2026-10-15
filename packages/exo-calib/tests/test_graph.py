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
from dagkit.sinks import Latest, LayerSink, RerunSink
from dagkit.sources import IterSource, SegmentRef
from dagkit.testing import Recorder, check_graph_docs, check_picture, collect_edges
from dagkit.time import STATIC
from dagkit.types import Producer
from simplecv.rerun_log_utils import RerunTyroConfig

from exo_calib.apis import calibrate
from exo_calib.apis.calibrate import PORT_TYPES, Config, build, check_producer
from exo_calib.graphs.calibrate import init_graph, refine_graph
from exo_calib.nodes.refine import RefineNode
from exo_calib.types import RigLayout

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]


def app_graphs(monkeypatch: pytest.MonkeyPatch, output_dir: Path | None) -> tuple[FakeScene, Iterator[Graph]]:
    """``build`` with the catalog layout and the models faked; nothing runs until the caller fills each graph's Latest sink."""
    scene = FakeScene()
    monkeypatch.setattr(calibrate, "discover_rig_layout", lambda segment, exo_rigs: RigLayout(scene.rig.names, ("ego/cam_00",), scene.rig.names))
    monkeypatch.setattr(calibrate, "model_factories", lambda *args: {name: (lambda value=value: value) for name, value in scene.resources().items()})
    config = Config(RerunTyroConfig(headless=True), scene.segment, output_dir=output_dir, register="off", hardware="x86_64-cuda")  # no catalog
    return scene, build(config, rr.RecordingStream("exo-calib-test", send_properties=False))


def test_the_app_graphs_validate_draw_what_they_declare_and_document_themselves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scene, steps = app_graphs(monkeypatch, tmp_path / "outputs")
    init: Graph = next(steps)
    init.nodes["keep"].obj.values["init"] = scene.rig  # type: ignore[attr-defined]  # as the run would: the Stage A rig ...
    pass1: Graph = next(steps)
    pass1.nodes["keep"].obj.values["refined"] = scene.rig  # type: ignore[attr-defined]  # ... and the pass-1 rig, handed on
    pass2: Graph = next(steps)
    graphs: list[Graph] = [init, pass1, pass2]
    assert [g.name for g in graphs] == ["exocalib-init", "exocalib-refine-pass1", "exocalib-refine"]
    assert {"viewer", "report", "exocalib_kp2d", "exocalib_refined", "exocalib_align"} <= set(pass2.nodes) and "viewer" not in pass1.nodes
    for k, graph in enumerate(graphs):
        graph.validate()
        (tmp_path / str(k)).mkdir()
        check_picture(graph, tmp_path / str(k))  # the graph view as the runner logs it, read back from an rrd == the declaration
    check_graph_docs(*graphs, snapshot=PACKAGE_DIR / "GRAPH.md")  # the three graphs the debug tab shows


def test_a_stopped_graph_hands_nothing_on_and_ends_the_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    keep, control = Latest({p: PORT_TYPES[p] for p in ("init",)}), Control()
    keep.reset(RunContext("exocalib-init", "keep", "lossless", control=control))
    keep.step(Tick(0, "init", FakeScene().rig, {}), Recorder())
    control.stop()
    keep.flush(Recorder())
    assert keep.values == {}
    _, steps = app_graphs(monkeypatch, tmp_path / "outputs")
    assert next(steps).name == "exocalib-init"
    assert next(steps, None) is None  # its Latest sink stayed empty: no refine pass is built


def test_without_an_output_dir_the_app_streams_to_the_viewer_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``output_dir=None`` (verify, the form's unchecked save box): the same graphs, a viewer sink and no layer sink, and
    no file at all, not even ``eval.json``."""
    monkeypatch.chdir(tmp_path)
    scene, steps = app_graphs(monkeypatch, None)
    graphs: list[Graph] = []
    for port, value in (("init", scene.rig), ("refined", scene.rig), ("eval", "the report")):
        graphs.append(next(steps))
        graphs[-1].nodes["keep"].obj.values[port] = value  # type: ignore[attr-defined]  # as the run would
    assert next(steps, None) is None
    assert [g.name for g in graphs] == ["exocalib-init", "exocalib-refine-pass1", "exocalib-refine"]
    assert [sorted(n for n, spec in g.nodes.items() if isinstance(spec.obj, RerunSink | LayerSink)) for g in graphs] == [["viewer"], [], ["viewer"]]
    assert list(tmp_path.iterdir()) == []


def test_one_app_run_loads_each_refine_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """MoGe-2 and the TensorRT detector and pose engines load once per App run and serve both refine passes (Stage A
    shares MoGe); G3T stays Stage A's, built by its graph and freed with it."""
    scene, steps = app_graphs(monkeypatch, None)
    models = ("g3t", "moge", "detector", "pose", "frames")
    monkeypatch.setattr(calibrate, "model_factories", lambda *args: dict.fromkeys(models, object))  # a new object per call
    graphs: list[Graph] = []
    for port in ("init", "refined"):
        graphs.append(next(steps))
        graphs[-1].nodes["keep"].obj.values[port] = scene.rig  # type: ignore[attr-defined]  # as the run would
    graphs.append(next(steps))
    for name in ("moge", "detector", "pose"):
        loaded = [g.resources[name]() for g in graphs if name in g.resources]
        assert len(loaded) == (3 if name == "moge" else 2) and all(model is loaded[0] for model in loaded), name
    assert graphs[0].resources["g3t"]() is not graphs[0].resources["g3t"]()


def test_resume_refuses_a_layer_another_graph_profile_or_parameter_set_wrote(monkeypatch: pytest.MonkeyPatch) -> None:
    written = Producer("exo-calib", "0.1.0+abc1234", "exocalib-init", "x86_64-cuda", params_sha="0123456789ab")
    monkeypatch.setattr(SegmentRef, "entry", lambda self: None)  # no catalog: the stored Producer is faked below
    monkeypatch.setattr(calibrate, "read_producer", lambda dataset, segment, layer: written)
    check_producer(SegmentRef("d", "s"), "exocalib_init", replace(written, version="0.2.0+def5678"))  # a newer checkout resumes
    for change in ({"graph": "exocalib-refine"}, {"profile": "aarch64-cuda"}, {"params_sha": "ba9876543210"}):
        with pytest.raises(ValueError, match="cannot resume from exocalib_init"):
            check_producer(SegmentRef("d", "s"), "exocalib_init", replace(written, **change))
    monkeypatch.setattr(calibrate, "read_producer", lambda dataset, segment, layer: None)  # the layer has no Producer
    with pytest.raises(ValueError, match="cannot resume from exocalib_init"):
        check_producer(SegmentRef("d", "s"), "exocalib_init", written)


def synthetic_refine_graph(scene: FakeScene, keep: Latest) -> Graph:
    """The real refine graph with the catalog video swapped for synthetic framesets (same ports) and the focal stage on CPU."""
    factories = {name: (lambda value=value: value) for name, value in scene.resources().items()}
    layout = RigLayout(scene.rig.names, (), scene.rig.names)
    graph = refine_graph(scene.segment, layout, (scene.rig, scene.rig), 12, factories, {"keep": keep}, report=True)
    frames = [scene.frames(i) for i in range(0, 48, 4)]
    video = IterSource(graph.nodes["video"].outputs, lambda: iter([Event(STATIC, "rig", scene.rig), *(Event(f.t_ns, "frames", Fenced(f, None)) for f in frames)]))
    graph.nodes["video"] = replace(graph.nodes["video"], obj=video)
    graph.resources["frames"] = factories["frames"]  # the scene's frames: the real reader belonged to the replaced source
    graph.nodes["refine"] = replace(graph.nodes["refine"], obj=RefineNode(device="cpu"))
    return graph


def test_both_graphs_validate() -> None:
    scene = FakeScene()
    synthetic_refine_graph(scene, Latest({p: PORT_TYPES[p] for p in ("refined",)})).validate()
    init_graph(scene.segment, scene.rig.names, 0, {"g3t": object, "moge": object}, {"keep": Latest({p: PORT_TYPES[p] for p in ("init",)})}).validate()


def test_sequential_equals_threaded_lossless() -> None:
    scene = FakeScene()
    keeps: list[Latest] = [Latest({p: PORT_TYPES[p] for p in ("refined", "eval", "kp2d")}), Latest({p: PORT_TYPES[p] for p in ("refined", "eval", "kp2d")})]
    seen = [
        collect_edges(synthetic_refine_graph(scene, keeps[0]), run_sequential),
        collect_edges(synthetic_refine_graph(scene, keeps[1]), lambda graph: run_threaded(graph, mode="lossless")),
    ]
    records = [[(e.t_ns, asdict(e.value)) for e in s["refine"][0]["kp2d"]] for s in seen]  # the refine node's whole input
    np.testing.assert_equal(records[1], records[0])
    assert len(records[0]) == 12
    np.testing.assert_equal(asdict(keeps[1].values["refined"]), asdict(keeps[0].values["refined"]))  # type: ignore[call-overload]
    assert keeps[1].values["eval"] == keeps[0].values["eval"]


def test_a_refine_pass_reads_its_frames_through_the_video_sources_own_reader() -> None:
    """One ``SegmentFrames`` per pass: the metric rescale decodes from the packets the source fetched (no second fetch)."""
    scene = FakeScene()
    layout = RigLayout(scene.rig.names, (), scene.rig.names)
    factories = {name: object for name in ("detector", "pose", "moge", "frames")}
    graph = refine_graph(scene.segment, layout, (scene.rig, scene.rig), 12, factories, {}, report=False)
    assert graph.resources["frames"]() is graph.nodes["video"].obj.frames  # type: ignore[attr-defined]


def test_refine_graph_runs_on_the_segment_clock() -> None:
    scene = FakeScene()
    layout = RigLayout(scene.rig.names, (), scene.rig.names)
    factories = {name: object for name in ("detector", "pose", "moge", "frames")}
    graph = refine_graph(replace(scene.segment, timeline="capture_time"), layout, (scene.rig, scene.rig), 12, factories, {}, report=True)
    graph.validate()
    assert {node.obj.clock for node in graph.nodes.values() if hasattr(node.obj, "clock")} == {"capture_time"}
