"""The App's graphs: they validate, draw what they declare and document themselves, a stopped graph ends the App, and the
sequential and threaded (lossless) runners agree (fakes, no GPU, no catalog)."""

from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

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
from exo_calib.graphs.calibrate import init_graph, refine_graph, rig_seed
from exo_calib.nodes.refine import RefineNode
from exo_calib.types import RigLayout

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]


def app_graphs(monkeypatch: pytest.MonkeyPatch, output_dir: Path | None) -> tuple[FakeScene, Iterator[Graph]]:
    """``build`` with the catalog layout and the models faked; nothing runs until the caller fills each graph's Latest sink."""
    scene = FakeScene()
    monkeypatch.setattr(calibrate, "discover_rig_layout", lambda segment, exo_rigs, lens: RigLayout(scene.rig.names, ("ego/cam_00",), scene.rig.names))
    monkeypatch.setattr(calibrate, "model_factories", lambda *args: {name: (lambda value=value: value) for name, value in scene.resources().items()})
    config = Config(RerunTyroConfig(headless=True), scene.segment, output_dir=output_dir, register="off", hardware="x86_64-cuda")  # no catalog
    return scene, build(config, rr.RecordingStream("exo-calib-test", send_properties=False))


def hand_on(graph: Graph, value: object) -> Graph:
    """Fill the graph's ``Latest`` sink as its run would: ``value`` on every port it keeps (rigs, or the report)."""
    keep = graph.nodes["keep"].obj
    assert isinstance(keep, Latest)
    keep.values = dict.fromkeys(keep.inputs, value)
    return graph


def test_the_app_graphs_validate_draw_what_they_declare_and_document_themselves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scene, steps = app_graphs(monkeypatch, tmp_path / "outputs")
    graphs: list[Graph] = [hand_on(next(steps), scene.rig), hand_on(next(steps), scene.rig), next(steps)]  # the Stage A and pass-1 rigs, handed on
    pass1, pass2 = graphs[1:]
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
    graphs: list[Graph] = [hand_on(next(steps), value) for value in (scene.rig, scene.rig, "the report")]
    assert next(steps, None) is None
    assert [g.name for g in graphs] == ["exocalib-init", "exocalib-refine-pass1", "exocalib-refine"]
    assert [sorted(n for n, spec in g.nodes.items() if isinstance(spec.obj, RerunSink | LayerSink)) for g in graphs] == [["viewer"], [], ["viewer"]]
    assert list(tmp_path.iterdir()) == []


def test_one_app_run_loads_each_refine_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """MoGe-2 and the TensorRT detector and pose engines load once per App run and serve both refine passes (Stage A
    shares MoGe); G3T stays Stage A's, built by its graph and freed with it."""
    scene, steps = app_graphs(monkeypatch, None)
    models = ("g3t", "moge", "detector", "pose", "frames", "smplx")
    monkeypatch.setattr(calibrate, "model_factories", lambda *args: dict.fromkeys(models, object) | {"frames": FakeScene})  # a new object per call
    graphs: list[Graph] = [hand_on(next(steps), scene.rig), hand_on(next(steps), scene.rig), next(steps)]
    for name in ("moge", "detector", "pose"):
        loaded = [g.resources[name]() for g in graphs if name in g.resources]
        assert len(loaded) == (3 if name == "moge" else 2) and all(model is loaded[0] for model in loaded), name
    assert graphs[0].resources["g3t"]() is not graphs[0].resources["g3t"]()


def resumed(monkeypatch: pytest.MonkeyPatch, resume_from: Literal["init", "kp2d"], built: list[str]) -> tuple[FakeScene, Iterator[Graph]]:
    """``build`` resuming from ``resume_from``, layout and models faked (``built`` records every model a factory builds), the
    Producer check passed (its own test is below) and a catalog read by the App itself a failure: only the seed reads it."""
    scene = FakeScene()
    monkeypatch.setattr(calibrate, "discover_rig_layout", lambda segment, exo_rigs, lens: scene.layout)
    monkeypatch.setattr(calibrate, "model_factories", lambda *args: {name: (lambda name=name: built.append(name)) for name in ("g3t", "moge", "detector", "pose", "frames", "smplx")})
    monkeypatch.setattr(calibrate, "check_producer", lambda segment, layer, producer: None)
    monkeypatch.setattr(SegmentRef, "entry", lambda self: pytest.fail("the App read the catalog itself"))
    config = Config(RerunTyroConfig(headless=True), scene.segment, register="off", resume_from=resume_from, hardware="x86_64-cuda")
    return scene, build(config, rr.RecordingStream("exo-calib-test", send_properties=False))


def test_a_kp2d_resume_builds_no_detector_or_pose_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--resume-from kp2d`` is one graph whose seed reads the stored Stage A rig, Stage B records and GT rig when it runs,
    and it declares only the models its nodes use: MoGe and the frames for the rescale, SMPL-X for the body, no detector, no
    pose engine."""
    built: list[str] = []
    _, steps = resumed(monkeypatch, "kp2d", built)
    (graph,) = steps
    graph.validate()
    assert set(graph.nodes["seed"].outputs) == {"init", "kp2d", "rig"} and "video" not in graph.nodes
    for factory in graph.resources.values():  # as the runner does when the graph starts
        factory()
    assert sorted(built) == ["frames", "moge", "smplx"]


def test_an_init_resume_tracks_through_the_stored_rig_and_hands_it_on_to_pass_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--resume-from init``: pass 1's seed reads the stored Stage A rig as both rigs; pass 2 refines against that rig
    again and tracks through pass 1's, both handed on in memory by pass 1's ``Latest`` sink."""
    scene, steps = resumed(monkeypatch, "init", [])
    pass1: Graph = next(steps)
    assert set(pass1.nodes["seed"].outputs) == {"init", "tracking_rig"} and "video" in pass1.nodes
    stored, refined = scene.rig, scene.rig.select(scene.rig.names[::-1])  # what the stored layer and pass 1 would give
    keep = pass1.nodes["keep"].obj
    assert isinstance(keep, Latest)
    keep.values = {"init": stored, "refined": refined}
    pass2: Graph = next(steps)
    seed = pass2.nodes["seed"].obj
    assert isinstance(seed, IterSource) and [(e.t_ns, e.port, e.value) for e in seed.events()] == [(STATIC, "init", stored), (STATIC, "tracking_rig", refined)]


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
    graph = refine_graph(scene.segment, scene.layout, rig_seed(scene.segment, init=scene.rig, tracking_rig=scene.rig), 12, factories, {"keep": keep}, report=True)
    frames = [scene.frames(i) for i in range(0, 48, 4)]
    video = IterSource(graph.nodes["video"].outputs, lambda: iter([Event(STATIC, "rig", scene.rig), *(Event(f.t_ns, "frames", Fenced(f, None)) for f in frames)]))
    graph.nodes["video"] = replace(graph.nodes["video"], obj=video)
    graph.resources["frames"] = factories["frames"]  # the scene's frames: the real reader belonged to the replaced source
    graph.nodes["refine"] = replace(graph.nodes["refine"], obj=RefineNode(scene.layout, device="cpu"))
    return graph


def test_both_graphs_validate() -> None:
    scene = FakeScene()
    synthetic_refine_graph(scene, Latest({p: PORT_TYPES[p] for p in ("refined",)})).validate()
    init_graph(scene.segment, scene.rig.names, 0, {"g3t": object, "moge": object, "frames": lambda: scene}, {"keep": Latest({p: PORT_TYPES[p] for p in ("init",)})}).validate()


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


def test_one_app_run_fetches_the_exo_videos_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """One ``SegmentFrames`` per App run: Stage A's video source, both refine passes' sources and the metric rescale's
    ``frames`` all read the packets the first graph fetched (each fetch of the 8 videos took ~1.8 s)."""
    scene, steps = app_graphs(monkeypatch, None)
    monkeypatch.setattr(calibrate, "model_factories", lambda *args: dict.fromkeys(("g3t", "moge", "detector", "pose", "smplx"), object) | {"frames": FakeScene})
    graphs: list[Graph] = [hand_on(next(steps), scene.rig), hand_on(next(steps), scene.rig), next(steps)]
    readers = [g.nodes["video"].obj.frames for g in graphs] + [g.resources["frames"]() for g in graphs[1:]]  # type: ignore[attr-defined]
    assert all(reader is readers[0] for reader in readers)


def test_refine_graph_runs_on_the_segment_clock() -> None:
    scene = FakeScene()
    factories = {name: object for name in ("detector", "pose", "moge", "smplx")} | {"frames": FakeScene}
    segment = replace(scene.segment, timeline="capture_time")
    graph = refine_graph(segment, scene.layout, rig_seed(segment, init=scene.rig, tracking_rig=scene.rig), 12, factories, {}, report=True)
    graph.validate()
    assert {node.obj.clock for node in graph.nodes.values() if hasattr(node.obj, "clock")} == {"capture_time"}
