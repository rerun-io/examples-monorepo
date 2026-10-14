"""Graph tier, unit lane: the SLAM graph validates in every input shape; process placement equals thread placement;
the profile's presets resolve (no GPU: nothing here runs a resource factory or a model)."""

import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
import rerun as rr
from dagkit.app import gradio_app, load_profile
from dagkit.graph import Block, DropOldest, Graph, Placement, Spawn
from dagkit.node import Emit, Event, PortType, RunContext, Tick
from dagkit.runners import run_threaded
from dagkit.sinks import RerunSink
from dagkit.sources import IterSource
from dagkit.testing import Recorder
from hypothesis import given, settings
from hypothesis import strategies as st
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.sources.files import VideoFileSource
from simplecv.types.frames import MultiViewFrames
from simplecv.types.rig import RigCameras

from mast3r_slam.apis.slam import Config, build, outputs, slam_config, video_source
from mast3r_slam.graphs.slam import slam_graph
from mast3r_slam.nodes.nerfstudio import NerfstudioSink
from mast3r_slam.sources.webcam import Webcam
from mast3r_slam.types import SlamConfig

PROFILES: Path = Path(__file__).resolve().parents[1] / "profiles"


def video(calibrated: bool) -> IterSource:
    ports: dict[str, type] = {"frames": MultiViewFrames} | ({"rig": RigCameras} if calibrated else {})
    return IterSource(ports, lambda: iter(()))


@pytest.mark.parametrize(("calibrated", "live"), [(False, False), (True, False), (False, True)])
def test_slam_graph_validates(calibrated: bool, live: bool, tmp_path: Path) -> None:
    ports, roots = outputs("rig_00/cam_00")
    sinks = {"viewer": RerunSink(rr.RecordingStream("test", send_properties=False), ports, roots)}
    graph: Graph = slam_graph(video(calibrated), SlamConfig(use_calib=calibrated), (384, 512), sinks, live=live, nerfstudio=NerfstudioSink(tmp_path))
    graph.validate()
    kinds = {(e.src.node, e.dst.node, e.dst.port): e.kind for e in graph.edges}
    assert kinds[("video", "frontend", "frames")] == (DropOldest(1) if live else Block(2))
    assert ("video", "frontend", "rig") in kinds if calibrated else ("video", "frontend", "rig") not in kinds
    assert graph.nodes["backend"].placement == Placement("process") and isinstance(graph.nodes["backend"].obj, Spawn)
    assert graph.nodes["frontend"].placement.where == "main" and set(graph.resources) == {"mast3r", "shared"}


class Affine:
    """A pure node: y = a * x + b per tick (picklable config, host values)."""

    inputs: Mapping[str, PortType] = {"x": float}
    outputs: Mapping[str, PortType] = {"y": float}

    def __init__(self, config: tuple[float, float]) -> None:
        self.a, self.b = config

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        out.put("y", tick.t_ns, self.a * float(tick.value) + self.b)  # type: ignore[arg-type]

    def flush(self, out: Emit) -> None:
        pass


class Collect(Recorder):
    inputs: Mapping[str, PortType] = {"y": float}
    outputs: Mapping[str, PortType] = {}

    def reset(self, ctx: RunContext) -> None:
        self.items.clear()

    def step(self, tick: Tick, out: Emit) -> None:
        self.put(tick.port, tick.t_ns, tick.value)

    def flush(self, out: Emit) -> None:
        pass


def affine_graph(values: list[float], process: bool) -> tuple[Graph, Collect]:
    graph, sink = Graph("affine"), Collect()

    def events() -> Iterator[Event]:
        return iter([Event(i, "x", v) for i, v in enumerate(values)])

    source = graph.source("source", IterSource({"x": float}, events))
    node = graph.node("affine", Spawn(Affine, (2.0, -1.0)) if process else Affine((2.0, -1.0)), placement=Placement("process" if process else "thread"))
    graph.edge(source["x"], node["x"])
    graph.edge(node["y"], graph.sink("collect", sink)["y"])
    return graph, sink


@given(values=st.lists(st.floats(-1e3, 1e3), min_size=1, max_size=6))
@settings(max_examples=3, deadline=None)
def test_a_pure_node_in_a_process_gives_the_same_outputs_as_in_a_thread(values: list[float]) -> None:
    results: list[list[tuple[str, int, object]]] = []
    for process in (False, True):  # the threaded runner gives the thread placement its own worker thread
        graph, sink = affine_graph(values, process)
        run_threaded(graph, mode="lossless")
        results.append(sink.items)
    assert results[0] == results[1] == [("y", i, 2.0 * v - 1.0) for i, v in enumerate(values)]


@pytest.mark.parametrize("preset", [None, "base", "eval", "fast", "calib"])
def test_profile_presets_resolve(preset: str | None) -> None:
    cfg: SlamConfig = load_profile(PROFILES, "x86_64-cuda").configure("slam", SlamConfig(), preset)
    assert cfg.subsample == {None: 1, "base": 1, "eval": 2, "fast": 5, "calib": 2}[preset]
    assert cfg.img_size == (224 if preset == "fast" else 512) and cfg.use_calib == (preset == "calib")


def test_a_file_input_beats_webcam_zero_and_zero_means_unset(tmp_path: Path) -> None:
    """The generated gradio form sends 0 for an empty number field (webcam, max_frames, subsample)."""
    config = Config(RerunTyroConfig(headless=True), video=tmp_path / "clip.mp4", webcam=0, max_frames=0, subsample=0, hardware="x86_64-cuda", preset="fast")
    cfg: SlamConfig = slam_config(config)
    assert cfg.subsample == 5  # the preset's
    source = video_source(config, cfg)
    assert isinstance(source, VideoFileSource) and source.stride == 5 and source.max_frames is None
    assert isinstance(video_source(Config(RerunTyroConfig(headless=True), video=None, webcam=0), cfg), Webcam)


def test_the_generated_gradio_form_builds_without_the_example_data() -> None:
    assert Config.__dataclass_fields__["video"].default is None  # a default file that is not downloaded crashes the form
    gradio_app(Config, build, title="MASt3R-SLAM")


def test_the_app_runs_the_threaded_runner_only() -> None:
    """The lossless frontend waits for the backend at its next step and at flush; the sequential runner steps the backend
    only between frontend calls, so the wait in flush would never end."""
    with pytest.raises(ValueError, match="threaded runner only"):
        build(Config(RerunTyroConfig(headless=True), runner="sequential"), rr.RecordingStream("test", send_properties=False))


def test_the_graph_and_cpu_ops_import_without_the_kernel_extensions() -> None:
    """Unit lane (no GPU): the Mojo / CUDA kernels load on first use, never at import (the Mojo module opens a GPU context)."""
    loaded = "import sys, mast3r_slam.apis.slam; print([m for m in ('mast3r_slam_mojo_backends', 'mast3r_slam._backends') if m in sys.modules])"
    assert subprocess.run([sys.executable, "-c", loaded], capture_output=True, text=True, check=True).stdout.splitlines()[-1] == "[]"
