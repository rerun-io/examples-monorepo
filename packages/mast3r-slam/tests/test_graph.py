"""Graph tier, unit lane: the SLAM graph validates in every input shape; process placement equals thread placement;
the profile's presets resolve; the app graph draws its declaration and documents itself; every Config field has a form
widget (no GPU: nothing here runs a resource factory or a model)."""

import dataclasses
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import cv2
import numpy as np
import pytest
import rerun as rr
import tyro
from dagkit import sinks
from dagkit.app import load_profile
from dagkit.doc import describe, picture
from dagkit.form import gradio_app
from dagkit.graph import Block, DropOldest, Graph, Placement, Spawn
from dagkit.node import Emit, Event, PortType, RunContext, Tick
from dagkit.runners import run_threaded
from dagkit.sinks import RerunSink
from dagkit.sources import IterSource, SegmentRef
from dagkit.testing import Recorder, check_form, check_graph_docs, check_picture
from dagkit.time import STATIC
from dagkit.types import Producer
from dagkit.verify import Case, load_card
from hypothesis import given, settings
from hypothesis import strategies as st
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.sources.catalog_video import SegmentFrames, VideoSource
from simplecv.sources.files import ImageDirSource, VideoFileSource, WebcamSource
from simplecv.types.frames import MultiViewFrames
from simplecv.types.rig import RigCameras

from mast3r_slam.apis.slam import PACKAGE_DIR, Config, build, outputs, slam_config, video_source
from mast3r_slam.graphs.slam import frame_hw, slam_graph
from mast3r_slam.nodes.nerfstudio import NerfstudioSink
from mast3r_slam.types import SlamConfig
from mast3r_slam.verify import case_config, examples

PROFILES: Path = Path(__file__).resolve().parents[1] / "profiles"


def video(calibrated: bool) -> IterSource:
    ports: dict[str, type] = {"frames": MultiViewFrames} | ({"rig": RigCameras} if calibrated else {})
    return IterSource(ports, lambda: iter(()))


@pytest.mark.parametrize(("calibrated", "live"), [(False, False), (True, False), (False, True)])
def test_slam_graph_validates(calibrated: bool, live: bool, tmp_path: Path) -> None:
    ports, roots = outputs("rig_00/cam_00")
    sinks = {"viewer": RerunSink(rr.RecordingStream("test", send_properties=False), ports, roots)}
    graph: Graph = slam_graph(video(calibrated), SlamConfig(use_calib=calibrated), sinks, live=live, nerfstudio=NerfstudioSink(tmp_path))
    graph.validate()
    kinds = {(e.src.node, e.dst.node, e.dst.port): e.kind for e in graph.edges}
    assert kinds[("video", "frontend", "frames")] == (DropOldest(1) if live else Block(2))
    assert ("video", "frontend", "rig") in kinds if calibrated else ("video", "frontend", "rig") not in kinds
    assert graph.nodes["backend"].placement == Placement("process") and isinstance(graph.nodes["backend"].obj, Spawn)
    assert graph.nodes["frontend"].placement.where == "main" and set(graph.resources) == {"mast3r", "shared"}
    assert (graph.nodes["nerfstudio"].kind, graph.nodes["nerfstudio"].needs) == ("sink", ("shared",))  # an output that reads `shared`


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


def test_a_file_input_beats_the_webcam_and_the_preset_keeps_its_stride(tmp_path: Path) -> None:
    config = Config(RerunTyroConfig(headless=True), video=tmp_path / "clip.mp4", webcam=0, hardware="x86_64-cuda", preset="fast")
    cfg: SlamConfig = slam_config(config)
    assert cfg.subsample == 5  # the preset's
    source = video_source(config, cfg)
    assert isinstance(source, VideoFileSource) and source.stride == 5 and source.max_frames is None
    assert isinstance(video_source(Config(RerunTyroConfig(headless=True), video=None, webcam=0), cfg), WebcamSource)


def test_the_default_input_is_the_short_case_and_the_examples_are_the_configs_verify_runs(tmp_path: Path) -> None:
    """A run without arguments (and the form's defaults) reads the slam short case's clip; the gradio examples are the
    card's short cases as App configs, and every case's App config resolves the SlamConfig verify always ran. A given
    file still beats the default segment."""
    cases: list[Case] = [c for block in load_card(PACKAGE_DIR / "VERIFY.toml").graph for c in block.cases]
    default = video_source(Config(RerunTyroConfig(headless=True)), SlamConfig())
    assert isinstance(default, VideoSource) and default.cameras == ("rig_00/cam_00",)
    assert isinstance(default.frames, SegmentFrames) and default.frames.segment == next(c for c in cases if c.name == "short").segment_ref()
    assert [e.segment for e in examples()] == [c.segment_ref() for c in cases if c.clip == "short"]
    for case in cases:
        config: Config = dataclasses.replace(case_config(case), hardware="x86_64-cuda")
        assert slam_config(config) == SlamConfig(use_calib=case.params["calib"], img_size=case.params["img_size"], subsample=case.params["subsample"])
        assert config.undistort == case.params.get("undistort", False) and config.camera == "rig_00/cam_00"
    assert isinstance(video_source(Config(RerunTyroConfig(headless=True), video=tmp_path / "clip.mp4"), SlamConfig()), VideoFileSource)


def test_frame_hw_sizes_from_the_rig_or_a_file_resolution_without_a_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = RigCameras(("rig_00/cam_00",), np.eye(3)[None], np.eye(4)[None], np.array([[640, 480]], dtype=np.int64))

    def rig_first() -> Iterator[Event]:
        yield Event(STATIC, "rig", rig)
        raise AssertionError("frame_hw read past the rig")

    cv2.imwrite(str(tmp_path / "000.png"), np.zeros((480, 640, 3), dtype=np.uint8))
    monkeypatch.setattr(ImageDirSource, "events", lambda self: pytest.fail("frame_hw read the first frame"))
    from_file: tuple[int, int] = frame_hw(ImageDirSource(tmp_path), 512)
    assert frame_hw(IterSource({"frames": MultiViewFrames, "rig": RigCameras}, rig_first), 512) == from_file == (384, 512)


@pytest.mark.parametrize(("preset", "profile"), [(None, "x86_64-cuda/base"), ("eval", "x86_64-cuda/eval")])
def test_the_layer_producer_carries_the_package_version_and_the_hardware_profile(preset: str | None, profile: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[Producer] = []
    monkeypatch.setattr(sinks, "log_producer", lambda recording, _layer, producer: logged.append(producer))
    build(Config(RerunTyroConfig(headless=True), hardware="x86_64-cuda", preset=preset, layer_dir=tmp_path), rr.RecordingStream("test", send_properties=False))  # type: ignore[arg-type]
    assert logged == [Producer.of("mast3r-slam", "mast3r-slam", profile)] and "+" in logged[0].version  # version + git sha7


def test_building_the_app_graph_reads_nothing_from_its_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``shared`` buffers are sized when the run starts, so building the graph (the CLI, the form, verify) neither
    queries the catalog nor opens a video."""
    monkeypatch.setattr(VideoSource, "events", lambda self: pytest.fail("build read its input"))
    build(Config(RerunTyroConfig(headless=True), hardware="x86_64-cuda"), rr.RecordingStream("test", send_properties=False)).validate()  # type: ignore[union-attr]


def test_the_viewer_draws_every_output_or_only_the_camera_poses() -> None:
    """``draw="poses"`` (verify's golden): the viewer takes the final keyframe and tracked poses only, not the map and the
    viewer's frames and keyframes (images and point maps), which grow with every frame; by default it takes every output."""
    for draw, taken in (("all", {"keyframes", "tracked", "map", "view", "keyframe", "log"}), ("poses", {"keyframes", "tracked"})):
        graph = build(Config(RerunTyroConfig(headless=True), hardware="x86_64-cuda", draw=draw), rr.RecordingStream("test", send_properties=False))
        assert isinstance(graph, Graph) and {e.dst.port for e in graph.edges if e.dst.node == "viewer"} == taken


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


def test_the_app_graph_draws_its_declaration_and_documents_itself(tmp_path: Path) -> None:
    """``build`` with every output on (catalog input = the default segment, layer, Nerfstudio): valid, the picture read back
    from an rrd equals the declaration, each box says where it runs (the backend is its own process), GRAPH.md is current."""
    config = Config(RerunTyroConfig(headless=True), hardware="x86_64-cuda", output_nerfstudio=tmp_path / "ns", layer_dir=tmp_path / "layers")
    graph: Graph = build(config, rr.RecordingStream("test", send_properties=False))  # type: ignore[assignment]
    graph.validate()
    assert set(graph.nodes) == {"video", "frontend", "backend", "nerfstudio", "viewer", "layer"}
    check_picture(graph, tmp_path)
    boxes: dict[str, str] = {item.id: item.label for item in picture(describe(graph))[0]}
    assert "· main" in boxes["frontend"] and "· process" in boxes["backend"] and "· main" not in boxes["nerfstudio"]
    check_graph_docs(graph, snapshot=PACKAGE_DIR / "GRAPH.md")


FORM: dict[str, str] = {
    "segment.dataset": "Textbox", "segment.segment": "Textbox", "segment.window_ns": "Textbox", "camera": "Textbox", "undistort": "Checkbox",
    "video": "File", "images": "Textbox", "webcam": "Textbox", "max_frames": "Textbox", "hardware": "Dropdown", "preset": "Dropdown",
    "img_size": "Dropdown", "subsample": "Textbox", "draw": "Dropdown", "output_nerfstudio": "Checkbox", "layer_dir": "Textbox", "register": "Dropdown",
    "runner": "Dropdown", "mode": "Dropdown",
}  # fmt: skip
"""The generated form in order: widget label -> component (directories and optional numbers are text boxes; empty = the default)."""


def test_every_config_field_has_its_form_widget_or_is_hidden_on_purpose() -> None:
    demo = check_form(Config, build, FORM, {"rr_config": "the gradio pane is the viewer"})
    widgets: dict[str, object] = {c.label: c for c in demo.blocks.values() if getattr(c, "label", None)}
    assert widgets["hardware"].choices == [(c, c) for c in ("x86_64-cuda", "aarch64-cuda", "robocap-cap", "auto")]  # type: ignore[attr-defined]
    assert {c for c, _ in widgets["preset"].choices} == set(load_profile(PROFILES, "x86_64-cuda").presets)  # type: ignore[attr-defined]


def test_the_catalog_source_muxes_its_zero_copy_packet_views() -> None:
    """This env runs Python 3.11, where a numpy array is no ``typing_extensions.Buffer``: ``SegmentFrames`` hands
    ``wrap_mp4`` the packet views ``read_catalog_videos`` returns, and beartype checks that call in the dev env."""
    samples = [np.frombuffer(b"\x00\x00\x00\x01\x65" + bytes(16), dtype=np.uint8), np.frombuffer(b"\x00\x00\x00\x01\x41" + bytes(8), dtype=np.uint8)]
    assert len(wrap_mp4(samples, [True, False], 30, "h264")) > 0


def test_a_segment_named_on_the_cli_runs_whole_and_none_clears_the_window() -> None:
    """Only the default clip is windowed (the VERIFY short case on MOO01's timeline): a segment named on the CLI runs
    whole unless it names its own window (TUM's timeline is Unix time, where the default window holds no frame), and
    ``--segment.window-ns None`` parses."""

    def parse(*args: str) -> object:
        return tyro.cli(Config, args=["--rr-config.headless", *args]).segment

    tum = ["segment:segment-ref", "--segment.dataset", "dataforge-vslam-lab-rgbdtum", "--segment.segment", "vslam-lab-rgbdtum__rgbd_dataset_freiburg1_desk"]
    assert isinstance(default := parse(), SegmentRef) and default.window_ns == (0, 30_000_000_000)
    assert isinstance(named := parse(*tum), SegmentRef) and named.segment.endswith("freiburg1_desk") and named.window_ns is None
    assert isinstance(clip := parse(*tum, "--segment.window-ns", "5", "9"), SegmentRef) and clip.window_ns == (5, 9)
    assert isinstance(whole := parse("segment:segment-ref", "--segment.window-ns", "None"), SegmentRef) and whole.window_ns is None
