"""Profiles (hand-written TOML, coerced, strict) and the generated gradio app."""

import dataclasses
import gc
import tempfile
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest
import rerun as rr
import rerun.blueprint as rrb
from dagkit_fakes import Add, Collect, source
from rerun.chunk import RrdReader
from serde import coerce, serde

from dagkit.apis import synthetic_demo
from dagkit.apis.synthetic_demo import DemoRerunConfig
from dagkit.app import NodeProfile, Profile, detect_hardware, gradio_app, health_views, load_profile, run_app, with_pipeline
from dagkit.graph import Graph
from dagkit.node import Emit, Mode, Tick
from dagkit.runners import run_sequential
from dagkit.sources import SegmentRef
from dagkit.testing import check_form

PROFILE = """
hardware = "x86_64-cuda"

[nodes.fit]
cpus = [6, 7]
params = { iterations = 30, lr = 1 }

[presets.fast.fit]
params = { iterations = 5 }

[presets.fast.track]
placement = "process"
"""


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class FitConfig:
    iterations: int = 100
    lr: float = 0.5
    polish: bool = True


def test_profile_merges_presets_and_configures_nodes(tmp_path: Path) -> None:
    (tmp_path / "x86_64-cuda.toml").write_text(PROFILE)
    profile: Profile = load_profile(tmp_path, "x86_64-cuda")
    assert profile.node("fit") == NodeProfile("thread", [6, 7], {"iterations": 30, "lr": 1})
    assert profile.node("fit", "fast") == NodeProfile("thread", [6, 7], {"iterations": 5, "lr": 1})
    assert profile.node("track", "fast").placement == "process"
    assert profile.configure("fit", FitConfig(), "fast") == FitConfig(iterations=5, lr=1.0, polish=True)
    with pytest.raises(KeyError, match="unknown preset 'slow'"):
        profile.node("fit", "slow")


def test_profile_refuses_unknown_keys_and_the_wrong_class(tmp_path: Path) -> None:
    (tmp_path / "x86_64-cuda.toml").write_text(PROFILE + "\nhostname = 'box'\n")
    with pytest.raises(ValueError, match="x86_64-cuda.toml"):
        load_profile(tmp_path, "x86_64-cuda")
    (tmp_path / "aarch64-cuda.toml").write_text('hardware = "x86_64-cuda"\n')
    with pytest.raises(ValueError, match="declares hardware 'x86_64-cuda'"):
        load_profile(tmp_path, "aarch64-cuda")


def test_detect_hardware_names_a_class_or_explains() -> None:
    try:
        assert detect_hardware() in ("x86_64-cuda", "aarch64-cuda", "robocap-cap")
    except RuntimeError as exc:
        assert "pass the hardware class explicitly" in str(exc)


@dataclass
class Inner:
    headless: bool = False


@dataclass
class DemoConfig:
    rr_config: Inner
    segment: SegmentRef | None = None
    image: Path | None = None
    frames: int = 10
    scale: float = 1.0
    name: str = "demo"
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"
    output_ply: Path | None = None


def test_gradio_app_builds_a_form_from_config_fields() -> None:
    def build(config: DemoConfig, recording: rr.RecordingStream) -> Graph:
        return Graph("empty")

    demo = gradio_app(DemoConfig, build, title="Demo", examples=[DemoConfig(Inner(), SegmentRef("ds", "seg"))])
    labels: list[str] = [getattr(block, "label", None) or "" for block in demo.blocks.values()]
    for expected in ("segment.dataset", "segment.segment", "image", "frames", "scale", "name", "runner", "mode", "output_ply: save it, then download it", "output_ply (download)"):
        assert expected in labels, expected
    assert "rr_config" not in labels


def test_a_preset_can_put_a_process_node_back_on_a_thread(tmp_path: Path) -> None:
    (tmp_path / "x86_64-cuda.toml").write_text('hardware = "x86_64-cuda"\n[nodes.fit]\nplacement = "process"\n[presets.debug.fit]\nplacement = "thread"\ncpus = []\n')
    profile: Profile = load_profile(tmp_path, "x86_64-cuda")
    assert profile.node("fit").placement == "process" and profile.node("fit", "debug") == NodeProfile("thread", [], {})


@dataclass
class FormConfig:
    rr_config: Inner = field(default_factory=Inner)
    segment: SegmentRef | None = None
    frames: int = 10
    label: str = field(default_factory=lambda: "factory value")
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"


def test_generated_run_keeps_defaults_none_segments_and_a_headless_viewer_config() -> None:
    built: list[FormConfig] = []

    def build(config: FormConfig, recording: rr.RecordingStream) -> Graph:
        built.append(config)
        return Graph("empty")

    demo = gradio_app(FormConfig, build, title="Form")
    run = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "run")
    # widget order: segment.dataset, segment.segment, segment.window_ns, frames, label, runner, mode
    list(run("", "", "", 10, None, "threaded", "lossless"))
    list(run("ds", "seg", "", 10, "typed", "threaded", "lossless"))
    assert built[0].segment is None and built[0].label == "factory value" and built[0].rr_config.headless
    assert built[1].segment == SegmentRef("ds", "seg") and built[1].label == "typed"


def test_generated_ui_never_spawns_a_viewer_for_the_synthetic_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    def spawn(self: rr.RecordingStream, *args: object, **kwargs: object) -> None:
        raise AssertionError("the generated UI spawned a native viewer")

    monkeypatch.setattr(rr.RecordingStream, "spawn", spawn)
    demo = gradio_app(synthetic_demo.Config, synthetic_demo.build, title="Synthetic")
    run = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "run")
    list(run(0.2, 30.0, 0.0, "threaded", "lossless"))  # widget order: seconds, hz, smooth_ms, runner, mode


@dataclass(frozen=True, slots=True)
class Nets:
    """Nested Config group (like robocap-live's HandsParams)."""

    models_dir: Path
    """Where the networks live (a directory: a textbox, not an upload)."""
    device: Literal["auto", "cpu", "cuda"] = "auto"
    threads: int = 0
    """Intra-op threads (0 = the runtime's default)."""


@dataclass(frozen=True, slots=True)
class Fill:
    steps: int = 25
    guidance: float = 30.0


@dataclass
class NestedConfig:
    rr_config: Inner
    nets: Nets
    fill: Fill = field(default_factory=lambda: Fill(steps=10))
    hardware: Literal["x86_64-cuda", "aarch64-cuda"] | Literal["auto"] = "auto"
    max_frames: int | None = None
    preset: str | None = None
    output_dir: Path = Path("data/outputs")
    output_ply: Path | None = None
    output_scene: Path | None = None
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"


def run_fn(demo: object) -> Callable[..., Iterator[tuple[object, ...]]]:
    return next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "run")  # type: ignore[attr-defined]


def test_nested_literal_optional_and_output_fields_round_trip_through_the_form() -> None:
    built: list[NestedConfig] = []

    def build(config: NestedConfig, recording: rr.RecordingStream) -> Graph:
        built.append(config)
        assert config.output_ply is None or config.output_scene is None  # the scene run below saves only output_scene
        if config.output_scene is not None:
            config.output_scene.mkdir()
            (config.output_scene / "frame.txt").write_text("1")
        return Graph("empty")

    demo = gradio_app(NestedConfig, build, title="Nested")
    blocks: list[Any] = list(demo.blocks.values())
    labels: list[str] = [getattr(b, "label", None) or "" for b in blocks]
    for expected in ("nets", "nets.models_dir", "nets.device", "nets.threads", "fill", "fill.steps", "fill.guidance", "hardware", "max_frames", "preset"):
        assert expected in labels, expected
    widget: dict[str, Any] = {label: b for label, b in zip(labels, blocks, strict=True)}
    assert type(widget["nets.models_dir"]).__name__ == "Textbox" and widget["nets.threads"].info == "Intra-op threads (0 = the runtime's default)"
    assert [c[1] for c in widget["hardware"].choices] == ["x86_64-cuda", "aarch64-cuda", "auto"]
    assert widget["fill.steps"].value == 10  # the default factory's value, not the class default
    assert type(widget["output_ply: save it, then download it"]).__name__ == "Checkbox" and widget["output_dir"].value == "data/outputs"
    # widget order: nets.models_dir, nets.device, nets.threads, fill.steps, fill.guidance, hardware, max_frames, preset, output_dir,
    # output_ply, output_scene, runner, mode
    assert type(widget["max_frames"]).__name__ == "Textbox"  # gr.Number would show and send 0 for None
    defaults = ("/models", "auto", 0, 10, 30, "auto", "", "", "data/outputs", False, False, "threaded", "lossless")  # 30: the browser's 30.0
    list(run_fn(demo)(*defaults))
    first: NestedConfig = built[0]
    assert first.nets == Nets(Path("/models")) and first.fill == Fill(steps=10) and first.max_frames is None and first.preset is None
    assert isinstance(first.fill.guidance, float)  # a beartyped float field never receives an int
    assert first.output_dir == Path("data/outputs") and first.output_ply is None and first.rr_config.headless
    offered: bool = False
    for last in run_fn(demo)("/models", "cuda", 4, 5, 1.5, "aarch64-cuda", "3", "fast", "elsewhere", False, True, "threaded", "lossless"):
        offered = isinstance(last[-1], str) and Path(last[-1]).is_file()  # gradio copies the download while the run waits at its yield
    second: NestedConfig = built[1]
    assert second.nets == Nets(Path("/models"), "cuda", 4) and second.fill == Fill(5, 1.5) and (second.hardware, second.max_frames, second.preset) == ("aarch64-cuda", 3, "fast")
    assert second.output_scene is not None and second.output_scene.name == "output.scene" and second.output_dir == Path("elsewhere")
    zipped: str | None = last[-1]  # type: ignore[assignment]  # output_scene is a directory: offered as a zip
    assert offered and zipped is not None and zipped.endswith("output.scene.zip") and last[-2] is None
    assert not Path(zipped).parent.exists()  # the run's scratch directory goes with the run


def test_a_form_run_leaves_no_scratch_directory_even_when_the_form_is_wrong(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import gradio as gr

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    demo = gradio_app(NestedConfig, lambda config, recording: Graph("empty"), title="Nested")
    list(run_fn(demo)("/models", "auto", 0, 10, 30, "auto", "", "", "data/outputs", True, False, "threaded", "lossless"))
    with pytest.raises(gr.Error, match="does not make a NestedConfig"):
        list(run_fn(demo)("/models", "auto", 0, 10, 30, "auto", "three", "", "data/outputs", False, False, "threaded", "lossless"))
    assert list(tmp_path.iterdir()) == []


NESTED_FORM: dict[str, str] = {
    "nets.models_dir": "Textbox", "nets.device": "Dropdown", "nets.threads": "Number", "fill.steps": "Number", "fill.guidance": "Number",
    "hardware": "Dropdown", "max_frames": "Textbox", "preset": "Textbox", "output_dir": "Textbox", "output_ply": "Checkbox", "output_scene": "Checkbox",
    "runner": "Dropdown", "mode": "Dropdown",
}  # fmt: skip


def test_check_form_wants_every_config_field_shown_in_order_or_hidden_on_purpose() -> None:
    def build(config: NestedConfig, recording: rr.RecordingStream) -> Graph:
        return Graph("empty")

    demo = check_form(NestedConfig, build, NESTED_FORM, {"rr_config": "the form renders into its own pane"})
    assert type(demo).__name__ == "Blocks"
    with pytest.raises(AssertionError, match="rr_config"):
        check_form(NestedConfig, build, NESTED_FORM, {})
    with pytest.raises(AssertionError, match="max_frames"):
        check_form(NestedConfig, build, {k: v for k, v in NESTED_FORM.items() if k != "max_frames"}, {"rr_config": "", "max_frames": ""})
    with pytest.raises(AssertionError):  # the form's order is part of the table
        check_form(NestedConfig, build, dict(reversed(NESTED_FORM.items())), {"rr_config": ""})


@dataclass
class PassesConfig:
    rr_config: DemoRerunConfig = field(default_factory=lambda: DemoRerunConfig(headless=True))
    values: int = 3
    runner: Literal["sequential", "threaded"] = "sequential"
    mode: Mode = "lossless"


def two_passes(config: PassesConfig, recording: rr.RecordingStream) -> Iterator[Graph]:
    """Pass 1 sums 0..values-1; pass 2 is built from that result (exo-calib's init -> refine shape)."""
    first, keep = Graph("pass1", about="Sum."), Collect(("x",))
    graph_ref = first.source("src", source([(t, "x", t) for t in range(config.values)]))
    first.edge(graph_ref["x"], first.sink("keep", keep)["x"])
    yield first
    second, total = Graph("pass2", about="Hand the sum on."), Collect(("x",))
    second.edge(second.source("src", source([(0, "x", sum(v for _, _, v in keep.items))]))["x"], second.sink("keep", total)["x"])
    yield second
    assert total.items == [("x", 0, 3)]
    recording.log("/passes/done", rr.TextLog("both passes ran"), static=True)


def test_a_two_pass_app_runs_every_graph_from_the_cli_and_the_form(tmp_path: Path) -> None:
    config = PassesConfig()
    config.rr_config.rec_stream.save(tmp_path / "cli.rrd")
    report = run_app(config, two_passes)
    assert report.graph == "pass2" and report.status == "complete"
    config.rr_config.rec_stream.flush()
    reader = RrdReader(tmp_path / "cli.rrd")
    paths: set[str] = {str(c.entity_path) for c in reader.stream(store=reader.recordings()[0]).to_chunks()}
    assert {"/dagkit/graph/pass1", "/dagkit/graph/pass2", "/dagkit/guide/pass1", "/dagkit/guide/pass2", "/passes/done"} <= paths
    streamed: list[tuple[object, ...]] = list(run_fn(gradio_app(PassesConfig, two_passes, title="Passes"))(3, "sequential", "lossless"))
    assert any(out[1] for out in streamed)  # the pane got the stream; the generator's final assert ran (no gr.Error)


def test_health_views_plot_exactly_what_the_runner_logs(tmp_path: Path) -> None:
    graph = Graph("chain", about="Add one.")
    src = graph.source("src", source([(t, "x", t) for t in range(3)]))
    add = graph.node("add", Add(1))
    graph.edge(src["x"], add["x"])
    graph.edge(add["y"], graph.sink("keep", Collect(("y",)))["y"])
    recording = rr.RecordingStream("dagkit-test", recording_id="health", send_properties=False)
    recording.save(tmp_path / "health.rrd")
    run_sequential(graph, health=recording)
    recording.flush()
    reader = RrdReader(tmp_path / "health.rrd")
    logged: set[str] = {str(c.entity_path) for c in reader.stream(store=reader.recordings()[0]).to_chunks()}
    origins: list[str] = [str(view.origin) for view in health_views([graph, graph])]
    assert origins == ["/dagkit/health/add/drops/x", "/dagkit/health/add/queue_depth/x", "/dagkit/health/keep/drops/y", "/dagkit/health/keep/queue_depth/y", "/dagkit/health/add/step_ms"]
    assert set(origins) <= logged


class Weights:
    """A model's GPU memory: a graph frees it when its nodes go."""


class HoldsWeights(Collect):
    def __init__(self, weights: Weights) -> None:
        super().__init__(("x",))
        self.weights: Weights | None = weights


def test_each_graph_and_each_form_run_frees_its_models_before_the_next_build() -> None:
    """A multi-pass App (exo-calib: Stage A models, then TensorRT engines) builds its next graph after the finished one
    let go of its models, by reference counting alone (no cyclic GC, no hand-written release in the App or a node)."""
    alive: list[weakref.ref] = []
    seen: list[list[bool]] = []

    def holding(name: str) -> Graph:
        weights = Weights()
        alive.append(weakref.ref(weights))
        graph = Graph(name, about="Hold weights.")
        graph.edge(graph.source("src", source([(0, "x", 1)]))["x"], graph.sink("keep", HoldsWeights(weights))["x"])
        return graph

    def passes(config: PassesConfig, recording: rr.RecordingStream) -> Iterator[Graph]:
        for name in ("stage-a", "refine"):
            seen.append([ref() is not None for ref in alive])  # earlier models, as this graph is built
            yield holding(name)

    gc.collect()
    gc.disable()
    try:
        run_app(PassesConfig(), passes)
        list(run_fn(gradio_app(PassesConfig, passes, title="Passes"))(3, "sequential", "lossless"))
    finally:
        gc.enable()
    assert seen == [[], [False], [False, False], [False, False, False]]


def test_with_pipeline_keeps_the_normal_view_first_and_active() -> None:
    normal = rrb.Blueprint(rrb.Horizontal(rrb.Spatial3DView(origin="/world", name="world")), rrb.TimePanel(state="collapsed"), collapse_panels=True)
    both: rrb.Blueprint = with_pipeline(normal, [rrb.Spatial2DView(origin="/world/cam", name="2 track · masks")])
    tabs: Any = both.root_container
    assert tabs.kind == rrb.components.ContainerKind.Tabs and tabs.active_tab == 0 and [c.name for c in tabs.contents] == ["main", "pipeline (debug)"]
    assert both.time_panel is normal.time_panel and both.collapse_panels
    views: list[str] = [type(v).__name__ for v in tabs.contents[1].contents[0].contents] + [type(tabs.contents[1].contents[1]).__name__]
    assert views == ["GraphView", "Horizontal", "TextDocumentView"]
    multi: Any = with_pipeline(normal, graphs=["exocalib-init", "exocalib-refine"]).root_container  # a multi-graph app
    two: Any = multi.contents[1]
    assert [(type(v).__name__, str(v.origin)) for v in two.contents[0].contents] == [("GraphView", "/dagkit/graph/exocalib-init"), ("GraphView", "/dagkit/graph/exocalib-refine")]
    assert [v.name for v in two.contents[1].contents] == ["guide exocalib-init: what each step does", "guide exocalib-refine: what each step does"]


@dataclass(frozen=True, slots=True)
class Models:
    models_dir: Path
    """Where the networks live (a directory)."""
    threads: int | None = 1


@dataclass
class RequiredConfig:
    """robocap-live's shape: required fields with no default, a server-side output directory, a registering publisher."""

    rr_config: Inner
    segment: SegmentRef
    output_dir: Path
    models: Models
    output_scene: Path | None = None
    register: Literal["new", "replace", "off"] = "new"
    runner: Literal["sequential", "threaded"] = "sequential"
    mode: Mode = "lossless"


def initial(demo: Any) -> list[object]:
    """The values the form's widgets start with, in the order Run receives them."""
    run = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "run")
    return [demo.blocks[b._id].value if hasattr(b, "_id") else demo.blocks[b].value for b in run.inputs]


def test_required_fields_start_from_the_first_example_and_the_window_round_trips() -> None:
    built: list[RequiredConfig] = []

    def build(config: RequiredConfig, recording: rr.RecordingStream) -> Graph:
        built.append(config)
        return Graph("empty")

    example = RequiredConfig(Inner(), SegmentRef("ds", "seg", window_ns=(5, 9)), Path("/srv/out"), Models(Path("/srv/models")), register="off")
    demo = gradio_app(RequiredConfig, build, title="Required", examples=[example])
    labels: list[str] = [getattr(b, "label", None) or "" for b in demo.blocks.values()]
    assert "segment.window_ns" in labels and "output_dir (download)" not in labels  # a server path: nothing to download
    values: list[object] = initial(demo)
    list(run_fn(demo)(*values))  # Run without touching the form: the first example's inputs, the Config's own register
    assert built[0] == dataclasses.replace(example, rr_config=built[0].rr_config, register="new") and built[0].segment.window_ns == (5, 9)
    window: int = 2  # widget order: segment.dataset, segment.segment, segment.window_ns, output_dir, ...
    list(run_fn(demo)(*values[:window], "", *values[window + 1 :]))
    assert built[1].segment.window_ns is None  # an empty window box: the whole segment


def test_a_form_without_values_for_required_fields_says_which() -> None:
    import gradio as gr

    demo = gradio_app(RequiredConfig, lambda config, recording: Graph("empty"), title="Required")
    with pytest.raises(gr.Error, match="models_dir"):
        list(run_fn(demo)(*initial(demo)))


def test_save_it_never_registers_and_an_empty_optional_number_is_none() -> None:
    built: list[RequiredConfig] = []

    def build(config: RequiredConfig, recording: rr.RecordingStream) -> Graph:
        built.append(config)
        return Graph("empty")

    example = RequiredConfig(Inner(), SegmentRef("ds", "seg"), Path("/srv/out"), Models(Path("/srv/models")))
    demo = gradio_app(RequiredConfig, build, title="Required", examples=[example])
    values: list[object] = initial(demo)
    # widget order: segment.dataset, segment.segment, segment.window_ns, output_dir, models.models_dir, models.threads, output_scene, register, ...
    assert values[5] == "1" and values[6] is False and values[7] == "new"
    list(run_fn(demo)(*values[:5], "", True, *values[7:]))
    assert built[0].models.threads is None  # cleared, not the default 1
    assert built[0].output_scene is not None and built[0].register == "off"  # a temporary output is never registered on the catalog
    list(run_fn(demo)(*values))
    assert (built[1].output_scene, built[1].register) == (None, "new")


def test_an_example_click_clears_what_every_example_leaves_empty() -> None:
    """gr.Examples drops the columns that are None in every example; the form clears them on a click instead of keeping edits."""
    demo = gradio_app(NestedConfig, lambda config, recording: Graph("empty"), title="Nested", examples=[NestedConfig(Inner(), Nets(Path("/m")))])
    clear = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "clear")
    cleared: list[str] = [demo.blocks[b._id if hasattr(b, "_id") else b].label for b in clear.outputs]
    assert {"max_frames", "preset"} <= set(cleared) and "nets.models_dir" not in cleared
    assert all(v is None for v in clear.fn(None).values())


class Stopper(Collect):
    """Presses Stop while its graph runs."""

    def step(self, tick: Tick, out: Emit) -> None:
        super().step(tick, out)
        assert self.ctx is not None and self.ctx.control is not None
        self.ctx.control.stop()


def test_stop_never_builds_another_graph_and_a_repeated_name_is_refused() -> None:
    import gradio as gr

    from dagkit.time import GraphError

    steps: list[str] = []

    def passes(config: PassesConfig, recording: rr.RecordingStream) -> Iterator[Graph]:
        for name in ("first", "second"):
            graph = Graph(name)
            graph.edge(graph.source("src", source([(0, "x", 0)]))["x"], graph.sink("keep", Stopper(("x",)))["x"])
            steps.append(f"built {name}")
            yield graph

    assert run_app(PassesConfig(), passes).status == "stopped" and steps == ["built first"]
    steps.clear()
    list(run_fn(gradio_app(PassesConfig, passes, title="Passes"))(3, "sequential", "lossless"))
    assert steps == ["built first"]

    def twice(config: PassesConfig, recording: rr.RecordingStream) -> Iterator[Graph]:
        yield from two_passes(config, recording)
        yield next(two_passes(config, recording))

    with pytest.raises(GraphError, match="pass1"):
        run_app(PassesConfig(), twice)
    with pytest.raises(gr.Error, match="pass1"):
        list(run_fn(gradio_app(PassesConfig, twice, title="Passes"))(3, "sequential", "lossless"))
