"""The hands-layer App (unit lane: no catalog, no GPU): the graph its ``build`` yields, built with fakes, draws exactly its
declaration and documents itself (``GRAPH.md``); the viewer opens on the run health as before, the graph view is a second
tab; a windowed run's hands reach the viewer."""

import dataclasses
import functools
import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, get_args, get_type_hints

import pytest
import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import gradio_app
from dagkit.graph import Graph
from dagkit.node import Event
from dagkit.sources import IterSource, SegmentRef
from dagkit.testing import check_graph_docs, check_picture
from dagkit.time import STATIC
from rerun.chunk import RrdReader
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.types.frames import LumaFrameset
from slam_rs.nodes.vio import VioParams

# pyrefly resolves `test_nodes` to another package's tests on the monorepo search path; pytest imports this one.
from test_nodes import SEGMENT, T0, cap_rig, framesets, no_nets, poses  # pyrefly: ignore[missing-module-attribute]

from robocap_live.apis import hands_layer as app
from robocap_live.apis.hands_layer import Config, blueprint, build, examples, main
from robocap_live.graphs import hands_layer
from robocap_live.nodes.hands import HandsNode, HandsParams
from robocap_live.types import CapRig, CatalogPoses

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]


def app_graphs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slam: Literal["vio", "catalog"] = "catalog") -> list[Graph]:
    """The graphs the App runs, as ``build`` yields them (none runs here); the catalog's layer list is faked (the sources read
    nothing until they run)."""
    monkeypatch.setattr(hands_layer, "segment_layers", lambda segment: ["base", "slam_rs"])
    config = Config(RerunTyroConfig(headless=True), SEGMENT, tmp_path, HandsParams(tmp_path), slam, VioParams(decode_device="cpu"), register="off")
    return list(itertools.islice(build(config, rr.RecordingStream("robocap-live-test")), 2))


def test_the_app_graphs_validate_draw_their_declaration_and_document_themselves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """VIO first, then the hands on the layer it published (a second source, ``slam``, reads it as the STATIC pose table)."""
    graphs: list[Graph] = app_graphs(tmp_path, monkeypatch, "vio")
    assert [g.name for g in graphs] == ["slam-rs-vio", "robocap-hands-layer"] and "slam" in graphs[1].nodes
    for graph in graphs:
        graph.validate()
        check_picture(graph, tmp_path)
    check_graph_docs(*graphs, snapshot=PACKAGE_DIR / "GRAPH.md")
    (catalog,) = app_graphs(tmp_path, monkeypatch, "catalog")
    assert "slam" not in catalog.nodes and catalog.edges[2].src.node == "segment"  # poses from the segment's slam_rs layer


def fake_inputs(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """The App's source replays ``count`` synthetic framesets; its core runs without hand networks (``test_nodes.no_nets``)."""

    def events() -> Iterator[Event]:
        yield Event(STATIC, "cap_rig", cap_rig())
        yield Event(STATIC, "poses", poses(count))
        yield from (Event(f.t_ns, "frames", f) for f in framesets(count))

    monkeypatch.setattr(
        app, "segment_source", lambda segment, **_: IterSource({"frames": LumaFrameset, "cap_rig": CapRig, "poses": CatalogPoses}, events)
    )
    monkeypatch.setattr(app, "HandsNode", functools.partial(HandsNode, core=no_nets))


def entities(rrd: Path) -> set[str]:
    reader = RrdReader(rrd)
    return {str(chunk.entity_path) for chunk in reader.stream(store=reader.recordings()[0]).to_chunks()}


@pytest.mark.parametrize("window", [True, False])
def test_a_windowed_run_copies_its_hands_into_the_viewer_and_a_whole_one_only_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: bool
) -> None:
    fake_inputs(monkeypatch, 5)
    segment: SegmentRef = dataclasses.replace(SEGMENT, window_ns=(T0, T0 + 10**9) if window else None)
    rrd: Path = tmp_path / "app.rrd"
    config = Config(RerunTyroConfig(headless=True, save=rrd), segment, tmp_path, HandsParams(tmp_path), "catalog", register="off")
    report = main(config)
    config.rr_config.rec_stream.flush()
    assert report.status == "complete" and report.nodes["hands"].ticks == 5
    (published,) = (tmp_path / SEGMENT.segment).glob("hands.*.rrd")
    assert published.name.startswith("hands.window." if window else "hands.")
    seen: set[str] = entities(rrd)
    assert {"/dagkit/graph/robocap-hands-layer", "/dagkit/guide/robocap-hands-layer", "/hands_layer/run"} <= seen
    assert ("/producers/hands" in seen) is window  # the layer file's own entities, copied unchanged


def test_the_viewer_opens_on_the_run_health_as_before_and_the_pipeline_tab_comes_second(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first tab plots each series the runner logs (one plot per series, as the viewer laid them out before the graph
    view), except a sink's single end-of-run step time; the "pipeline (debug)" tab follows and never opens first."""
    fake_inputs(monkeypatch, 3)
    rrd: Path = tmp_path / "app.rrd"
    config = Config(RerunTyroConfig(headless=True, save=rrd), SEGMENT, tmp_path, HandsParams(tmp_path), "catalog", register="off")
    main(config)
    config.rr_config.rec_stream.flush()
    logged: set[str] = {
        e for e in entities(rrd) if e.startswith("/dagkit/health/") and e not in ("/dagkit/health/publish/step_ms", "/dagkit/health/viewer/step_ms")
    }
    root = blueprint(app_graphs(tmp_path, monkeypatch)).root_container
    assert isinstance(root, rrb.Tabs) and root.active_tab == 0
    tabs: list[rrb.Container | rrb.View] = list(root.contents)
    assert [tab.name for tab in tabs] == ["main", "pipeline (debug)"] and isinstance(tabs[0], rrb.Grid)
    plots: list[rrb.TimeSeriesView] = [v for v in tabs[0].contents if isinstance(v, rrb.TimeSeriesView)]
    assert len(plots) == len(list(tabs[0].contents)) == 7 and {str(v.origin) for v in plots} == logged


FORM: dict[str, str] = {  # Config leaf -> the widget the generated form gives it
    "segment.dataset": "Textbox", "segment.segment": "Textbox", "segment.window_ns": "Textbox", "output_dir": "Textbox", "hands.models_dir": "Textbox",
    "hands.device": "Dropdown", "hands.ort_dylib": "File", "hands.ort_threads": "Number", "hands.overlays": "Dropdown",
    "slam": "Dropdown", "vio.profile": "Dropdown", "vio.backend": "Dropdown", "vio.decode_device": "Dropdown",
    "register": "Dropdown", "runner": "Dropdown", "mode": "Dropdown",
}  # fmt: skip
HIDDEN: dict[str, str] = {
    "rr_config": "the form streams into its own Rerun pane",
    "segment.catalog_url": "kept from the default SegmentRef",
    "segment.timeline": "kept from the default SegmentRef",
}


def leaves(cls: type, prefix: str = "") -> dict[str, object]:
    """Every field of a Config, nested dataclasses opened up (unless hidden whole), with its type hint."""
    out: dict[str, object] = {}
    for name, hint in get_type_hints(cls).items():
        path: str = prefix + name
        nested: bool = dataclasses.is_dataclass(hint) and path not in HIDDEN
        out |= leaves(hint, f"{path}.") if nested and isinstance(hint, type) else {path: hint}
    return out


def test_every_config_field_is_a_form_widget_or_deliberately_hidden() -> None:
    gr = pytest.importorskip("gradio", reason="the generated form needs gradio (robocap-live envs)")
    fields: dict[str, object] = leaves(Config)
    assert sorted(fields) == sorted([*FORM, *HIDDEN]), "a Config field is neither in the form nor deliberately hidden"
    demo = gradio_app(Config, build, title="RoboCap hands layer", examples=examples())
    widgets: dict[str, Any] = {c.label: c for c in demo.blocks.values() if getattr(c, "label", None) in FORM}
    for label, kind in FORM.items():
        widget = widgets[label]
        assert type(widget).__name__ == kind, f"{label}: {type(widget).__name__}, expected {kind}"
        if kind == "Dropdown":  # every Literal value, "auto" included
            assert [value for _, value in widget.choices] == list(get_args(fields[label])), label
    assert any(isinstance(c, gr.Accordion) and c.label == "hands" for c in demo.blocks.values()), "HandsParams is not grouped"
    # the fields the Config leaves open start from the example: Run on the untouched form is the short case, unregistered (windowed)
    assert (widgets["segment.segment"].value, widgets["segment.window_ns"].value) == (
        "robocap__fe6fede545c972fa__s00000014",
        "1655058237343500 1655063237343500",
    )


def test_the_examples_are_the_verify_short_cases_never_registered() -> None:
    """One click runs a known-good input: 5 s of the README sample (Cap B session 14) on CPU ONNX Runtime, as VERIFY.toml says;
    the first runs VIO on the CPU lane before the hands (``robocap``), the second reads the registered poses (``hands``)."""
    window: tuple[int, int] = (1_655_058_237_343_500, 1_655_063_237_343_500)
    for example, slam in zip(examples(), ("vio", "catalog"), strict=True):
        assert example.segment == SegmentRef("robocap", "robocap__fe6fede545c972fa__s00000014", window_ns=window) and example.slam == slam
        assert (example.register, example.hands.device, example.hands.ort_threads, example.hands.overlays) == ("off", "cpu", 4, "debug")
        assert (example.hands.models_dir / "detnet_full.onnx").parent == PACKAGE_DIR / "models"
    assert examples()[0].vio == VioParams("fast", "cpu", "cpu")
