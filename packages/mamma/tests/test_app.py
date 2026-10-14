"""The capture App built with fake models (no GPU, no catalog): the graph validates, documents itself (GRAPH.md) and its
picture reads back as declared; the blueprint keeps the normal view first; the generated form covers the Config."""

import dataclasses
from pathlib import Path

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import gradio_app
from dagkit.doc import describe
from dagkit.graph import Graph
from dagkit.sinks import LayerSink, RerunSink
from dagkit.testing import check_graph_docs, check_picture
from mamma_fakes import FakeDetector, FakeEncoder, FakeFitter, FakeLandmarker, FakeMasks, FakeScene
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.sources.catalog_video import CatalogVideoSource

from mamma.apis.blueprint import default_blueprint
from mamma.apis.capture import LAYER, PACKAGE_DIR, RING, Config, app_graph, build
from mamma.nodes.fit import FitNode
from mamma.nodes.landmarks import LandmarksNode
from mamma.nodes.track import TrackNode, TrackParams
from mamma.ops.window_fit import FitterConfig
from mamma.verify import example_cases

HIDDEN: dict[str, str] = {
    "rr_config": "the form renders into its own Rerun pane (headless)",
    "cameras": "a tuple: the form has no widget for it; None = the mamma-eval ring, as in every VERIFY case",
    "resize_hw": "a tuple: the form has no widget for it; the default is the VERIFY engine size",
}
"""Config fields the generated form leaves out on purpose."""


def app(config: Config) -> Graph:
    """The graph ``build`` returns for ``config``, with fake models and a catalog source that never connects."""
    video = CatalogVideoSource(config.segment, RING, decode="nvdec_rgb", resize_hw=config.resize_hw, keep_native=True)
    nodes = (
        TrackNode(TrackParams(expected_subjects=1), FakeDetector(), FakeEncoder(), FakeMasks()),  # type: ignore[arg-type]
        LandmarksNode(FakeLandmarker(FakeScene())),
        FitNode(FitterConfig(emit_stride=1), np.zeros((2, 3), dtype=np.int32), lambda K, T: FakeFitter()),  # type: ignore[arg-type,return-value]
    )
    return app_graph(config, rr.RecordingStream("mamma-test", send_properties=False), video, nodes, "x86_64-cuda/quality")


def test_the_app_graph_validates_reads_back_as_declared_and_documents_itself(tmp_path: Path) -> None:
    """With ``output_dir`` the App adds the ``mamma_fit`` layer sink: GRAPH.md describes that, the fullest graph (``off``: a
    sink that would register checks the catalog when it is built)."""
    graph = app(Config(RerunTyroConfig(headless=True), output_dir=tmp_path, register="off"))
    graph.validate()
    assert {name: type(spec.obj) for name, spec in graph.nodes.items() if name in ("viewer", LAYER)} == {"viewer": RerunSink, LAYER: LayerSink}
    check_picture(graph, tmp_path)
    check_graph_docs(graph, snapshot=PACKAGE_DIR / "GRAPH.md")


def test_the_viewer_only_app_graph_validates_and_reads_back_as_declared(tmp_path: Path) -> None:
    graph = app(Config(RerunTyroConfig(headless=True)))
    graph.validate()
    assert LAYER not in graph.nodes
    check_picture(graph, tmp_path)


def test_the_blueprint_keeps_the_normal_view_first_and_names_the_debug_outputs_by_box_number() -> None:
    blueprint: rrb.Blueprint = default_blueprint(RING)
    tabs = blueprint.root_container
    assert isinstance(tabs, rrb.Tabs) and tabs.active_tab == 0
    normal, debug = tabs.contents
    assert isinstance(normal, rrb.Vertical) and debug.name == "pipeline (debug)"
    numbers: dict[str, int] = {n.name: n.number for n in describe(app(Config(RerunTyroConfig(headless=True)))).nodes}

    def views(container: object) -> list[rrb.View]:
        return [container] if isinstance(container, rrb.View) else [v for c in getattr(container, "contents", []) for v in views(c)]

    outputs: list[str] = [str(v.name) for v in views(debug) if not isinstance(v, rrb.GraphView | rrb.TextDocumentView)]
    assert [name.split(" · ")[0] for name in outputs] == [f"{numbers[node]} {node}" for node in ("track", "landmarks", "fit")]
    assert {str(v.name) for v in views(normal)}.isdisjoint(outputs)


def form() -> gr.Blocks:
    return gradio_app(Config, build, title="mamma capture", examples=example_cases())


def test_every_config_field_has_a_form_widget_or_is_hidden_on_purpose() -> None:
    """Table test over the Config: one widget per field (the segment three: dataset, id and window), none for the HIDDEN ones."""
    labels: list[str] = [str(b.label) for b in form().blocks.values() if isinstance(b, gr.components.FormComponent) and b.label]
    widgets: dict[str, list[str]] = {f.name: [label for label in labels if label.split(":")[0].split(".")[0] == f.name] for f in dataclasses.fields(Config)}
    assert {name for name, found in widgets.items() if not found} == set(HIDDEN)
    assert widgets.pop("segment") == ["segment.dataset", "segment.segment", "segment.window_ns"]
    assert {name: len(found) for name, found in widgets.items() if name not in HIDDEN} == {name: 1 for name in widgets if name not in HIDDEN}, widgets


def test_one_click_on_an_example_fills_the_form_with_a_verify_case() -> None:
    """The form edits a segment's dataset, id and window and has no widget for the HIDDEN fields, so an example states a
    VERIFY case exactly (the default cameras and engine size)."""
    examples: list[Config] = example_cases()
    (table,) = [b for b in form().blocks.values() if isinstance(b, gr.Dataset)]
    labels: list[str] = [str(c.label) for c in table._components]
    default = Config(RerunTyroConfig(headless=True))
    contact: str = "mamma-eval__dance__250225_WestCoastSwing_Basic_Whip_03675_03676_1"
    assert [(e.segment.segment, e.segment.window_ns, e.subjects) for e in examples] == [
        (default.segment.segment, (0, 10_000_000_000), 1), (default.segment.segment, None, 1), (contact, None, 2)
    ]  # short, full, contact
    for row, example in zip(table.raw_samples, examples, strict=True):
        assert dict(zip(labels, row, strict=True)) == {
            "segment.dataset": example.segment.dataset, "segment.segment": example.segment.segment, "preset": "quality",
            "segment.window_ns": " ".join(map(str, example.segment.window_ns or ())),
            "subjects": example.subjects, "output_dir: save it, then download it": False, "register": "off",
            "hardware": default.hardware, "runner": default.runner, "mode": default.mode,
        }  # fmt: skip
        assert (example.video, example.cameras, example.resize_hw) == (None, RING, default.resize_hw)
