"""App layer: hardware profiles, the CLI (``run_app``) and the generated gradio app (``gradio_app``); one ``build`` for both.

``build(config, recording)`` returns a Graph, or a generator of Graphs for a multi-pass app (``graphs``). Both entry points
run every graph with ``health=recording``, and the runner then logs the graph view itself (``dagkit.doc.log_graph``): the
picture at ``/dagkit/graph/<graph>``, the guide at ``/dagkit/guide/<graph>``, the ``GraphDoc`` JSON and run-health colours
on ``wall_time``. No package wires any of it; a package only wraps its normal blueprint with ``with_pipeline(bp, outputs)``
to get the secondary "pipeline (debug)" tab, while its normal view stays the first, active tab (in the gradio pane too).
"""

import contextlib
import dataclasses
import functools
import itertools
import platform
import shutil
import signal
import tempfile
import threading
import time
import types
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, Union, cast, get_args, get_origin, get_type_hints, runtime_checkable

import rerun as rr
import rerun.blueprint as rrb
from serde import SerdeError, coerce, from_dict, serde, to_dict
from serde.toml import from_toml

from dagkit.doc import GRAPH_ROOT, GUIDE_ROOT, field_docs
from dagkit.graph import Graph
from dagkit.node import Control, Mode
from dagkit.runners import RunReport, run_sequential, run_threaded
from dagkit.sources import SegmentRef
from dagkit.time import GraphError

HardwareClass: TypeAlias = Literal["x86_64-cuda", "aarch64-cuda", "robocap-cap"]
"""Profiles are hardware classes (the repo is public): never hostnames, never GPU models."""
RUNNERS: dict[str, Callable[..., RunReport]] = {"sequential": run_sequential, "threaded": run_threaded}


@runtime_checkable
class AppConfig(Protocol):
    rr_config: Any
    runner: Literal["sequential", "threaded"]
    mode: Mode


C = TypeVar("C", bound=AppConfig)
D = TypeVar("D")
G = TypeVar("G")


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class NodeProfile:
    placement: Literal["main", "thread", "process"] | None = None  # None inherits (base default: thread)
    cpus: list[int] | None = None  # None inherits; [] clears
    params: dict[str, Any] = field(default_factory=dict)  # overrides of the node's config dataclass fields


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class Profile:
    hardware: HardwareClass
    nodes: dict[str, NodeProfile] = field(default_factory=dict)
    presets: dict[str, dict[str, NodeProfile]] = field(default_factory=dict)  # preset -> node -> overlay

    def node(self, name: str, preset: str | None = None) -> NodeProfile:
        if preset is not None and preset not in self.presets:
            raise KeyError(f"unknown preset {preset!r}; this profile has {sorted(self.presets)}")
        base: NodeProfile = self.nodes.get(name, NodeProfile())
        over: NodeProfile = self.presets[preset].get(name, NodeProfile()) if preset is not None else NodeProfile()
        cpus: list[int] | None = over.cpus if over.cpus is not None else base.cpus
        return NodeProfile(over.placement or base.placement or "thread", cpus, base.params | over.params)

    def configure(self, name: str, default: D, preset: str | None = None) -> D:
        return from_dict(type(default), to_dict(default) | self.node(name, preset).params)


def detect_hardware() -> HardwareClass:
    machine: str = platform.machine()
    compatible: Path = Path("/proc/device-tree/compatible")
    if machine == "aarch64" and compatible.exists() and b"rockchip,rk3588" in compatible.read_bytes():
        return "robocap-cap"
    if not Path("/proc/driver/nvidia").exists():
        raise RuntimeError(f"no hardware class for {machine} without an NVIDIA driver; pass the hardware class explicitly")
    return "aarch64-cuda" if machine in ("aarch64", "arm64") else "x86_64-cuda"


def load_profile(directory: Path, hardware: HardwareClass | Literal["auto"] = "auto") -> Profile:
    resolved: HardwareClass = detect_hardware() if hardware == "auto" else hardware
    path: Path = directory / f"{resolved}.toml"
    try:
        profile: Profile = from_toml(Profile, path.read_text())
    except (SerdeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if profile.hardware != resolved:
        raise ValueError(f"{path} declares hardware {profile.hardware!r}, expected {resolved!r}")
    return profile


def graphs(built: Graph | Iterator[Graph]) -> Iterator[Graph]:
    """``build`` returns one Graph, or an iterator of Graphs (a generator) that runs them in order: a driver that needs one
    graph's results to build the next (exo-calib: init, then refine passes) reads them from a sink between its yields."""
    return iter((built,)) if isinstance(built, Graph) else built


def run_app(config: C, build: Callable[[C, rr.RecordingStream], Graph | Iterator[Graph]]) -> RunReport:
    """The CLI: every graph ``build`` gives, on ``config.runner`` / ``config.mode``, Ctrl-C = stop; returns the last report.

    The runner logs each graph's picture, guide and GraphDoc into ``config.rr_config.rec_stream`` (``dagkit.doc``), and
    recolours it with the run health; ``with_pipeline`` adds the tab that shows them.
    """
    recording: rr.RecordingStream = config.rr_config.rec_stream
    control = Control()
    main: bool = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, lambda *_: control.stop()) if main else None
    reports: list[RunReport] = []
    try:
        for graph in graphs(build(config, recording)):
            if control.stopped:
                break
            reports.append(report := RUNNERS[config.runner](graph, mode=config.mode, control=control, health=recording))
            print(f"{report.graph}: {report.status}, {report.wall_s:.2f} s wall, {report.stream_s:.2f} s stream")
    finally:
        if main:
            signal.signal(signal.SIGINT, previous)
    if not reports:
        raise GraphError("build gave no graph to run")
    return reports[-1]


def with_pipeline(blueprint: rrb.Blueprint, outputs: Sequence[rrb.View] = ()) -> rrb.Blueprint:
    """``blueprint`` (the app's normal view, first and active) plus a "pipeline (debug)" tab: the graph picture over
    ``outputs`` (2-4 NEW views of the app's outputs: a view object sits in one place only) beside the guide, whose links
    select and outline what those views show. Panels and flags of ``blueprint`` are kept."""
    picture = rrb.GraphView(origin=GRAPH_ROOT, name="pipeline graph")
    left: rrb.Container | rrb.View = rrb.Vertical(picture, rrb.Horizontal(*outputs), row_shares=[3, 2]) if outputs else picture
    tab = rrb.Horizontal(left, rrb.TextDocumentView(origin=GUIDE_ROOT, name="guide: what each step does"), column_shares=[3, 2], name="pipeline (debug)")
    normal: rrb.Container = blueprint.root_container
    normal.name = normal.name or "main"
    panels: list[Any] = [getattr(blueprint, p) for p in ("top_panel", "blueprint_panel", "selection_panel", "time_panel") if hasattr(blueprint, p)]
    return rrb.Blueprint(
        rrb.Tabs(normal, tab, active_tab=0), *panels, auto_layout=blueprint.auto_layout, auto_views=blueprint.auto_views, collapse_panels=blueprint.collapse_panels
    )


def _base(hint: object) -> Any:
    return next(a for a in get_args(hint) if a is not type(None)) if get_origin(hint) in (Union, types.UnionType) else hint


def _choices(hint: object) -> list[Any] | None:
    """Every value of a Literal or a union of Literals (``HardwareClass | Literal["auto"]``); None for anything else."""
    members: list[object] = [a for a in (get_args(hint) if get_origin(hint) in (Union, types.UnionType) else (hint,)) if a is not type(None)]
    return [v for m in members for v in get_args(m)] if all(get_origin(m) is Literal for m in members) else None


def _nested(hint: object) -> bool:
    """A Config dataclass the form opens up (SegmentRef and the viewer config have their own handling)."""
    base: object = _base(hint)
    return dataclasses.is_dataclass(base) and base is not SegmentRef and not _viewer(base)


def _viewer(cls: object) -> bool:
    return dataclasses.is_dataclass(cls) and "headless" in {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]


def _default(f: dataclasses.Field, hint: object) -> object:
    """A field's default (MISSING if required); a viewer config's factory is never called (it may spawn a viewer)."""
    if f.default is not dataclasses.MISSING or f.default_factory is dataclasses.MISSING or _viewer(_base(hint)):
        return f.default
    return f.default_factory()


Leaf: TypeAlias = tuple[tuple[str, ...], object, object, str]  # (path, type hint, default, attribute docstring)


def _leaves(cls: type, template: object, prefix: tuple[str, ...] = ()) -> list[Leaf]:
    """Every form field of a Config, nested dataclasses opened up; defaults come from ``template`` when there is one."""
    hints: dict[str, Any] = get_type_hints(cls)
    docs: dict[str, str] = field_docs(cls)
    out: list[Leaf] = []
    for f in dataclasses.fields(cls):
        default: object = _default(f, hints[f.name]) if template is None else getattr(template, f.name)
        if _nested(hints[f.name]):
            out += _leaves(_base(hints[f.name]), None if default in (None, dataclasses.MISSING) else default, (*prefix, f.name))
        else:
            out.append(((*prefix, f.name), hints[f.name], default, docs.get(f.name, "")))
    return out


def _widgets(leaf: Leaf) -> list[Any]:
    """Path -> File (a directory, by name or docstring, -> Textbox), output* -> save checkbox (optional) or Textbox, bool ->
    Checkbox, int/float -> Number (optional: Textbox), Literals -> Dropdown, str -> Textbox, SegmentRef -> dataset +
    segment; others: none."""
    import gradio as gr

    path, hint, default, doc = leaf
    label, base = ".".join(path), _base(hint)
    value: Any = None if default is dataclasses.MISSING else default
    optional: bool = type(None) in get_args(hint)
    if base is SegmentRef:
        return [gr.Textbox(getattr(value, part, ""), label=f"{label}.{part}") for part in ("dataset", "segment")]
    if path[-1].startswith("output"):
        return [gr.Checkbox(False, label=f"{label}: save it, then download it", info=doc)] if optional else [gr.Textbox(str(value or ""), label=label, info=doc)]
    if (choices := _choices(hint)) is not None:
        return [gr.Dropdown(choices, value=value, label=label, info=doc)]
    if base is Path:
        directory: bool = path[-1].endswith("dir") or "directory" in doc
        return [gr.Textbox(str(value or ""), label=label, info=doc) if directory else gr.File(value=str(value) if value else None, label=label, type="filepath")]
    if base in (int, float) and optional:  # gr.Number shows and sends 0 for None: a text box keeps "empty" empty
        return [gr.Textbox("" if value is None else str(value), label=label, info=doc, placeholder="empty: the default")]
    makers: dict[object, Callable[..., Any]] = {bool: gr.Checkbox, int: lambda **kw: gr.Number(precision=0, **kw), float: gr.Number, str: gr.Textbox}
    return [makers[base](value=value, label=label, info=doc)] if base in makers else []


def _value(leaf: Leaf, got: list[Any], scratch: Path) -> object:
    """The Config value of one leaf's widgets; MISSING keeps the default (an empty widget, an unchecked save box)."""
    path, hint, default, _ = leaf
    base: object = _base(hint)
    if base is SegmentRef:  # empty boxes = no catalog input; edits keep the default's url, timeline, window
        template: SegmentRef = default if isinstance(default, SegmentRef) else SegmentRef("", "")
        return dataclasses.replace(template, dataset=str(got[0]), segment=str(got[1])) if any(got) else None
    if path[-1].startswith("output") and type(None) in get_args(hint):
        return scratch / path[-1].replace("output_", "output.", 1) if got[0] else dataclasses.MISSING
    if got[0] is None or (got[0] == "" and (type(None) in get_args(hint) or base in (Path, int, float))):
        return dataclasses.MISSING
    return base(got[0]) if base in (Path, int, float) else got[0]  # the browser sends 20 for 20.0 and text from a text box


def _assemble(cls: type, given: dict[tuple[str, ...], object], template: object, prefix: tuple[str, ...] = ()) -> Any:
    """The Config from the set leaves: unset leaves keep their defaults, nested dataclasses are built when needed."""
    hints: dict[str, Any] = get_type_hints(cls)
    kwargs: dict[str, object] = {}
    for f in dataclasses.fields(cls):
        path: tuple[str, ...] = (*prefix, f.name)
        if path in given:
            kwargs[f.name] = given[path]
        elif _viewer(_base(hints[f.name])):
            kwargs[f.name] = _base(hints[f.name])(headless=True)  # the UI renders into its own pane: never spawn a viewer
        elif _nested(hints[f.name]) and (any(p[: len(path)] == path for p in given) or _default(f, hints[f.name]) is dataclasses.MISSING):
            sub: object = getattr(template, f.name) if template is not None else _default(f, hints[f.name])
            kwargs[f.name] = _assemble(_base(hints[f.name]), given, None if sub in (None, dataclasses.MISSING) else sub, path)
    return cls(**kwargs) if template is None else dataclasses.replace(cast(Any, template), **kwargs)


def _download(value: object, scratch: Path) -> str | None:
    """A file output as it is; a directory the form chose is zipped (gr.File cannot return a directory)."""
    if not isinstance(value, Path) or not value.exists():
        return None
    return str(value) if value.is_file() else shutil.make_archive(str(value), "zip", value) if value.is_relative_to(scratch) else None


def gradio_app(config_type: type[G], build: Callable[[G, rr.RecordingStream], Graph | Iterator[Graph]], *, title: str, examples: Sequence[G] = ()) -> Any:
    """The form for ``config_type`` and a Rerun pane; Run builds the same graphs as the CLI (``build``) and streams them."""
    import gradio as gr
    from gradio_rerun import Rerun

    leaves: list[Leaf] = _leaves(config_type, None)
    outputs: list[Leaf] = [leaf for leaf in leaves if leaf[0][-1].startswith("output")]
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        control_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                form: list[tuple[Leaf, list[Any]]] = []
                for prefix, group in itertools.groupby(leaves, key=lambda leaf: leaf[0][:-1]):
                    with gr.Accordion(".".join(prefix), open=True) if prefix else contextlib.nullcontext():
                        form += [(leaf, w) for leaf in group if (w := _widgets(leaf))]
                run_button, stop_button = gr.Button("Run", variant="primary"), gr.Button("Stop")
                files = [gr.File(label=".".join(leaf[0]) + " (download)") for leaf in outputs]
            viewer = Rerun(streaming=True, height=720, scale=3)
        flat: list[Any] = [w for _, widgets in form for w in widgets]

        def run(*values: object) -> Iterator[tuple[object, ...]]:
            it: Iterator[object] = iter(values)
            scratch = Path(tempfile.mkdtemp(prefix="dagkit-app-"))
            given: dict[tuple[str, ...], object] = {}
            for leaf, widgets in form:
                if (value := _value(leaf, [next(it) for _ in widgets], scratch)) is not dataclasses.MISSING:
                    given[leaf[0]] = value
            config: G = _assemble(config_type, given, None)
            recording = rr.RecordingStream(title.replace(" ", "_"), recording_id=uuid.uuid4().hex)
            stream, control, errors = recording.binary_stream(), Control(), []

            def body() -> None:
                try:
                    for graph in graphs(build(config, recording)):
                        if not control.stopped:
                            RUNNERS[getattr(config, "runner", "threaded")](graph, mode=getattr(config, "mode", "lossless"), control=control, health=recording)
                except BaseException as exc:  # delivered to the UI thread below
                    errors.append(exc)

            thread = threading.Thread(target=body)
            thread.start()
            while thread.is_alive():
                time.sleep(0.2)
                yield (control, stream.read(), *(None for _ in files))
            recording.flush()
            if errors:
                raise gr.Error(repr(errors[0]))
            yield (control, stream.read(), *(_download(functools.reduce(getattr, leaf[0], config), scratch) for leaf in outputs))

        run_button.click(run, inputs=flat, outputs=[control_state, viewer, *files])
        stop_button.click(lambda control: control and control.stop(), inputs=[control_state], outputs=[])
        if examples:
            gr.Examples([[v for leaf, ws in form for v in _example(leaf, e)[: len(ws)]] for e in examples], inputs=flat)
    return demo


def _example(leaf: Leaf, example: object) -> list[object]:
    """Widget values of one leaf for an example Config."""
    value: object = functools.reduce(lambda obj, name: getattr(obj, name, None), leaf[0], example)
    if isinstance(value, SegmentRef) or _base(leaf[1]) is SegmentRef:
        return [getattr(value, "dataset", ""), getattr(value, "segment", "")]
    if leaf[0][-1].startswith("output") and type(None) in get_args(leaf[1]):
        return [value is not None]
    return [str(value) if isinstance(value, Path) else value]
