"""Hardware profiles, the CLI entry point and the graph view's blueprint helpers; ``dagkit.form`` generates the gradio app from the same ``build(config, recording)``."""

import platform
import signal
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

import rerun as rr
import rerun.blueprint as rrb
from serde import SerdeError, coerce, from_dict, serde, to_dict
from serde.toml import from_toml

from dagkit.doc import GRAPH_ROOT, GUIDE_ROOT, HEALTH_ROOT
from dagkit.graph import AsOf, Graph
from dagkit.node import Control, Mode
from dagkit.runners import RunReport, run_sequential, run_threaded
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


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class NodeProfile:
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
        return NodeProfile(base.params | over.params)

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


def _run_graphs(config: AppConfig, build: Callable[..., Graph | Iterator[Graph]], recording: rr.RecordingStream, control: Control) -> RunReport | None:
    """The CLI's and the form's one loop: Stop is checked before the next graph is built, and a stopped build is closed."""
    result: Graph | Iterator[Graph] = build(config, recording)
    built: Iterator[Graph] = iter((result,)) if isinstance(result, Graph) else result
    names: set[str] = set()
    report: RunReport | None = None
    try:
        while not control.stopped and (graph := next(built, None)) is not None:
            if graph.name in names:  # its picture, guide and health colours would land on the earlier graph's
                raise GraphError(f"graph name {graph.name!r} repeats in one app")
            names.add(graph.name)
            report = RUNNERS[config.runner](graph, mode=config.mode, control=control, health=recording)
            print(f"{report.graph}: {report.status}, {report.wall_s:.2f} s wall, {report.stream_s:.2f} s stream")
            del graph  # its nodes free their models before the next graph is built
    finally:
        getattr(built, "close", lambda: None)()  # a stopped multi-pass build never runs past its yield
    return report


def run_app(config: C, build: Callable[[C, rr.RecordingStream], Graph | Iterator[Graph]]) -> RunReport:
    """Run every built graph with health logging and Ctrl-C to stop; return the last report (an empty build raises)."""
    control = Control()
    main: bool = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, lambda *_: control.stop()) if main else None
    try:
        report: RunReport | None = _run_graphs(config, build, config.rr_config.rec_stream, control)
    finally:
        if main:
            signal.signal(signal.SIGINT, previous)
    if report is None:
        raise GraphError("build gave no graph to run")
    return report


def with_pipeline(blueprint: rrb.Blueprint, outputs: Sequence[rrb.View] = (), graphs: Sequence[str] = ()) -> rrb.Blueprint:
    """Append a debug tab, keeping the active normal view, panels and flags. ``outputs`` must be NEW view objects;
    ``graphs`` gives each graph its own picture and guide tabs (overlapping pictures cannot share a GraphView)."""
    names: list[str] = [f"/{g}" for g in graphs] or [""]
    pictures: list[rrb.View] = [rrb.GraphView(origin=GRAPH_ROOT.rstrip("/") + n, name=f"graph{n.replace('/', ' ')}") for n in names]
    guides: list[rrb.View] = [rrb.TextDocumentView(origin=GUIDE_ROOT.rstrip("/") + n, name=f"guide{n.replace('/', ' ')}: what each step does") for n in names]
    picture: rrb.Container | rrb.View = rrb.Tabs(*pictures) if len(pictures) > 1 else pictures[0]
    left: rrb.Container | rrb.View = rrb.Vertical(picture, rrb.Horizontal(*outputs), row_shares=[3, 2]) if outputs else picture
    tab = rrb.Horizontal(left, rrb.Tabs(*guides) if len(guides) > 1 else guides[0], column_shares=[3, 2], name="pipeline (debug)")
    normal: rrb.Container = blueprint.root_container
    normal.name = normal.name or "main"
    panels: list[Any] = [getattr(blueprint, p) for p in ("top_panel", "blueprint_panel", "selection_panel", "time_panel") if hasattr(blueprint, p)]
    return rrb.Blueprint(
        rrb.Tabs(normal, tab, active_tab=0), *panels, auto_layout=blueprint.auto_layout, auto_views=blueprint.auto_views, collapse_panels=blueprint.collapse_panels
    )


def health_views(graphs: Sequence[Graph]) -> list[rrb.View]:
    """One plot per run-health series the runner logs for ``graphs``: drops and queue depth per trigger edge, step ms per node."""
    series: list[str] = [f"{e.dst.node}/{what}/{e.dst.port}" for g in graphs for e in g.edges if not isinstance(e.kind, AsOf) for what in ("drops", "queue_depth")]
    series += [f"{name}/step_ms" for g in graphs for name, spec in g.nodes.items() if spec.kind == "node"]
    return [rrb.TimeSeriesView(origin=HEALTH_ROOT + s, name=s) for s in dict.fromkeys(series)]
