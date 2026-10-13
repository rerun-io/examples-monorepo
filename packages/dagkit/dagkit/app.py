"""App helpers: hardware-class profiles, one entry for CLI and UI, the generated gradio app (SPEC §2.9)."""

import dataclasses
import platform
import signal
import threading
import time
import types
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, Union, get_args, get_origin, runtime_checkable

import rerun as rr
from serde import SerdeError, coerce, from_dict, serde, to_dict
from serde.toml import from_toml

from dagkit.graph import Graph
from dagkit.node import Control, Mode
from dagkit.runners import RunReport, run_sequential, run_threaded
from dagkit.sources import SegmentRef

HardwareClass: TypeAlias = Literal["x86_64-cuda", "aarch64-cuda", "robocap-cap"]
"""Profiles are hardware classes (the repo is public): never hostnames, never GPU models."""


@runtime_checkable
class AppConfig(Protocol):
    """What `run_app` reads from a package Config."""

    rr_config: Any
    runner: Literal["sequential", "threaded"]
    mode: Mode


C = TypeVar("C", bound=AppConfig)
D = TypeVar("D")
G = TypeVar("G")


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class NodeProfile:
    """Per-node placement and params of one hardware class (or one preset overlay)."""

    placement: Literal["main", "thread", "process"] = "thread"
    """Where the threaded runner runs the node."""
    cpus: list[int] | None = None
    """CPU affinity."""
    params: dict[str, Any] = field(default_factory=dict)
    """Overrides of the node's config dataclass fields."""


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class Profile:
    """`packages/<pkg>/profiles/<hardware>.toml`: node settings plus named operating points."""

    hardware: HardwareClass
    """The class this file is for."""
    nodes: dict[str, NodeProfile] = field(default_factory=dict)
    """Base settings per node."""
    presets: dict[str, dict[str, NodeProfile]] = field(default_factory=dict)
    """Operating points (e.g. quality / fast): preset -> node -> overlay."""

    def node(self, name: str, preset: str | None = None) -> NodeProfile:
        """Base settings merged with the preset overlay (overlay params win; a non-default overlay placement/cpus wins)."""
        base: NodeProfile = self.nodes.get(name, NodeProfile())
        if preset is not None and preset not in self.presets:
            raise KeyError(f"unknown preset {preset!r}; this profile has {sorted(self.presets)}")
        overlay: NodeProfile | None = self.presets[preset].get(name) if preset is not None else None
        if overlay is None:
            return base
        placement = overlay.placement if overlay.placement != "thread" else base.placement
        return NodeProfile(placement, overlay.cpus if overlay.cpus is not None else base.cpus, base.params | overlay.params)

    def configure(self, name: str, default: D, preset: str | None = None) -> D:
        """The node's config (`@serde(type_check=coerce)`, since params come from TOML) with profile/preset params over `default`."""
        return from_dict(type(default), to_dict(default) | self.node(name, preset).params)


def detect_hardware() -> HardwareClass:
    """platform.machine() + the NVIDIA driver; the RoboCap cap is aarch64 with a rockchip,rk3588 device tree."""
    machine: str = platform.machine()
    compatible: Path = Path("/proc/device-tree/compatible")
    if machine == "aarch64" and compatible.exists() and b"rockchip,rk3588" in compatible.read_bytes():
        return "robocap-cap"
    if not Path("/proc/driver/nvidia").exists():
        raise RuntimeError(f"no hardware class for {machine} without an NVIDIA driver; pass the hardware class explicitly")
    return "aarch64-cuda" if machine in ("aarch64", "arm64") else "x86_64-cuda"


def load_profile(directory: Path, hardware: HardwareClass | Literal["auto"] = "auto") -> Profile:
    """Read `<directory>/<hardware>.toml` (hand-written, so coerced; unknown keys refused)."""
    resolved: HardwareClass = detect_hardware() if hardware == "auto" else hardware
    path: Path = directory / f"{resolved}.toml"
    try:
        profile: Profile = from_toml(Profile, path.read_text())
    except (SerdeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if profile.hardware != resolved:
        raise ValueError(f"{path} declares hardware {profile.hardware!r}, expected {resolved!r}")
    return profile


def _run(graph: Graph, runner: str, mode: Mode, control: Control, health: rr.RecordingStream | None) -> RunReport:
    if runner == "sequential":
        return run_sequential(graph, mode=mode, control=control, health=health)
    return run_threaded(graph, mode=mode, control=control, health=health)


def run_app(config: C, build: Callable[[C, rr.RecordingStream], Graph]) -> RunReport:
    """The CLI entry: recording = `config.rr_config.rec_stream`, Control.stop on SIGINT, runner and mode from config."""
    recording: rr.RecordingStream = config.rr_config.rec_stream
    control = Control()
    main: bool = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, lambda *_: control.stop()) if main else None
    try:
        report: RunReport = _run(build(config, recording), config.runner, config.mode, control, recording)
    finally:
        if main:
            signal.signal(signal.SIGINT, previous)
    print(f"{report.graph}: {report.status}, {report.wall_s:.2f} s wall, {report.stream_s:.2f} s stream")
    return report


def _widget(name: str, hint: object, default: object) -> list[Any] | None:
    import gradio as gr

    if get_origin(hint) in (Union, types.UnionType):
        hint = next(a for a in get_args(hint) if a is not type(None))
    value: Any = None if default is dataclasses.MISSING else default
    if hint is SegmentRef:
        segment: SegmentRef | None = value if isinstance(value, SegmentRef) else None
        return [gr.Textbox(segment.dataset if segment else "", label=f"{name}.dataset"), gr.Textbox(segment.segment if segment else "", label=f"{name}.segment")]
    if dataclasses.is_dataclass(hint) or name.startswith("output"):
        return None
    if get_origin(hint) is Literal:
        return [gr.Dropdown(list(get_args(hint)), value=value, label=name)]
    makers: dict[object, Callable[..., Any]] = {bool: gr.Checkbox, int: gr.Number, float: gr.Number, str: gr.Textbox}
    if hint is Path:
        return [gr.File(value=str(value) if value else None, label=name, type="filepath")]
    return [makers[hint](value=value, label=name)] if hint in makers else None


def gradio_app(config_type: type[G], build: Callable[[G, rr.RecordingStream], Graph], *, title: str, examples: Sequence[G] = ()) -> Any:
    """Generated UI (a `gr.Blocks`): a form from Config fields, Run (threaded runner into a Rerun pane), Stop, output files, examples."""
    import gradio as gr
    from gradio_rerun import Rerun

    hints: dict[str, object] = {f.name: f.type for f in dataclasses.fields(config_type)}  # type: ignore[arg-type]
    defaults: dict[str, object] = {f.name: f.default for f in dataclasses.fields(config_type)}  # type: ignore[arg-type]
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        control_state = gr.State(None)
        inputs: dict[str, list[Any]] = {}
        with gr.Row():
            with gr.Column(scale=1):
                for name, hint in hints.items():
                    if (widgets := _widget(name, hint, defaults[name])) is not None:
                        inputs[name] = widgets
                run_button, stop_button = gr.Button("Run", variant="primary"), gr.Button("Stop")
                outputs = [gr.File(label=n) for n in hints if n.startswith("output")]
            viewer = Rerun(streaming=True, height=720, scale=3)
        flat: list[Any] = [w for widgets in inputs.values() for w in widgets]

        def config_from(values: Sequence[object]) -> G:
            it: Iterator[object] = iter(values)
            kwargs: dict[str, object] = {}
            for name, widgets in inputs.items():
                got: list[object] = [next(it) for _ in widgets]
                kwargs[name] = SegmentRef(str(got[0]), str(got[1])) if hints[name] is SegmentRef else (Path(str(got[0])) if hints[name] is Path and got[0] else got[0])
            for name, hint in hints.items():
                if name not in kwargs and defaults[name] is dataclasses.MISSING and dataclasses.is_dataclass(hint):
                    fields = {f.name for f in dataclasses.fields(hint)}  # type: ignore[arg-type]
                    kwargs[name] = hint(**({"headless": True} if "headless" in fields else {}))  # type: ignore[operator]
            return config_type(**kwargs)

        def run(*values: object) -> Iterator[tuple[object, ...]]:
            config: G = config_from(values)
            recording = rr.RecordingStream(title.replace(" ", "_"), recording_id=uuid.uuid4().hex)
            stream = recording.binary_stream()
            control = Control()
            errors: list[BaseException] = []
            thread = threading.Thread(target=lambda: _guarded(errors, lambda: _run(build(config, recording), "threaded", getattr(config, "mode", "lossless"), control, recording)))
            thread.start()
            while thread.is_alive():
                time.sleep(0.2)
                yield (control, stream.read(), *(None for _ in outputs))
            recording.flush()
            if errors:
                raise gr.Error(repr(errors[0]))
            files = [str(getattr(config, n)) if getattr(config, n, None) and Path(str(getattr(config, n))).exists() else None for n in hints if n.startswith("output")]
            yield (control, stream.read(), *files)

        run_button.click(run, inputs=flat, outputs=[control_state, viewer, *outputs])
        stop_button.click(lambda control: control.stop() if control is not None else None, inputs=[control_state], outputs=[])
        if examples:
            gr.Examples([[v for name, ws in inputs.items() for v in _example_values(getattr(e, name), hints[name], len(ws))] for e in examples], inputs=flat)
    return demo


def _example_values(value: object, hint: object, count: int) -> list[object]:
    if isinstance(value, SegmentRef):
        return [value.dataset, value.segment]
    return [str(value) if isinstance(value, Path) else value][:count]


def _guarded(errors: list[BaseException], body: Callable[[], object]) -> None:
    try:
        body()
    except BaseException as exc:  # handed to the UI thread, which raises it as gr.Error (nothing swallowed)
        errors.append(exc)
