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
RUNNERS: dict[str, Callable[..., RunReport]] = {"sequential": run_sequential, "threaded": run_threaded}


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
    cpus: list[int] | None = None
    params: dict[str, Any] = field(default_factory=dict)  # overrides of the node's config dataclass fields


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class Profile:
    """`packages/<pkg>/profiles/<hardware>.toml`: node settings plus named operating points (presets)."""

    hardware: HardwareClass
    nodes: dict[str, NodeProfile] = field(default_factory=dict)
    presets: dict[str, dict[str, NodeProfile]] = field(default_factory=dict)  # preset -> node -> overlay

    def node(self, name: str, preset: str | None = None) -> NodeProfile:
        """Base merged with the preset overlay: overlay params win; a non-default overlay placement/cpus wins."""
        if preset is not None and preset not in self.presets:
            raise KeyError(f"unknown preset {preset!r}; this profile has {sorted(self.presets)}")
        base: NodeProfile = self.nodes.get(name, NodeProfile())
        overlay: NodeProfile | None = self.presets[preset].get(name) if preset is not None else None
        if overlay is None:
            return base
        placement = overlay.placement if overlay.placement != "thread" else base.placement
        return NodeProfile(placement, overlay.cpus if overlay.cpus is not None else base.cpus, base.params | overlay.params)

    def configure(self, name: str, default: D, preset: str | None = None) -> D:
        """The node's config (`@serde(type_check=coerce)`: params come from TOML) with profile/preset params over `default`."""
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
    """Read `<directory>/<hardware>.toml` (hand-written: coerced; unknown keys refused)."""
    resolved: HardwareClass = detect_hardware() if hardware == "auto" else hardware
    path: Path = directory / f"{resolved}.toml"
    try:
        profile: Profile = from_toml(Profile, path.read_text())
    except (SerdeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if profile.hardware != resolved:
        raise ValueError(f"{path} declares hardware {profile.hardware!r}, expected {resolved!r}")
    return profile


def run_app(config: C, build: Callable[[C, rr.RecordingStream], Graph]) -> RunReport:
    """The CLI entry: recording = `config.rr_config.rec_stream` (also gets the health view), SIGINT stops."""
    recording: rr.RecordingStream = config.rr_config.rec_stream
    control = Control()
    main: bool = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, lambda *_: control.stop()) if main else None
    try:
        report: RunReport = RUNNERS[config.runner](build(config, recording), mode=config.mode, control=control, health=recording)
    finally:
        if main:
            signal.signal(signal.SIGINT, previous)
    print(f"{report.graph}: {report.status}, {report.wall_s:.2f} s wall, {report.stream_s:.2f} s stream")
    return report


def _base(hint: object) -> object:
    """`X | None` -> X."""
    return next(a for a in get_args(hint) if a is not type(None)) if get_origin(hint) in (Union, types.UnionType) else hint


def _widgets(name: str, hint: object, default: object) -> list[Any]:
    """Form widgets of one Config field (none for nested dataclasses such as rr_config, and for output paths)."""
    import gradio as gr

    hint = _base(hint)
    value: Any = None if default is dataclasses.MISSING else default
    if hint is SegmentRef:
        return [gr.Textbox(getattr(value, part, ""), label=f"{name}.{part}") for part in ("dataset", "segment")]
    if dataclasses.is_dataclass(hint) or name.startswith("output"):
        return []
    if get_origin(hint) is Literal:
        return [gr.Dropdown(list(get_args(hint)), value=value, label=name)]
    if hint is Path:
        return [gr.File(value=str(value) if value else None, label=name, type="filepath")]
    makers: dict[object, Callable[..., Any]] = {bool: gr.Checkbox, int: lambda **kw: gr.Number(precision=0, **kw), float: gr.Number, str: gr.Textbox}
    return [makers[hint](value=value, label=name)] if hint in makers else []


def gradio_app(config_type: type[G], build: Callable[[G, rr.RecordingStream], Graph], *, title: str, examples: Sequence[G] = ()) -> Any:
    """Generated UI (a `gr.Blocks`): a form from Config fields, Run (threaded runner into a Rerun pane), Stop, output files."""
    import gradio as gr
    from gradio_rerun import Rerun

    fields: list[dataclasses.Field] = list(dataclasses.fields(config_type))  # type: ignore[arg-type]
    output_names: list[str] = [f.name for f in fields if f.name.startswith("output")]
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        control_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                inputs: dict[str, list[Any]] = {f.name: w for f in fields if (w := _widgets(f.name, f.type, f.default))}
                run_button, stop_button = gr.Button("Run", variant="primary"), gr.Button("Stop")
                files = [gr.File(label=name) for name in output_names]
            viewer = Rerun(streaming=True, height=720, scale=3)
        flat: list[Any] = [w for widgets in inputs.values() for w in widgets]

        def config_from(values: Sequence[object]) -> G:
            it: Iterator[object] = iter(values)
            kwargs: dict[str, object] = {}
            for f in fields:
                got: list[object] = [next(it) for _ in inputs.get(f.name, [])]
                if f.name in inputs:
                    base: object = _base(f.type)
                    kwargs[f.name] = SegmentRef(str(got[0]), str(got[1])) if base is SegmentRef else Path(str(got[0])) if base is Path and got[0] else got[0]
                elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and dataclasses.is_dataclass(f.type):
                    headless: bool = "headless" in {g.name for g in dataclasses.fields(f.type)}  # type: ignore[arg-type]
                    kwargs[f.name] = f.type(**({"headless": True} if headless else {}))  # type: ignore[operator]
            return config_type(**kwargs)

        def run(*values: object) -> Iterator[tuple[object, ...]]:
            config: G = config_from(values)
            recording = rr.RecordingStream(title.replace(" ", "_"), recording_id=uuid.uuid4().hex)
            stream, control, errors = recording.binary_stream(), Control(), []
            graph: Graph = build(config, recording)

            def body() -> RunReport:
                return run_threaded(graph, mode=getattr(config, "mode", "lossless"), control=control, health=recording)

            thread = threading.Thread(target=_guarded, args=(errors, body))
            thread.start()
            while thread.is_alive():
                time.sleep(0.2)
                yield (control, stream.read(), *(None for _ in files))
            recording.flush()
            if errors:
                raise gr.Error(repr(errors[0]))
            written: list[str | None] = [str(p) if (p := getattr(config, name)) and Path(str(p)).exists() else None for name in output_names]
            yield (control, stream.read(), *written)

        run_button.click(run, inputs=flat, outputs=[control_state, viewer, *files])
        stop_button.click(lambda control: control and control.stop(), inputs=[control_state], outputs=[])
        if examples:
            gr.Examples([[v for name, ws in inputs.items() for v in _example_values(getattr(e, name))[: len(ws)]] for e in examples], inputs=flat)
    return demo


def _example_values(value: object) -> list[object]:
    return [value.dataset, value.segment] if isinstance(value, SegmentRef) else [str(value) if isinstance(value, Path) else value]


def _guarded(errors: list[BaseException], body: Callable[[], object]) -> None:
    try:
        body()
    except BaseException as exc:  # handed to the UI thread, which raises it as gr.Error (nothing swallowed)
        errors.append(exc)
