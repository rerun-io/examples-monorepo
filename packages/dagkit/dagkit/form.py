"""The gradio app generated from an App's Config: one widget group per field, Run streams ``build``'s graphs into a Rerun pane.

What a field is comes from its declaration, never its name: the viewer is AppConfig's ``rr_config`` (never a widget; the
form passes ``<its type>(headless=True)``), and a ``Path`` field states its widget in its metadata:
``field(..., metadata={"form": "dir"})`` is a directory typed as text (a file picker cannot choose one), and on a
``Path | None`` field ``{"form": "save"}`` is a "save it, then download it" box (the run writes into the form's scratch
directory and the form offers the result). Any other ``Path`` is a file picker.
"""

import contextlib
import dataclasses
import functools
import itertools
import shutil
import tempfile
import threading
import time
import types
import uuid
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import rerun as rr

from dagkit.app import AppConfig, _run_graphs
from dagkit.doc import field_docs
from dagkit.graph import Graph
from dagkit.node import Control
from dagkit.sources import SegmentRef

G = TypeVar("G")  # a form's Config type: not bound to AppConfig, as beartype cannot issubclass-check a data Protocol
VIEWER: str = "rr_config"
"""AppConfig's viewer field: never a widget, its default factory is never called (it may spawn a viewer)."""
Form: TypeAlias = Literal["dir", "save", ""]
"""A field's ``metadata["form"]``: ``dir``, ``save``, or "" (the widget its type gives)."""
Leaf: TypeAlias = tuple[tuple[str, ...], object, object, str, Form]  # (path, type hint, default, attribute docstring, form)


def _base(hint: object) -> Any:
    return next(a for a in get_args(hint) if a is not type(None)) if get_origin(hint) in (Union, types.UnionType) else hint


def _nested(hint: object, path: tuple[str, ...]) -> bool:
    """A Config dataclass the form opens up (SegmentRef and the viewer have their own handling)."""
    base: object = _base(hint)
    return dataclasses.is_dataclass(base) and base is not SegmentRef and path != (VIEWER,)


def _default(f: dataclasses.Field, path: tuple[str, ...]) -> object:
    """A field's default (MISSING if required); the viewer's factory is never called."""
    if f.default is not dataclasses.MISSING or f.default_factory is dataclasses.MISSING or path == (VIEWER,):
        return f.default
    return f.default_factory()


def _form(f: dataclasses.Field, hint: object) -> Form:
    form: object = f.metadata.get("form", "")
    if form not in ("dir", "save", "") or (form and _base(hint) is not Path) or (form == "save" and type(None) not in get_args(hint)):
        raise ValueError(f"{f.name}: metadata form={form!r}; 'dir' needs a Path field, 'save' a Path | None one")
    return cast(Form, form)


def leaves(cls: type, template: object, prefix: tuple[str, ...] = ()) -> list[Leaf]:
    """Every form field of a Config, nested dataclasses opened up; defaults come from ``template`` when there is one."""
    hints: dict[str, Any] = get_type_hints(cls)
    docs: dict[str, str] = field_docs(cls)
    out: list[Leaf] = []
    for f in dataclasses.fields(cls):
        path: tuple[str, ...] = (*prefix, f.name)
        default: object = _default(f, path) if template is None else getattr(template, f.name)
        if _nested(hints[f.name], path):
            out += leaves(_base(hints[f.name]), None if default in (None, dataclasses.MISSING) else default, path)
        else:
            out.append((path, hints[f.name], default, docs.get(f.name, ""), _form(f, hints[f.name])))
    return out


def _widgets(leaf: Leaf) -> list[Any]:
    """Widgets for a leaf; unsupported types (including the viewer) remain hidden."""
    import gradio as gr

    path, hint, default, doc, form = leaf
    label, base = ".".join(path), _base(hint)
    value: Any = None if default is dataclasses.MISSING else default
    optional: bool = type(None) in get_args(hint)
    if base is SegmentRef:
        window: str = " ".join(map(str, getattr(value, "window_ns", None) or ()))
        placeholder: str = "start stop (ns on the segment's timeline); empty: the whole segment"
        return [gr.Textbox(getattr(value, part, ""), label=f"{label}.{part}") for part in ("dataset", "segment")] + [gr.Textbox(window, label=f"{label}.window_ns", placeholder=placeholder)]
    if form == "save":
        return [gr.Checkbox(False, label=f"{label}: save it, then download it", info=doc)]
    members: list[object] = [a for a in (get_args(hint) if get_origin(hint) in (Union, types.UnionType) else (hint,)) if a is not type(None)]
    if all(get_origin(m) is Literal for m in members):
        return [gr.Dropdown([v for m in members for v in get_args(m)], value=value, label=label, info=doc)]
    if base is Path:
        return [gr.Textbox(str(value or ""), label=label, info=doc) if form == "dir" else gr.File(value=str(value) if value else None, label=label, type="filepath")]
    if base in (int, float) and optional:  # gr.Number shows and sends 0 for None: a text box keeps "empty" empty
        return [gr.Textbox("" if value is None else str(value), label=label, info=doc, placeholder="empty: the default")]
    makers: dict[object, Callable[..., Any]] = {bool: gr.Checkbox, int: lambda **kw: gr.Number(precision=0, **kw), float: gr.Number, str: gr.Textbox}
    return [makers[base](value=value, label=label, info=doc)] if base in makers else []


def _value(leaf: Leaf, got: list[Any], scratch: Path) -> object:
    """The Config value of one leaf's widgets: an emptied optional field is None, an empty required one (or an unchecked save
    box) MISSING, which keeps the default."""
    path, hint, default, _, form = leaf
    base: object = _base(hint)
    optional: bool = type(None) in get_args(hint)
    if base is SegmentRef:  # empty dataset and segment = no catalog input; edits keep the default's url and timeline
        template: SegmentRef = default if isinstance(default, SegmentRef) else SegmentRef("", "")
        window: tuple[int, ...] = tuple(int(x) for x in str(got[2] or "").replace(",", " ").split())
        if len(window) not in (0, 2):
            raise ValueError(f"{'.'.join(path)}.window_ns: two integers (start stop) or nothing, got {got[2]!r}")
        return dataclasses.replace(template, dataset=str(got[0]), segment=str(got[1]), window_ns=(window[0], window[1]) if window else None) if any(got[:2]) else None
    if form == "save":  # a scratch file named after the field: output_ply -> output.ply
        return scratch / path[-1].replace("_", ".", 1) if got[0] else dataclasses.MISSING
    if got[0] is None or (got[0] == "" and (optional or base in (Path, int, float))):
        return None if optional else dataclasses.MISSING
    return base(got[0]) if base in (Path, int, float) else got[0]  # the browser sends 20 for 20.0 and text from a text box


def _assemble(cls: type, given: dict[tuple[str, ...], object], template: object, prefix: tuple[str, ...] = ()) -> Any:
    """The Config from the set leaves: unset leaves keep their defaults, nested dataclasses are built when needed."""
    hints: dict[str, Any] = get_type_hints(cls)
    kwargs: dict[str, object] = {}
    for f in dataclasses.fields(cls):
        path: tuple[str, ...] = (*prefix, f.name)
        if path in given:
            kwargs[f.name] = given[path]
        elif path == (VIEWER,):
            kwargs[f.name] = _base(hints[f.name])(headless=True)  # the UI renders into its own pane: never spawn a viewer
        elif _nested(hints[f.name], path) and (any(p[: len(path)] == path for p in given) or _default(f, path) is dataclasses.MISSING):
            sub: object = getattr(template, f.name) if template is not None else _default(f, path)
            kwargs[f.name] = _assemble(_base(hints[f.name]), given, None if sub in (None, dataclasses.MISSING) else sub, path)
    return cls(**kwargs) if template is None else dataclasses.replace(cast(Any, template), **kwargs)


def _download(value: object) -> str | None:
    """A saved file as it is, a saved directory zipped (gr.File cannot return a directory)."""
    if not isinstance(value, Path) or not value.exists():
        return None
    return str(value) if value.is_file() else shutil.make_archive(str(value), "zip", value)


def gradio_app(config_type: type[G], build: Callable[[G, rr.RecordingStream], Graph | Iterator[Graph]], *, title: str, examples: Sequence[G] = ()) -> Any:
    """The form for ``config_type`` and a Rerun pane; Run builds the same graphs as the CLI (``build``) and streams them.

    A Config with a ``register`` field never registers a run that saves into the scratch directory."""
    import gradio as gr
    from gradio_rerun import Rerun

    first: object = examples[0] if examples else None  # a required field the Config leaves open starts from the first example
    form_leaves: list[Leaf] = [(p, h, functools.reduce(getattr, p, first) if d is dataclasses.MISSING and first is not None else d, doc, kind) for p, h, d, doc, kind in leaves(config_type, None)]
    outputs: list[Leaf] = [leaf for leaf in form_leaves if leaf[4] == "save"]
    with gr.Blocks(title=title, delete_cache=(3600, 3600)) as demo:  # hourly: drop downloads older than an hour from gradio's cache
        gr.Markdown(f"# {title}")
        control_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                form: list[tuple[Leaf, list[Any]]] = []
                for prefix, group in itertools.groupby(form_leaves, key=lambda leaf: leaf[0][:-1]):
                    with gr.Accordion(".".join(prefix), open=True) if prefix else contextlib.nullcontext():
                        form += [(leaf, w) for leaf in group if (w := _widgets(leaf))]
                run_button, stop_button = gr.Button("Run", variant="primary"), gr.Button("Stop")
                files = [gr.File(label=".".join(leaf[0]) + " (download)") for leaf in outputs]
            viewer = Rerun(streaming=True, height=720, scale=3)
        flat: list[Any] = [w for _, widgets in form for w in widgets]

        def run(*values: object) -> Iterator[tuple[object, ...]]:
            it: Iterator[object] = iter(values)
            with tempfile.TemporaryDirectory(prefix="dagkit-app-", ignore_cleanup_errors=True) as tmp:  # gone with the run: gradio
                scratch = Path(tmp)  # copies each download into its cache while the run waits at the yield that offers it
                try:
                    given: dict[tuple[str, ...], object] = {leaf[0]: v for leaf, ws in form if (v := _value(leaf, [next(it) for _ in ws], scratch)) is not dataclasses.MISSING}
                    config: AppConfig = _assemble(config_type, given, None)  # an instance check is fine
                except (TypeError, ValueError) as exc:  # a required field left empty, a malformed number or window
                    raise gr.Error(f"the form does not make a {config_type.__name__}: {exc}") from None
                if getattr(config, "register", "off") != "off" and any(isinstance(v, Path) and v.is_relative_to(scratch) for v in given.values()):
                    config = dataclasses.replace(cast(Any, config), register="off")  # a temporary output is never registered on the catalog
                recording = rr.RecordingStream(title.replace(" ", "_"), recording_id=uuid.uuid4().hex)
                stream, control, errors = recording.binary_stream(), Control(), []

                def body() -> None:
                    try:
                        _run_graphs(config, build, recording, control)
                    except BaseException as exc:  # delivered to the UI thread below
                        errors.append(exc)

                thread = threading.Thread(target=body)
                thread.start()
                try:
                    while thread.is_alive():
                        time.sleep(0.2)
                        yield (control, stream.read(), *(None for _ in files))
                finally:  # also when gradio closes this iterator at a yield: the run stops before its scratch directory goes
                    control.stop()
                    thread.join()
                recording.flush()
                if errors:
                    raise gr.Error(repr(errors.pop()))  # the list lets go of the traceback (and the tensors its frames hold)
                yield (control, stream.read(), *(_download(functools.reduce(getattr, leaf[0], config)) for leaf in outputs))

        run_button.click(run, inputs=flat, outputs=[control_state, viewer, *files])
        stop_button.click(lambda control: control and control.stop(), inputs=[control_state], outputs=[])
        if examples:
            table = gr.Examples([[v for leaf, ws in form for v in _example(leaf, e)[: len(ws)]] for e in examples], inputs=flat)
            if cleared := [w for w, kept in zip(flat, table.input_has_examples, strict=True) if not kept]:

                def clear(_: object) -> dict[Any, None]:  # gr.Examples skips columns that are None in every example: clear them
                    return dict.fromkeys(cleared)

                table.dataset.click(clear, inputs=[table.dataset], outputs=cleared)
    return demo


def _example(leaf: Leaf, example: object) -> list[object]:
    """Widget values of one leaf for an example Config."""
    value: object = functools.reduce(lambda obj, name: getattr(obj, name, None), leaf[0], example)
    if isinstance(value, SegmentRef) or _base(leaf[1]) is SegmentRef:
        return [getattr(value, "dataset", ""), getattr(value, "segment", ""), " ".join(map(str, getattr(value, "window_ns", None) or ()))]
    if leaf[4] == "save":
        return [value is not None]
    return [str(value) if isinstance(value, Path) else value]
