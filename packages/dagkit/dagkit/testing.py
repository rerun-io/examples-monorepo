import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import rerun as rr
from rerun.chunk import RrdReader

from dagkit.app import _leaves, gradio_app
from dagkit.doc import GRAPH_ROOT, GraphDoc, describe, guide, log_graph
from dagkit.graph import AsOf, DropOldest, Graph, Spawn, normalise
from dagkit.node import Emit, Event, Fenced, Node, RunContext, Tick
from dagkit.time import STATIC, TimeNs, check_monotonic
from dagkit.types import Type

Output = tuple[str, TimeNs, object]


class Recorder:
    def __init__(self) -> None:
        self.items: list[Output] = []

    def put(self, port: str, t_ns: TimeNs, value: object) -> None:
        self.items.append((port, t_ns, value))

    def watermark(self, t_ns: TimeNs) -> None:
        pass


def host(value: object) -> object:
    """``value`` as numpy compares it: Fenced unwrapped, dataclasses as dicts, sequences as lists, tensors (anything with
    ``detach``; dagkit never imports torch) as host arrays."""
    value = value.value if isinstance(value, Fenced) else value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: host(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: host(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [host(v) for v in value]
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value


def check_node_contract(node: Node, ticks: Sequence[Tick], ctx: RunContext | None = None) -> list[Output]:
    ctx = ctx or RunContext("contract", "node", "lossless")
    runs: list[list[Output]] = []
    for resets in (1, 2):
        out = Recorder()
        for _ in range(resets):
            node.reset(ctx)
        for tick in ticks:
            node.step(tick, out)
        node.flush(out)
        drained: int = len(out.items)
        node.flush(out)
        assert len(out.items) == drained, "a second flush emitted outputs"
        last: dict[str, TimeNs] = {}
        for port, t_ns, _ in out.items:
            assert port in node.outputs, f"put on undeclared port {port!r}"
            check_monotonic(last.get(port), t_ns, f"output {port}")
            last[port] = t_ns
        runs.append(out.items)
    plain: list[list[object]] = [[(p, t, host(v)) for p, t, v in run] for run in runs]
    np.testing.assert_equal(plain[1], plain[0], err_msg="a run after reset differs from the first run (reset is not idempotent)")  # arrays by value
    return runs[0]


class _Spy:
    def __init__(self, node: object, seen: list[object]) -> None:
        self.node, self.seen = node, seen

    def __getattr__(self, name: str) -> object:  # inputs, outputs, emits_late, reset, flush, abort
        return getattr(self.node, name)

    def step(self, tick: Tick, out: Emit) -> None:
        self.seen.append((tick.port, tick.t_ns, tick.value, dict(tick.asof)))
        self.node.step(tick, out)  # type: ignore[attr-defined]

    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None:
        self.seen.append({port: list(events) for port, events in inputs.items()})
        self.node.run(inputs, out)  # type: ignore[attr-defined]


def layer_table(values: Sequence[tuple[TimeNs, Type]], root: str, timeline: str = "video_time") -> pa.Table:
    """What a catalog reader returns for these values (no server): one row per time, ``<entity>:<component>`` columns,
    statics repeated in every row (one row when every value is static)."""
    cells: dict[str, dict[TimeNs, list]] = defaultdict(dict)
    statics: dict[str, list] = {}
    for t_ns, value in values:
        for log in value.to_rerun(root):
            for batch in log.data.as_component_batches():
                name: str = f"{log.path}:{batch.component_descriptor()}"
                if log.static or t_ns == STATIC:
                    statics[name] = batch.as_arrow_array().to_pylist()
                else:
                    cells[name][t_ns] = batch.as_arrow_array().to_pylist()
    times: list[TimeNs] = sorted({t for t, _ in values if t != STATIC}) or [0]
    columns: dict[str, Any] = {timeline: pa.array(times, type=pa.int64())} | {n: [c.get(t) for t in times] for n, c in cells.items()}
    return pa.table(columns | {n: [v] * len(times) for n, v in statics.items()})


def check_form(config_type: type, build: Callable[..., object], widgets: Mapping[str, str], hidden: Mapping[str, str], *, examples: Sequence = ()) -> Any:
    """The generated form shows ``widgets`` (form label without its ": save it" note -> gradio class name, in form order) and
    leaves out the ``hidden`` Config fields (each with its reason): every field is in exactly one of them. Returns the app."""
    demo = gradio_app(config_type, build, title=config_type.__name__, examples=examples)  # type: ignore[arg-type]
    run = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "run")
    shown: list[tuple[str, str]] = [(str(w.label).split(":")[0], type(w).__name__) for w in run.inputs]
    assert shown == list(widgets.items()), f"the form shows {shown}"
    for path in (".".join(leaf[0]) for leaf in _leaves(config_type, None)):
        has_widget: bool = any(label == path or label.startswith(f"{path}.") for label, _ in shown)
        assert has_widget != (path in hidden), f"{path}: {'shown and hidden' if has_widget else 'neither shown nor hidden on purpose'}"
    return demo


def collect_edges(graph: Graph, run: Callable[[Graph], object]) -> dict[str, list[object]]:
    seen: dict[str, list[object]] = defaultdict(list)
    spied = Graph(graph.name)
    spied.resources, spied.edges = dict(graph.resources), list(graph.edges)
    for name, spec in graph.nodes.items():
        keep: bool = spec.kind == "source" or isinstance(spec.obj, Spawn)
        spied.nodes[name] = spec if keep else replace(spec, obj=_Spy(spec.obj, seen[name]))
    run(spied)
    return dict(seen)


def drawn_problems(graph: Graph, ids: Sequence[str], labels: Sequence[str], lines: Sequence[tuple[str, str]], edges: Sequence[str]) -> list[str]:
    """Check boxes (one per node), edge coverage, what each label shows (port; into a step also its Type, GPU mark and how it
    arrives) and every line (it joins two items and carries a declared edge) independently against the declaration."""
    box: dict[str, str] = {i: label for i, label in zip(ids, labels, strict=True) if i in graph.nodes}
    problems: list[str] = [f"node {n} has no box" for n in graph.nodes if n not in box]
    problems += [f"node {i}: {c} boxes" for i, c in Counter(ids).items() if c > 1]
    problems += [f"box {n} reads {label.split(chr(10))[0]!r}" for n, label in box.items() if label.split("\n")[0].split("  ", 1)[-1] != n]
    how: dict[type, str] = {AsOf: "latest", DropOldest: "drops old"}
    declared: Counter[str] = Counter(f"{e.src.node}.{e.src.port}->{e.dst.node}.{e.dst.port} {how.get(type(e.kind), 'waits')}" for e in graph.edges)
    drawn: Counter[str] = Counter(line for text in edges for line in text.splitlines())
    problems += [f"declared, drawn {drawn[e]} times: {e}" for e in declared if drawn[e] != 1] + [f"drawn, not declared: {e}" for e in drawn if e not in declared]
    pairs: set[tuple[str, str]] = set(lines)
    problems += [f"line {a} -> {b} joins an unknown item" for a, b in lines if a not in ids or b not in ids]
    meant: set[tuple[str, str]] = set()  # the lines the drawn edges account for
    for item, label, text in zip(ids, labels, edges, strict=True):
        for line in text.splitlines():
            connection, kind = line.split(" ", 1)
            (src_node, src_port), (dst_node, dst_port) = [port.rsplit(".", 1) for port in connection.split("->")]
            into_step: bool = dst_node in graph.nodes and graph.nodes[dst_node].kind != "sink"
            meant |= {(item, dst_node)} | (set() if item.endswith(" <-") else {(src_node, item)})
            problems += [f"{item} does not point at {dst_node}"] if (item, dst_node) not in pairs else []
            problems += [f"{item} is not fed by {src_node}"] if not item.endswith(" <-") and (src_node, item) not in pairs else []
            want: str = re.escape(src_port)
            if into_step and src_port in graph.nodes.get(src_node, graph.nodes[dst_node]).outputs:
                fenced, classes = normalise(graph.nodes[src_node].outputs[src_port])
                shown: str = " | ".join(c.__name__ for c in classes) + (" (GPU)" if fenced else "")
                want = rf"^{re.escape(dst_port if src_port == dst_port else f'{src_port} as {dst_port}')}: {re.escape(shown)} (← .+ )?· {kind}$"
            problems += [f"{item}: {label!r} lacks {want!r}"] if not any(re.search(want, row) for row in label.splitlines()) else []
    return problems + [f"line {a} -> {b} is not a declared edge" for a, b in lines if (a, b) not in meant]


def picture_problems(graph: Graph, rrd: Path) -> list[str]:
    """``drawn_problems`` of the graph view an rrd holds (the newest static picture at ``/dagkit/graph/<graph>``)."""
    reader = RrdReader(rrd)
    cols: dict[str, list] = {}
    for chunk in (c for c in reader.stream(store=reader.recordings()[0]).to_chunks() if str(c.entity_path) == GRAPH_ROOT + graph.name):
        batch = chunk.to_record_batch()
        cols |= {name: batch.column(name)[-1].as_py() for name in ("GraphNodes:node_ids", "GraphNodes:labels", "GraphEdges:edges", "edges") if name in batch.schema.names}
    if len(cols) < 4:
        return [f"no picture of {graph.name!r} in {rrd}"]
    lines: list[tuple[str, str]] = [(e["first"], e["second"]) for e in cols["GraphEdges:edges"]]
    return drawn_problems(graph, cols["GraphNodes:node_ids"], cols["GraphNodes:labels"], lines, cols["edges"])


def check_picture(graph: Graph, directory: Path) -> Path:
    """Log the graph view to ``<directory>/<graph>.rrd`` as the runners do and read it back: the picture must equal the declaration."""
    path: Path = directory / f"{graph.name}.rrd"
    recording = rr.RecordingStream("dagkit-picture", recording_id=graph.name, send_properties=False)
    recording.save(path)
    log_graph(recording, graph)
    recording.flush()
    recording.disconnect()
    problems: list[str] = picture_problems(graph, path)
    assert not problems, "\n".join(problems)
    return path


def check_graph_docs(*graphs: Graph, snapshot: Path | None = None) -> None:
    """Check node titles, port docs, graph descriptions and unique links. Compare unlinked guides with ``snapshot``;
    a missing file or ``DAGKIT_UPDATE_SNAPSHOTS=1`` writes it."""
    problems: list[str] = []
    docs: list[GraphDoc] = [describe(graph) for graph in graphs]
    for doc in docs:
        problems += [] if doc.about.strip() else [f"graph {doc.name!r}: give Graph(..., about=...) one or two sentences"]
        for n in doc.nodes:
            problems += [f"{n.cls} ({n.code.split('#')[0]}): its docstring needs a first line of <= 40 chars (has {n.title!r})"] if not 0 < len(n.title) <= 40 else []
            problems += [f"{p.type} ({p.code.split('#')[0]}): the port Type needs a docstring" for p in [*n.inputs, *n.outputs] if p.code and not p.doc]
        problems += [f"graph {doc.name!r}: link text {t!r} repeats" for t, c in Counter(re.findall(r"\[([^\]]+)\]\(", guide(doc, health=True))).items() if c > 1]
    if snapshot is not None:
        text: str = "<!-- generated by dagkit.testing.check_graph_docs; DAGKIT_UPDATE_SNAPSHOTS=1 rewrites it -->\n" + "\n".join(guide(doc, links=False) for doc in docs)
        if os.environ.get("DAGKIT_UPDATE_SNAPSHOTS") == "1" or not snapshot.exists():
            snapshot.write_text(text)
        elif snapshot.read_text() != text:
            problems.append(f"{snapshot} is stale: run the test with DAGKIT_UPDATE_SNAPSHOTS=1 and commit the file")
    assert not problems, "\n".join(dict.fromkeys(problems))
