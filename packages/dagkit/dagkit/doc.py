"""Describe a Graph as typed facts, a GraphView picture and a linked markdown guide; log all three with ``log_graph``."""

import ast
import dataclasses
import inspect
import itertools
import textwrap
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias, cast

import rerun as rr
from serde import serde
from serde.json import to_json

from dagkit.graph import AsOf, DropOldest, Graph, NodeKind, Spawn, normalise
from dagkit.node import PortType
from dagkit.sinks import LayerFileSink
from dagkit.types import git

How: TypeAlias = Literal["waits", "latest", "drops old"]
Rgb: TypeAlias = tuple[int, int, int]
GRAPH_ROOT, GUIDE_ROOT, HEALTH_ROOT = "/dagkit/graph/", "/dagkit/guide/", "/dagkit/health/"
REPO_URL: str = "https://github.com/rerun-io/examples-monorepo"
ROLE: dict[str, tuple[str, Rgb, str]] = {
    "source": ("source", (90, 170, 255), "source"), "node": ("once per item", (255, 180, 70), "runs once per item"),
    "segment": ("once per segment", (200, 130, 255), "runs once on the whole segment"), "sink": ("output", (110, 220, 130), "output"),
}  # fmt: skip
KIND_RGB: dict[str, Rgb] = {"waits": (125, 135, 150), "latest": (40, 170, 200), "drops old": (240, 120, 40), "": (135, 135, 135)}  # dark and light themes


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PortDoc:
    name: str
    type: str  # class names, `A | B` for a union
    doc: str  # first sentence of the Type's own docstring
    gpu: bool  # a Fenced port: the value stays on the GPU
    code: str  # `path/in/repo.py#L<line>` ('' for builtins and unions)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class EdgeDoc:
    src: str
    src_port: str
    dst: str
    dst_port: str
    how: How
    capacity: int  # queued items before the sender waits (Block) or old ones drop (DropOldest); 0 for AsOf


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ParamDoc:
    name: str
    value: str
    doc: str


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class NodeDoc:
    name: str
    number: int  # reading order: box number and guide section
    kind: NodeKind
    cls: str
    title: str  # first line of the class docstring
    about: str  # the rest of it, else the module docstring's first paragraph
    code: str
    placement: str
    needs: list[str]
    inputs: list[PortDoc]
    outputs: list[PortDoc]
    params: list[ParamDoc]
    layer: str = ""  # the catalog layer a layer sink publishes
    drawn: dict[str, list[str]] = field(default_factory=dict)  # sink input port -> entity paths it logged on the viewer recording


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class GraphDoc:
    name: str
    about: str
    commit: str  # the code links point at it ('' outside a git checkout)
    nodes: list[NodeDoc]  # in reading order
    edges: list[EdgeDoc]
    resources: list[str]


def _code(cls: type) -> str:
    """``path/in/repo.py#L<line>`` of a class (its module name outside the checkout)."""
    try:
        path, line = Path(inspect.getsourcefile(cls) or "").resolve(), inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        return cls.__module__
    root: str = git("rev-parse", "--show-toplevel")
    return f"{path.relative_to(root)}#L{line}" if root and path.is_relative_to(root) else cls.__module__


def own_doc(cls: type) -> str:
    """The class's own docstring ('' without one; the signature a dataclass generates does not count)."""
    doc: str = vars(cls).get("__doc__") or ""
    return "" if doc.startswith(f"{cls.__name__}(") else inspect.cleandoc(doc)


def _sentence(text: str) -> str:
    return " ".join(text.split("\n\n")[0].split()).split(". ")[0].rstrip(".")


def field_docs(cls: type) -> dict[str, str]:
    """Attribute docstrings (the string right below a field), as tyro reads them."""
    try:
        body: list[ast.stmt] = cast(ast.ClassDef, ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]).body
    except (OSError, TypeError):
        return {}
    return {
        a.target.id: _sentence(b.value.value)
        for a, b in itertools.pairwise(body)
        if isinstance(a, ast.AnnAssign) and isinstance(a.target, ast.Name) and isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str)
    }  # fmt: skip


def _port(name: str, port_type: PortType) -> PortDoc:
    fenced, classes = normalise(port_type)
    own: type | None = classes[0] if len(classes) == 1 and classes[0].__module__ != "builtins" else None
    return PortDoc(name, " | ".join(c.__name__ for c in classes), _sentence(own_doc(own)) if own else "", fenced, _code(own) if own else "")


def _ranks(names: Sequence[str], pairs: Sequence[tuple[str, str]]) -> dict[str, int]:
    """Longest path from a source (``names`` in topological order)."""
    rank: dict[str, int] = {}
    for name in names:
        rank[name] = 1 + max((rank[src] for src, dst in pairs if dst == name), default=-1)
    return rank


def describe(graph: Graph, drawn: Mapping[str, Mapping[str, Collection[str]]] | None = None) -> GraphDoc:
    """The facts of a valid graph; ``drawn`` = per sink, per input port, the entity paths it logged (``RerunSink.drawn``)."""
    graph.validate()
    rank: dict[str, int] = _ranks(graph.order(), [(e.src.node, e.dst.node) for e in graph.edges])
    nodes: list[NodeDoc] = []
    for number, name in enumerate(sorted(graph.nodes, key=lambda n: (rank[n], graph.nodes[n].kind == "sink")), 1):
        spec = graph.nodes[name]
        cls: type = spec.obj.cls if isinstance(spec.obj, Spawn) else type(spec.obj)
        title, _, about = own_doc(cls).partition("\n")
        module_doc: str = inspect.cleandoc(getattr(inspect.getmodule(cls), "__doc__", None) or "").split("\n\n")[0]
        params: object = getattr(spec.obj, "params", None)
        has_params: bool = dataclasses.is_dataclass(params) and not isinstance(params, type)
        docs: dict[str, str] = field_docs(type(params)) if has_params else {}
        values: list[ParamDoc] = [ParamDoc(f.name, repr(getattr(params, f.name)), docs.get(f.name, "")) for f in dataclasses.fields(params)] if has_params else []  # type: ignore[arg-type]
        nodes.append(NodeDoc(
            name, number, spec.kind, cls.__name__, title.strip().rstrip("."), " ".join((about.strip() or module_doc).split()), _code(cls), spec.placement.where,
            list(spec.needs), [_port(p, t) for p, t in spec.inputs.items()], [_port(p, t) for p, t in spec.outputs.items()], values,
            spec.obj.layer if isinstance(spec.obj, LayerFileSink) else "", {p: sorted(paths) for p, paths in (drawn or {}).get(name, {}).items()},
        ))  # fmt: skip
    edges: list[EdgeDoc] = [EdgeDoc(e.src.node, e.src.port, e.dst.node, e.dst.port,
        "latest" if isinstance(e.kind, AsOf) else "drops old" if isinstance(e.kind, DropOldest) else "waits",
        0 if isinstance(e.kind, AsOf) else e.kind.capacity) for e in graph.edges]  # fmt: skip
    return GraphDoc(graph.name, graph.about, git("rev-parse", "HEAD"), nodes, edges, sorted(graph.resources))


@dataclass(slots=True)
class Item:
    """One GraphView node: a step's box, the data label on a line, or a box's tag of further inputs."""

    id: str
    label: str
    color: Rgb
    does: str  # one line for the selection panel
    edges: str = ""  # the declared edges it draws, one `src.port->dst.port how` per line (the read-back check parses it)
    x: float = 0.0
    y: float = 0.0

    @property
    def size(self) -> tuple[float, float]:
        lines: list[str] = self.label.split("\n")
        return max(map(len, lines)) * 6.6 + 16.0, len(lines) * 15.0 + 10.0

    def rect(self) -> tuple[float, float, float, float]:
        w, h = self.size
        return self.x - w / 2, self.x + w / 2, self.y - h / 2, self.y + h / 2


def crosses(a: Item, b: Item, items: Sequence[Item], margin: float = 3.0) -> bool:
    """Does the straight line a -> b pass under a third item (the viewer draws lines under the boxes)? Liang-Barsky clip."""
    dx, dy = b.x - a.x, b.y - a.y
    for item in (i for i in items if i.id not in (a.id, b.id)):
        x0, x1, y0, y1 = item.rect()
        low, high, hit = 0.0, 1.0, True
        for p, q in ((-dx, a.x - x0 + margin), (dx, x1 + margin - a.x), (-dy, a.y - y0 + margin), (dy, y1 + margin - a.y)):
            if p == 0:
                hit = hit and q >= 0
            elif p < 0:
                low = max(low, q / p)
            else:
                high = min(high, q / p)
        if hit and low <= high:
            return True
    return False


def picture(doc: GraphDoc) -> tuple[list[Item], list[tuple[str, str]]]:
    """GraphView items (boxes first, in reading order) and lines. A line that would pass under a third item becomes part of a tag."""
    nodes: dict[str, NodeDoc] = {n.name: n for n in doc.nodes}
    out_type: dict[tuple[str, str], PortDoc] = {(n.name, p.name): p for n in doc.nodes for p in n.outputs}
    groups: dict[tuple[str, str, str], list[EdgeDoc]] = defaultdict(list)  # (src, dst, dst port; '' into a sink: its ports share a label)
    for e in sorted(doc.edges, key=lambda e: nodes[e.src].number):
        groups[(e.src, e.dst, "" if nodes[e.dst].kind == "sink" else e.dst_port)].append(e)
    rank: dict[str, int] = _ranks(list(nodes), [(e.src, e.dst) for e in doc.edges])
    side: dict[str, str] = {}  # sink -> the one box feeding it, if it is that box's first such sink
    for n in (n for n in doc.nodes if n.kind == "sink"):
        if len(feeders := {e.src for e in doc.edges if e.dst == n.name}) == 1 and (feeder := feeders.pop()) not in side.values():
            side[n.name] = feeder
    last: int = max((rank[n] for n in nodes if n not in side), default=0)
    rank |= {n: last for n in nodes if nodes[n].kind == "sink" and n not in side} | {s: rank[p] for s, p in side.items()}

    def text(key: tuple[str, str, str], tag: bool) -> str:
        src, _, port = key
        es: list[EdgeDoc] = groups[key]
        frm: str = f" ← {nodes[src].number} {src}" if tag else ""
        if not port:
            return textwrap.fill(", ".join(e.src_port for e in es), 30, break_long_words=False) + frm
        typ: PortDoc = out_type[(src, es[0].src_port)]
        name: str = port if es[0].src_port == port else f"{es[0].src_port} as {port}"
        return f"{name}: {typ.type}{' (GPU)' if typ.gpu else ''}{frm} · {es[0].how}"

    def lines(keys: Sequence[tuple[str, str, str]]) -> str:
        return "\n".join(f"{e.src}.{e.src_port}->{e.dst}.{e.dst_port} {e.how}" for k in keys for e in groups[k])

    boxes: dict[str, Item] = {}
    for n in doc.nodes:
        how: str = f"output · layer {n.layer}" if n.layer else ROLE[n.kind][0]
        how += "" if n.placement == "thread" or n.kind in ("source", "sink") else f" · {n.placement}"
        label: str = f"{n.number}  {n.name}\n{n.title or n.cls}\n{how}" + (f" · uses {', '.join(n.needs)}" if n.needs else "")
        boxes[n.name] = Item(n.name, label, ROLE[n.kind][1], f"{n.title} ({n.cls}, {n.code})")
    sides: dict[str, str] = {p: s for s, p in side.items()}
    tagged: set[tuple[str, str, str]] = {k for k, es in groups.items() if k[1] not in side and (es[0].how == "latest" or rank[k[1]] - rank[k[0]] != 1)}
    while True:
        tags: dict[str, Item] = {}
        for dst in nodes:
            if keys := [k for k in groups if k in tagged and k[1] == dst]:
                tags[dst] = Item(f"{dst} <-", "\n".join(text(k, True) for k in keys), KIND_RGB[groups[keys[0]][0].how if keys[0][2] else ""], "inputs from further up", lines(keys))
        labels: dict[tuple[str, str, str], Item] = {
            k: Item(f"{k[0]} -> {k[1]}" + (f".{k[2]}" if k[2] else ""), text(k, False), KIND_RGB[es[0].how if k[2] else ""], "data on a line", lines([k]))
            for k, es in groups.items() if k not in tagged
        }  # fmt: skip
        gap_mid: dict[int, float] = {}
        y: float = 0.0
        for r in range(last + 1):
            if not (row := [n for n in nodes if rank[n] == r and n not in side]):
                continue
            if r > 0:
                gap: float = max((lab.size[1] for k, lab in labels.items() if rank[k[1]] == r and k[1] not in side), default=0.0) + 40
                gap_mid[r], y = y + gap / 2, y + gap
            want: dict[str, list[float]] = {n: [boxes[k[0]].x for k in labels if k[1] == n] for n in row}
            row.sort(key=lambda n: (sum(want[n]) / len(want[n]) if want[n] else float("-inf"), nodes[n].number))
            height: float = max(max(boxes[n].size[1], tags[n].size[1] if n in tags else 0.0, boxes[sides[n]].size[1] if n in sides else 0.0, labels[(n, sides[n], '')].size[1] if n in sides else 0.0) for n in row)
            right: float | None = None
            for n in row:
                box, w = boxes[n], boxes[n].size[0]
                indent: float = tags[n].size[0] + 24 if n in tags else 0.0
                box.x, box.y = sum(want[n]) / len(want[n]) if want[n] else 0.0, y + height / 2
                box.x = box.x if right is None else max(box.x, right + 40 + indent + w / 2)
                if n in tags:
                    tags[n].x, tags[n].y = box.x - w / 2 - 24 - tags[n].size[0] / 2, box.y
                right = box.x + w / 2
                if n in sides:  # a side output: label and sink on the box's centre line
                    lab, sink = labels[(n, sides[n], "")], boxes[sides[n]]
                    lab.x, lab.y = right + 20 + lab.size[0] / 2, box.y
                    sink.x, sink.y = right + 40 + lab.size[0] + sink.size[0] / 2, box.y
                    right = sink.x + sink.size[0] / 2
            y += height
        between: dict[tuple[str, str, str], Item] = {k: lab for k, lab in labels.items() if k[1] not in side}
        for k, lab in between.items():
            lab.x, lab.y = (boxes[k[0]].x + boxes[k[1]].x) / 2, gap_mid[rank[k[1]]]
        for _, gap_labels in itertools.groupby(sorted(between.values(), key=lambda lab: (lab.y, lab.x)), key=lambda lab: lab.y):
            for a, b in itertools.pairwise(gap_labels):  # side by side in one gap: push the right one over
                b.x = max(b.x, a.rect()[1] + 8 + b.size[0] / 2)
        items: list[Item] = [*boxes.values(), *labels.values(), *tags.values()]
        bad: set[tuple[str, str, str]] = {k for k, lab in between.items() if crosses(boxes[k[0]], lab, items) or crosses(lab, boxes[k[1]], items)}
        if not bad:
            return items, [(k[0], lab.id) for k, lab in labels.items()] + [(lab.id, k[1]) for k, lab in labels.items()] + [(t.id, n) for n, t in tags.items()]
        tagged |= bad


def guide(doc: GraphDoc, *, links: bool = True, health: bool = False) -> str:
    """Markdown: how to read the picture, one section per node in reading order, the data types. Link texts are unique."""
    root: str = GRAPH_ROOT + doc.name
    by: dict[str, NodeDoc] = {n.name: n for n in doc.nodes}
    into: dict[str, list[EdgeDoc]] = defaultdict(list)
    out_of: dict[tuple[str, str], list[EdgeDoc]] = defaultdict(list)
    seen: dict[tuple[str, str], list[str]] = defaultdict(list)  # (producer, port) -> entity paths drawn by any sink
    for e in doc.edges:
        into[e.dst].append(e)
        out_of[(e.src, e.src_port)].append(e)
        seen[(e.src, e.src_port)] += [p for p in by[e.dst].drawn.get(e.dst_port, []) if p not in seen[(e.src, e.src_port)]]

    def link(path: str, text: str) -> str:
        return f"[{text}](recording://{path.lstrip('/')})" if links else ""

    def code(where: str, text: str) -> str:
        return f"[{text}]({REPO_URL}/blob/{doc.commit}/{where})" if links and doc.commit and "#L" in where else f"`{where.split('#')[0]}`"

    md: list[str] = [
        f"# {doc.name}", doc.about, "",
        "**Read the picture top to bottom.** A box is one step: its number and name, what it does, how it runs (blue = source, "
        "orange = once per item, purple = once on the whole segment, green = output; while a run is live: grey = no input yet, "
        "green = ok, yellow = its queue is half full, red = it dropped items). A label on a line is the data that flows, "
        "`port: Type`, and how it arrives:",
        "- **waits**: every item, in order; the sender waits while the queue is full (nothing is lost)",
        "- **latest**: the newest value at or before the step's time; it never starts the step",
        "- **drops old**: only the newest items; older ones are dropped while the step is busy", "",
        "`(GPU)` after a type: the value stays on the GPU. A tag left of a box lists its inputs from further up "
        "(`port: Type ← 1 video · latest`). Box numbers match the sections below." + (" Click a link to select the data in the views of this tab." if links else ""), "",
    ]  # fmt: skip
    for n in doc.nodes:
        how: str = ROLE[n.kind][2]
        how += "" if n.placement == "thread" or n.kind in ("source", "sink") else f", placed on {n.placement}"
        refs: str = " · ".join(r for r in (code(n.code, f"code of {n.number} {n.name}"), link(f"{root}[#{n.number - 1}]", f"box {n.number} {n.name}")) if r)
        md += [f"## {n.number} · {n.name}: {n.title or n.cls}", f"*{how}* · `{n.cls}` · {refs}", ""] + ([n.about, ""] if n.about else [])
        if n.kind != "sink" and into[n.name]:
            md += ["| takes | type | from | arrives |", "|---|---|---|---|"]
            for e in into[n.name]:
                p: PortDoc = next(p for p in n.inputs if p.name == e.dst_port)
                md.append(f"| **{e.dst_port}** | `{p.type}`{' (GPU)' if p.gpu else ''} | {by[e.src].number} {e.src} | {e.how}{f' ({e.capacity})' if e.capacity else ''} |")
            md.append("")
        if n.kind != "sink" and n.outputs:
            md += ["| gives | type | to |", "|---|---|---|"]
            see: list[str] = []
            for p in n.outputs:
                to: list[str] = [f"{by[name].number} {name}" for name in sorted({e.dst for e in out_of[(n.name, p.name)]}, key=lambda name: by[name].number)]
                md.append(f"| **{p.name}** | `{p.type}`{' (GPU)' if p.gpu else ''} | {', '.join(to) or 'nobody (unused)'} |")
                drawn: list[str] = seen.get((n.name, p.name), [])
                paths: list[str] = [path for path in drawn if not any(other.startswith(path + "/") for other in drawn)]  # leaves: a parent may outline nothing
                if links and paths:  # one link per line: a wrapped link misses its click
                    see += [f"- {link(path, f'{n.name}.{p.name} → {path}')}" for path in paths[:2]] + ([f"- {n.name}.{p.name}: +{len(paths) - 2} more"] if len(paths) > 2 else [])
            md += ["", "See it:", *see, ""] if see else [""]
        if n.kind == "sink":
            fed: dict[str, list[str]] = defaultdict(list)
            for e in into[n.name]:
                fed[e.src].append(e.dst_port)
            md += [("Saves " if n.layer else "Takes ") + "; ".join(f"**{', '.join(ps)}** from {by[s].number} {s}" for s, ps in fed.items())
                   + (f" into the catalog layer `{n.layer}`." if n.layer else "."), ""]  # fmt: skip
        md += [f"Uses the graph resources {', '.join(f'`{r}`' for r in n.needs)} (built once, shared).", ""] if n.needs else []
        if n.params:
            md += ["Settings:", *(f"- `{p.name}`" + (f" = {p.value[:40]}" if links else "") + (f": {p.doc}" if p.doc else "") for p in n.params[:5])]
            md += [f"- +{len(n.params) - 5} more", ""] if len(n.params) > 5 else [""]
        if links and health and n.kind != "source":
            live: list[str] = [link(f"{HEALTH_ROOT}{n.name}/step_ms", f"{n.name} step ms")]
            live += [link(f"{HEALTH_ROOT}{n.name}/queue_depth/{e.dst_port}", f"{n.name} queue {e.dst_port}") for e in into[n.name] if e.how != "latest"]
            md += ["Live: " + " · ".join(live), ""]
    types: dict[str, PortDoc] = {p.type: p for n in doc.nodes for p in [*n.inputs, *n.outputs]}
    md += ["## Data types", "| type | what it is | code |", "|---|---|---|"]
    md += [f"| `{t}` | {p.doc or ('(builtin)' if not p.code else '(no docstring)')} | {code(p.code, f'{t} code') if p.code else ''} |" for t, p in sorted(types.items())]
    md += ["", f"Generated by `dagkit.doc` from the code at `{doc.commit[:7]}`. Machine-readable: {link(f'{root}/describe', f'{doc.name} describe json')}."] if links else []
    return "\n".join(md) + "\n"


def log_graph(
    recording: rr.RecordingStream, graph: Graph, *, drawn: Mapping[str, Mapping[str, Collection[str]]] | None = None, health: bool = False
) -> tuple[list[str], list[Rgb]]:
    """Picture, guide and GraphDoc JSON (static); base colours at t=0 on wall_time (never static: run health recolours on
    wall_time; the runner logs them on the graph clock at its first source time). Returns the item ids and their base colours."""
    doc: GraphDoc = describe(graph, drawn)
    items, edges = picture(doc)
    root: str = GRAPH_ROOT + graph.name
    recording.log(GUIDE_ROOT + graph.name, rr.TextDocument(guide(doc, health=health), media_type=rr.MediaType.MARKDOWN), static=True)
    recording.log(f"{root}/describe", rr.TextDocument(to_json(doc)), static=True)
    if not items:  # an empty graph: nothing to draw
        return [], []
    nodes = rr.GraphNodes([i.id for i in items], labels=[i.label for i in items], positions=[(i.x, i.y) for i in items])
    recording.log(root, nodes, rr.GraphEdges(edges, graph_type="directed"), rr.AnyValues(does=[i.does for i in items], edges=[i.edges for i in items]), static=True)
    recording.set_time("wall_time", duration=0.0)
    recording.log(root, rr.GraphNodes.from_fields(colors=[i.color for i in items]))
    recording.reset_time()
    return [i.id for i in items], [i.color for i in items]
