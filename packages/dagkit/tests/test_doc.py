"""The graph view (dagkit.doc): the picture tells the truth about the declaration, its geometry never misleads, the guide
and GraphDoc hold together, the read-back check and the docs check catch what they promise. Unit lane: no viewer."""

import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import rerun as rr
from dagkit_fakes import Collect, Point
from hypothesis import given, settings
from hypothesis import strategies as st
from rerun.chunk import RrdReader
from serde.json import from_json, to_json

from dagkit.apis.synthetic_demo import Config, DemoRerunConfig, build
from dagkit.doc import GraphDoc, Item, describe, guide, picture
from dagkit.graph import AsOf, Block, DropOldest, EdgeSpec, Graph, NodeRef, PortKind, PortRef
from dagkit.node import Emit, Event, RunContext, Tick
from dagkit.runners import _Run, run_sequential
from dagkit.sources import IterSource
from dagkit.testing import check_graph_docs, check_picture, drawn_problems, picture_problems


class Step:
    """Add one to every value."""

    outputs: Mapping[str, type] = {"y": int}

    def __init__(self, asof: bool) -> None:
        self.inputs: dict[str, type] = {"x": int, "a": int} if asof else {"x": int}

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        out.put("y", tick.t_ns, int(tick.value) + 1)  # type: ignore[call-overload]

    def flush(self, out: Emit) -> None:
        pass


class Total:
    """Sum the whole segment."""

    outputs: Mapping[str, type] = {"y": int}

    def __init__(self, asof: bool) -> None:
        self.inputs: dict[str, type] = {"x": int, "a": int} if asof else {"x": int}

    def reset(self, ctx: RunContext) -> None:
        pass

    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None:
        if events := inputs["x"]:
            out.put("y", events[-1].t_ns, sum(int(e.value) for e in events))  # type: ignore[call-overload]


def numbers(times: Sequence[int]) -> IterSource:
    """A count (port x) and a side value (port s) at every time."""

    def events() -> Iterator[Event]:
        return (Event(t, port, i) for i, t in enumerate(times) for port in ("x", "s"))

    return IterSource({"x": int, "s": int}, events)


def random_graph(draw: Callable[[st.SearchStrategy[Any]], Any]) -> Graph:
    """Valid graphs: steps and segments with an optional AsOf side input, sinks fed by one or several producers. Drawn in
    the test body (``st.data()``): beartype samples containers with ``random``, which hypothesis forbids in strategies."""
    graph = Graph("random", about="A random valid graph.")
    graph.source("src", numbers([0, 1, 2]))
    producers: list[tuple[str, str]] = [("src", "x"), ("src", "s")]
    for i in range(draw(st.integers(1, 9))):
        kind: str = draw(st.sampled_from(["node", "node", "segment", "sink"]))
        if kind == "sink":
            feeds: list[tuple[str, str]] = draw(st.lists(st.sampled_from(producers), min_size=1, max_size=4))
            sink: NodeRef = graph.sink(f"out{i}", Collect(tuple(f"p{j}" for j in range(len(feeds)))))
            for j, (node, port) in enumerate(feeds):
                graph.edge(NodeRef(node)[port], sink[f"p{j}"], Block(draw(st.integers(1, 3))))
            continue
        asof: bool = draw(st.booleans())
        ref: NodeRef = graph.node(f"n{i}", Step(asof) if kind == "node" else Total(asof))  # type: ignore[arg-type]
        trigger: PortKind = draw(st.sampled_from([Block(2), DropOldest(1)])) if kind == "node" else Block(4)
        graph.edge(_ref(draw(st.sampled_from(producers))), ref["x"], trigger)
        if asof:
            graph.edge(_ref(draw(st.sampled_from(producers))), ref["a"], AsOf())
        producers.append((ref.name, "y"))
    return graph


def _ref(producer: tuple[str, str]) -> PortRef:
    return NodeRef(producer[0])[producer[1]]


def overlap(a: Item, b: Item) -> bool:
    ax0, ax1, ay0, ay1 = a.rect()
    bx0, bx1, by0, by1 = b.rect()
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def under(a: Item, b: Item, other: Item, steps: int = 400) -> bool:
    """Brute force, independent of the layout's clip: does the straight line a -> b pass over `other`'s rectangle?"""
    x0, x1, y0, y1 = other.rect()
    return any(x0 < a.x + f * (b.x - a.x) < x1 and y0 < a.y + f * (b.y - a.y) < y1 for f in (i / steps for i in range(1, steps)))


@given(data=st.data())
@settings(max_examples=150, deadline=None)
def test_the_picture_draws_the_declaration_and_never_misleads(data: st.DataObject) -> None:
    graph: Graph = random_graph(data.draw)
    doc: GraphDoc = describe(graph)
    items, lines = picture(doc)
    by_id: dict[str, Item] = {i.id: i for i in items}
    # structure: one box per node, every edge in exactly one label or tag, text and lines agree (the read-back check)
    assert drawn_problems(graph, [i.id for i in items], [i.label for i in items], lines, [i.edges for i in items]) == []
    assert len(by_id) == len(items)
    # numbers = reading order: boxes come first, numbered 1..n; every edge runs from a lower number to a higher one
    number: dict[str, int] = {n.name: n.number for n in doc.nodes}
    assert [i.id for i in items[: len(doc.nodes)]] == [n.name for n in doc.nodes] and sorted(number.values()) == list(range(1, len(number) + 1))
    assert all(number[e.src.node] < number[e.dst.node] for e in graph.edges)
    # geometry: no two items overlap; no line runs under a third item (the viewer draws lines under the boxes)
    assert not [(a.id, b.id) for k, a in enumerate(items) for b in items[k + 1 :] if overlap(a, b)]
    assert not [(a, b, o.id) for a, b in lines for o in items if o.id not in (a, b) and under(by_id[a], by_id[b], o)]
    # the guide's link texts are unique (the agent's only handle on a link) and GraphDoc survives JSON
    drawn = {n: {p: [f"/world/{n}/{p}/{k}" for k in range(3)] for p in spec.inputs} for n, spec in graph.nodes.items() if spec.kind == "sink"}
    texts: list[str] = re.findall(r"\[([^\]]+)\]\(", guide(describe(graph, drawn), health=True))
    assert [t for t, c in Counter(texts).items() if c > 1] == []
    assert from_json(GraphDoc, to_json(doc)) == doc


def demo_graph() -> Graph:
    return build(Config(DemoRerunConfig(headless=True)), rr.RecordingStream("dagkit-test"))


def test_the_structure_check_catches_a_lying_picture() -> None:
    graph: Graph = demo_graph()
    items, lines = picture(describe(graph))
    ids, labels, edges = [i.id for i in items], [i.label for i in items], [i.edges for i in items]
    assert drawn_problems(graph, ids, labels, lines, edges) == []
    tag: int = ids.index("smooth <-")
    assert any("drawn 0 times" in p for p in drawn_problems(graph, ids, labels, lines, [*edges[:tag], "", *edges[tag + 1 :]]))
    assert any("lacks" in p for p in drawn_problems(graph, ids, [*labels[:tag], "gain: float", *labels[tag + 1 :]], lines, edges))
    assert any("does not point at smooth" in p for p in drawn_problems(graph, ids, labels, [x for x in lines if x != ("smooth <-", "smooth")], edges))


def test_read_back_from_an_rrd_matches_and_a_changed_graph_does_not(tmp_path: Path) -> None:
    graph: Graph = demo_graph()
    rrd: Path = check_picture(graph, tmp_path)  # the helper every package calls in its unit lane
    first: EdgeSpec = graph.edges[0]
    graph.edges[0] = EdgeSpec(first.src, first.dst, Block(1))  # negative controls: after logging, change a kind ...
    assert any("drawn 0 times: helix.raw->smooth.raw waits" in p for p in picture_problems(graph, rrd))
    graph.edges[0] = first
    graph.edge(first.src, NodeRef("rerun")["smooth"])  # ... or add an edge
    assert any("drawn 0 times: helix.raw->rerun.smooth waits" in p for p in picture_problems(graph, rrd))
    assert picture_problems(Graph("other"), rrd) == [f"no picture of 'other' in {rrd}"]


def test_guide_lists_steps_types_and_links_into_the_recording() -> None:
    graph: Graph = demo_graph()
    text: str = guide(describe(graph, {"rerun": {"smooth": ["/world/smooth/point", "/signal/world/smooth"]}}), health=True)
    for expected in (
        "## 2 · smooth: Smooth the signal with a gain",
        "| **raw** | `Sample` | 1 helix | drops old (1) |",
        "| **gain** | `float` | 1 helix | latest |",
        "- [smooth.smooth → /world/smooth/point](recording://world/smooth/point)",
        "[box 2 smooth](recording://dagkit/graph/synthetic[#1])",
        "[smooth step ms](recording://dagkit/health/smooth/step_ms)",
        "| `Sample` | One point of the helix and its signal value |",
    ):
        assert expected in text, expected
    plain: str = guide(describe(graph), links=False)
    assert "](" not in plain and "recording://" not in plain and "#L" not in plain


class Undocumented:
    inputs: Mapping[str, type] = {"x": int}
    outputs: Mapping[str, type] = {"y": Point}

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        pass

    def flush(self, out: Emit) -> None:
        pass


class Wordy(Undocumented):
    """Turn every integer that arrives into a labelled point at the origin.

    The body says more.
    """


class Bare:
    pass


class Plain(Undocumented):
    """Make a point."""

    outputs: Mapping[str, type] = {"y": Bare}


def test_the_docs_check_names_every_gap_and_keeps_the_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph = Graph("gaps")
    src: NodeRef = graph.source("src", numbers([0]))
    for name, node in (("plain", Undocumented()), ("wordy", Wordy()), ("bare", Plain())):
        graph.edge(src["x"], graph.node(name, node)["x"])
    with pytest.raises(AssertionError) as caught:
        check_graph_docs(graph)
    message: str = str(caught.value)
    for expected in ("give Graph(..., about=...)", "Undocumented (", "has ''", "Wordy (", "has 'Turn every integer", "Bare (", "needs a docstring"):
        assert expected in message, expected
    assert "Point" not in message and "Plain (" not in message
    snapshot: Path = tmp_path / "GRAPH.md"
    check_graph_docs(demo_graph(), snapshot=snapshot)  # a documented graph passes; a missing snapshot is written
    assert snapshot.read_text().startswith("<!-- generated") and "## 2 · smooth: Smooth the signal with a gain" in snapshot.read_text()
    snapshot.write_text(snapshot.read_text().replace("Smooth the signal", "Blur the signal"))
    with pytest.raises(AssertionError, match="GRAPH.md is stale"):
        check_graph_docs(demo_graph(), snapshot=snapshot)
    monkeypatch.setenv("DAGKIT_UPDATE_SNAPSHOTS", "1")
    check_graph_docs(demo_graph(), snapshot=snapshot)
    assert "Smooth the signal" in snapshot.read_text()


def test_the_geometry_checks_have_teeth() -> None:
    items, _ = picture(describe(demo_graph()))
    by_id: dict[str, Item] = {i.id: i for i in items}
    label, rerun, helix, smooth = by_id["helix -> smooth.raw"], by_id["rerun"], by_id["helix"], by_id["smooth"]
    assert not overlap(label, rerun) and not under(helix, smooth, rerun)
    label.x, label.y = rerun.x, rerun.y
    assert overlap(label, rerun)
    rerun.x, rerun.y = (helix.x + smooth.x) / 2, (helix.y + smooth.y) / 2
    assert under(helix, smooth, rerun)


def test_run_health_recolours_the_boxes(tmp_path: Path) -> None:
    graph = Graph("health", about="Health colours.")
    src: NodeRef = graph.source("src", IterSource({"x": int, "s": int}, lambda: iter([Event(t, "x", t) for t in range(3)])))
    for name, port in (("busy", "x"), ("idle", "s")):
        graph.edge(src[port], graph.node(name, Step(False))["x"])
    recording = rr.RecordingStream("dagkit-test", recording_id="health", send_properties=False)
    recording.save(tmp_path / "health.rrd")
    run_sequential(graph, health=recording)
    recording.flush()
    recording.disconnect()
    reader = RrdReader(tmp_path / "health.rrd")
    rows: list[dict] = [row for c in reader.stream(store=reader.recordings()[0]).to_chunks() if str(c.entity_path) == "/dagkit/graph/health"
                        for row in c.to_record_batch().to_pylist() if row.get("GraphNodes:colors") and row.get("wall_time")]  # fmt: skip
    last: list[int] = max(rows, key=lambda row: row["wall_time"])["GraphNodes:colors"]
    items: list[Item] = picture(describe(graph))[0]
    colors: dict[str, int] = {item.id: color for item, color in zip(items, last, strict=True)}
    assert colors["busy"] == 0x6EDC82FF and colors["idle"] == 0x828282FF  # green: it ran; grey: no input ever came
    run = _Run(graph, "realtime", None, threaded=True)
    busy = run.states["busy"]
    busy.ticks, busy.triggers[0].drops = 1, 2
    assert run._light(busy) == (255, 80, 80)  # dropped since the last health log
    assert run._light(busy) == (110, 220, 130)  # ... and not again
    busy.triggers[0].queue.append((0, 0, 0))
    assert run._light(busy) == (110, 220, 130)  # one item in a Block(4) queue is fine ...
    busy.triggers[0].queue.append((1, 1, 1))
    assert run._light(busy) == (240, 210, 60)  # ... half full is backpressure


def test_the_demo_app_documents_its_graph(tmp_path: Path) -> None:
    """What every package does for its app graphs, built with fakes: picture == declaration, docs complete, GRAPH.md current."""
    check_picture(demo_graph(), tmp_path)
    check_graph_docs(demo_graph(), snapshot=Path(__file__).resolve().parents[1] / "GRAPH.md")
