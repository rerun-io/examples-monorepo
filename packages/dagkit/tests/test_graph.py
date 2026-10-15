"""Graph.validate rejects every listed problem (SPEC §2.4) and normalises Fenced/union port types."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from dagkit_fakes import Add, Collect, Square, Sum, source

from dagkit.graph import AsOf, Block, DropOldest, Graph, Placement, Spawn, normalise
from dagkit.node import Fenced
from dagkit.sinks import LayerSink
from dagkit.sources import SegmentRef
from dagkit.time import GraphError


class Base:
    pass


class Derived(Base):
    pass


class Typed:
    """Node with configurable port types."""

    def __init__(self, inputs: Mapping[str, object], outputs: Mapping[str, object]) -> None:
        self.inputs = dict(inputs)
        self.outputs = dict(outputs)

    def reset(self, ctx: object) -> None: ...
    def step(self, tick: object, out: object) -> None: ...
    def flush(self, out: object) -> None: ...


def problems(graph: Graph) -> str:
    with pytest.raises(GraphError) as info:
        graph.validate()
    return str(info.value)


def test_normalise_fenced_and_unions() -> None:
    assert normalise(Derived) == (False, (Derived,))
    assert normalise(Fenced[Derived]) == (True, (Derived,))
    assert normalise(Base | Derived) == (False, (Base, Derived))
    assert normalise(Fenced[Base | Derived]) == (True, (Base, Derived))


def test_valid_graph_passes_and_orders_topologically() -> None:
    graph = Graph("ok")
    src = graph.source("src", source([]))
    add = graph.node("add", Add(1))
    graph.edge(src["x"], add["x"])
    graph.edge(add["y"], graph.sink("sink", Collect(("y",)))["y"])
    graph.validate()
    assert graph.order() == ["src", "add", "sink"]


def test_port_types_subclass_union_and_fenced_unwrap() -> None:
    graph = Graph("types")
    producer = graph.node("p", Typed({}, {"a": Fenced[Derived], "b": Derived, "c": Base}))
    consumer = graph.node("c", Typed({"a": Base, "b": Base | int, "c": Derived}, {}))
    graph.edge(producer["a"], consumer["a"])  # host consumer takes T from Fenced[T]
    graph.edge(producer["b"], consumer["b"])  # subclass into a union member
    graph.edge(producer["c"], consumer["c"])  # Base is not a Derived
    report: str = problems(graph)
    assert "p.c -> c.c" in report and "p.a -> c.a" not in report and "p.b -> c.b" not in report


def test_validate_lists_every_problem(tmp_path: Path) -> None:
    graph = Graph("bad")
    src = graph.source("src", source([]))
    other = graph.source("other", source([], clock="device_time"))
    graph.source("src", source([]))  # duplicate name
    a = graph.node("a", Add(1, with_asof=True))
    b = graph.node("b", Add(2))
    graph.edge(src["x"], a["x"])
    graph.edge(other["x"], a["a"], Block())  # second producer on a trigger port of a non-sink
    graph.edge(src["x"], b["x"])
    graph.edge(src["x"], b["x"])  # connected twice
    graph.edge(src["nope"], b["x"])  # unknown port
    seg = graph.node("seg", Sum())
    graph.edge(src["x"], seg["x"], DropOldest())  # DropOldest into a SegmentNode
    layer = LayerSink.__new__(LayerSink)
    layer.inputs, layer.outputs = {"y": int}, {}
    graph.edge(b["y"], graph.sink("layer", layer)["y"], DropOldest())  # DropOldest into a LayerSink
    graph.node("spawn_thread", Spawn(Square, 0))  # Spawn without process placement
    graph.node("proc", Square(0), placement=Placement("process"))  # process without Spawn
    graph.node("needs", Add(0), needs=["model"])  # unknown resource
    graph.node("fenced", Spawn(Typed, None), placement=Placement("process"))
    graph.sink("emits", Typed({}, {"z": int}))  # a sink with outputs
    graph.node("m1", Add(0), placement=Placement("main"))
    graph.node("m2", Add(0), placement=Placement("main"))
    report: str = problems(graph)
    for expected in [
        "duplicate name 'src'",
        "a: trigger ports fed by several producers",
        "b.x: input connected 2 times",
        "src.nope -> b.x: unknown node or port",
        "seg.x: DropOldest into a SegmentNode",
        "layer.y: DropOldest into a LayerSink",
        "spawn_thread: Spawn recipe iff placement 'process'",
        "proc: Spawn recipe iff placement 'process'",
        "needs: unknown resource 'model'",
        "emits: a sink has no outputs",
        "sources on mixed clocks",
        "more than one node placed on 'main'",
        "m1.x: input connected 0 times",
    ]:
        assert expected in report, expected


def test_process_node_ports_must_be_host_types() -> None:
    class FencedNode(Square):
        outputs = {"y": Fenced[int]}

    graph = Graph("proc")
    graph.node("p", Spawn(FencedNode, 0), placement=Placement("process"))
    assert "p: a process-placed node takes picklable host Types, not Fenced" in problems(graph)


def test_cycles_are_rejected() -> None:
    graph = Graph("cycle")
    a = graph.node("a", Add(1))
    b = graph.node("b", Add(2))
    graph.edge(a["y"], b["x"])
    graph.edge(b["y"], a["x"], AsOf())
    assert "cycle through ['a', 'b']" in problems(graph)


def test_sinks_feed_each_input_from_the_port_named_like_it() -> None:
    graph = Graph("sinks")
    src = graph.source("src", source([], ports=("x", "a")))
    layer = LayerSink.__new__(LayerSink)
    layer.inputs, layer.outputs, layer.segment = {"x": int}, {}, SegmentRef("d", "s")
    graph.sinks({"x": src["x"], "a": src["a"]}, {"layer": layer, "both": Collect(("a", "x"))})
    graph.validate()
    edges = [(e.src.port, e.dst.node, e.dst.port, e.kind) for e in graph.edges]
    assert edges == [("x", "layer", "x", Block()), ("a", "both", "a", Block()), ("x", "both", "x", Block())]


def test_capacities_and_retention_must_be_positive() -> None:
    graph = Graph("capacity")
    src = graph.source("src", source([], ports=("x", "a")))
    node = graph.node("n", Add(0, with_asof=True))
    graph.edge(src["x"], node["x"], Block(0))
    graph.edge(src["a"], node["a"], AsOf(keep=0))
    report: str = problems(graph)
    assert "n.x: Block(capacity=0) needs capacity / keep >= 1" in report and "n.a: AsOf(keep=0)" in report
