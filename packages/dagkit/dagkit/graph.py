"""Declared graphs: port kinds, port-type normalisation, placement and validation (SPEC §2.4)."""

import types
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, Union, get_args, get_origin

from dagkit.node import Fenced, Node, PortType, SegmentNode, Source
from dagkit.sinks import LayerSink
from dagkit.time import GraphError


@dataclass(frozen=True, slots=True)
class Block:
    """Lossless edge: the producer blocks while the queue is full."""

    capacity: int = 4
    """Queue bound (ignored by the sequential runner)."""


@dataclass(frozen=True, slots=True)
class DropOldest:
    """Realtime edge: when full, drop and count the oldest waiting item. Lossless mode turns it into Block."""

    capacity: int = 1
    """Queue bound."""


@dataclass(frozen=True, slots=True)
class AsOf:
    """Join edge: the consumer reads the newest value at or before its tick time; it never triggers a step."""

    wait_ns: int | None = None
    """Realtime only: how long to wait for the producer watermark (None or 0: never). Lossless always waits."""
    keep: int = 128
    """Realtime only: history bound."""


PortKind: TypeAlias = Block | DropOldest | AsOf
NodeKind: TypeAlias = Literal["source", "node", "segment", "sink"]


@dataclass(frozen=True, slots=True)
class Placement:
    """Where the threaded runner runs a node."""

    where: Literal["main", "thread", "process"] = "thread"
    """`main` = the caller thread (at most one node), `process` needs a `Spawn` recipe."""
    cpus: tuple[int, ...] | None = None
    """CPU affinity of the node's thread (Linux)."""


@dataclass(frozen=True, slots=True)
class Spawn:
    """Picklable recipe for a process-placed node: the child builds `cls(config)`; ports are class attributes."""

    cls: type
    """Node class (importable by module path)."""
    config: object
    """Picklable constructor argument."""


@dataclass(frozen=True, slots=True)
class PortRef:
    """One port of one node."""

    node: str
    """Node name."""
    port: str
    """Port name."""


@dataclass(frozen=True, slots=True)
class NodeRef:
    """Handle returned by `Graph.source/node/sink`; `ref["port"]` names a port."""

    name: str
    """Node name."""

    def __getitem__(self, port: str) -> PortRef:
        return PortRef(self.name, port)


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """A declared node: the object, its kind, placement and resource needs."""

    name: str
    obj: object
    kind: NodeKind
    placement: Placement
    needs: tuple[str, ...]

    @property
    def inputs(self) -> Mapping[str, PortType]:
        owner: object = self.obj.cls if isinstance(self.obj, Spawn) else self.obj
        return getattr(owner, "inputs", {})

    @property
    def outputs(self) -> Mapping[str, PortType]:
        owner: object = self.obj.cls if isinstance(self.obj, Spawn) else self.obj
        return getattr(owner, "outputs", {})


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """A typed edge between two ports."""

    src: PortRef
    dst: PortRef
    kind: PortKind


def normalise(port_type: PortType) -> tuple[bool, tuple[type, ...]]:
    """`Fenced[T]` -> (True, members of T); `A | B` -> (False, (A, B)); a class -> (False, (cls,))."""
    if port_type is Fenced:
        return True, (object,)
    fenced: bool = get_origin(port_type) is Fenced
    inner: PortType = get_args(port_type)[0] if fenced else port_type
    members: tuple = get_args(inner) if get_origin(inner) in (Union, types.UnionType) else (inner,)
    classes: tuple[type, ...] = tuple(get_origin(m) or m for m in members)
    return fenced, tuple(c if isinstance(c, type) else object for c in classes)


class Graph:
    """Which nodes, typed edges, port kinds, resources and placement. No math, no models, no I/O."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.resources: dict[str, Callable[[], object]] = {}
        self.nodes: dict[str, NodeSpec] = {}
        self.edges: list[EdgeSpec] = []
        self._problems: list[str] = []

    def _add(self, spec: NodeSpec) -> NodeRef:
        if spec.name in self.nodes or spec.name in self.resources:
            self._problems.append(f"duplicate name {spec.name!r}")
        else:
            self.nodes[spec.name] = spec
        return NodeRef(spec.name)

    def resource(self, name: str, factory: Callable[[], object]) -> str:
        """Declare a graph-owned object, built once by the runner (declaration order) and closed after the run."""
        if name in self.resources or name in self.nodes:
            self._problems.append(f"duplicate name {name!r}")
        self.resources[name] = factory
        return name

    def source(self, name: str, source: Source) -> NodeRef:
        return self._add(NodeSpec(name, source, "source", Placement(), ()))

    def node(self, name: str, node: Node | SegmentNode | Spawn, *, placement: Placement = Placement(), needs: Sequence[str] = ()) -> NodeRef:
        kind: NodeKind = "segment" if hasattr(node, "run") and not hasattr(node, "step") else "node"
        return self._add(NodeSpec(name, node, kind, placement, tuple(needs)))

    def sink(self, name: str, sink: Node) -> NodeRef:
        """A Node with `outputs == {}`; it may take triggers from several producers."""
        return self._add(NodeSpec(name, sink, "sink", Placement(), ()))

    def edge(self, src: PortRef, dst: PortRef, kind: PortKind = Block()) -> None:
        self.edges.append(EdgeSpec(src, dst, kind))

    def checkpoint(self, src: PortRef, sink: LayerSink) -> None:
        """Write any edge to a catalog layer: adds the sink and a Block edge into its port of the same name."""
        ref: NodeRef = self.sink(f"{src.node}_{src.port}_layer", sink)
        self.edge(src, ref[src.port if src.port in sink.inputs else next(iter(sink.inputs))])

    def order(self) -> list[str]:
        """Topological order of node names; raises `GraphError` on a cycle."""
        indegree: dict[str, int] = dict.fromkeys(self.nodes, 0)
        successors: dict[str, list[str]] = {name: [] for name in self.nodes}
        for edge in self.edges:
            if edge.src.node in self.nodes and edge.dst.node in self.nodes:
                indegree[edge.dst.node] += 1
                successors[edge.src.node].append(edge.dst.node)
        ready: list[str] = [name for name in self.nodes if indegree[name] == 0]
        ordered: list[str] = []
        while ready:
            name: str = ready.pop(0)
            ordered.append(name)
            for successor in successors[name]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if len(ordered) != len(self.nodes):
            raise GraphError(f"cycle through {sorted(set(self.nodes) - set(ordered))}")
        return ordered

    def validate(self) -> None:
        """Raise `GraphError` listing EVERY problem of the declared graph."""
        problems: list[str] = list(self._problems)
        incoming: dict[PortRef, list[EdgeSpec]] = defaultdict(list)
        for edge in self.edges:
            src: NodeSpec | None = self.nodes.get(edge.src.node)
            dst: NodeSpec | None = self.nodes.get(edge.dst.node)
            if src is None or dst is None or edge.src.port not in src.outputs or edge.dst.port not in dst.inputs:
                problems.append(f"edge {edge.src.node}.{edge.src.port} -> {edge.dst.node}.{edge.dst.port}: unknown node or port")
                continue
            incoming[edge.dst].append(edge)
            src_types: tuple[type, ...] = normalise(src.outputs[edge.src.port])[1]
            dst_types: tuple[type, ...] = normalise(dst.inputs[edge.dst.port])[1]
            if not all(any(issubclass(s, d) for d in dst_types) for s in src_types):
                problems.append(f"edge {edge.src.node}.{edge.src.port} -> {edge.dst.node}.{edge.dst.port}: {src_types} is not {dst_types}")
            if isinstance(edge.kind, DropOldest) and (dst.kind == "segment" or isinstance(dst.obj, LayerSink)):
                problems.append(f"{edge.dst.node}.{edge.dst.port}: DropOldest into a {'SegmentNode' if dst.kind == 'segment' else 'LayerSink'}")
        for name, spec in self.nodes.items():
            if spec.kind == "sink" and spec.outputs:
                problems.append(f"{name}: a sink has no outputs")
            for port in spec.inputs:
                if (count := len(incoming[PortRef(name, port)])) != 1:
                    problems.append(f"{name}.{port}: input connected {count} times (needs exactly 1)")
            producers: set[str] = {e.src.node for port in spec.inputs for e in incoming[PortRef(name, port)] if not isinstance(e.kind, AsOf)}
            if spec.kind == "node" and len(producers) > 1:
                problems.append(f"{name}: trigger ports fed by several producers {sorted(producers)} (join through AsOf)")
            if isinstance(spec.obj, Spawn) != (spec.placement.where == "process"):
                problems.append(f"{name}: Spawn recipe iff placement 'process'")
            if spec.placement.where == "process" and any(normalise(t)[0] for t in [*spec.inputs.values(), *spec.outputs.values()]):
                problems.append(f"{name}: a process-placed node takes picklable host Types, not Fenced")
            problems += [f"{name}: unknown resource {need!r}" for need in spec.needs if need not in self.resources]
        clocks: set[str] = {getattr(spec.obj, "clock", "") for spec in self.nodes.values() if spec.kind == "source"}
        if len(clocks) > 1:
            problems.append(f"sources on mixed clocks {sorted(clocks)} (one clock per graph)")
        if sum(spec.placement.where == "main" for spec in self.nodes.values()) > 1:
            problems.append("more than one node placed on 'main'")
        try:
            self.order()
        except GraphError as exc:
            problems.append(str(exc))
        if problems:
            raise GraphError(f"graph {self.name!r}:\n  " + "\n  ".join(problems))
