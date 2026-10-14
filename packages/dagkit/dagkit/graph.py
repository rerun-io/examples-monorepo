import graphlib
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
    capacity: int = 4  # ignored by the sequential runner


@dataclass(frozen=True, slots=True)
class DropOldest:
    capacity: int = 1


@dataclass(frozen=True, slots=True)
class AsOf:
    wait_ns: int | None = None  # realtime only: how long to wait for the producer watermark (None or 0: never). Lossless always waits
    keep: int = 128  # realtime only: history bound


PortKind: TypeAlias = Block | DropOldest | AsOf
NodeKind: TypeAlias = Literal["source", "node", "segment", "sink"]


@dataclass(frozen=True, slots=True)
class Placement:
    where: Literal["main", "thread", "process"] = "thread"  # `main` = the caller thread (at most one node); `process` needs a `Spawn` recipe
    cpus: tuple[int, ...] | None = None  # thread affinity (Linux)


@dataclass(frozen=True, slots=True)
class Spawn:
    cls: type
    config: object


@dataclass(frozen=True, slots=True)
class PortRef:
    node: str
    port: str


@dataclass(frozen=True, slots=True)
class NodeRef:
    name: str

    def __getitem__(self, port: str) -> PortRef:
        return PortRef(self.name, port)


@dataclass(frozen=True, slots=True)
class NodeSpec:
    name: str
    obj: object
    kind: NodeKind
    placement: Placement
    needs: tuple[str, ...]

    @property
    def inputs(self) -> Mapping[str, PortType]:
        return getattr(self.obj.cls if isinstance(self.obj, Spawn) else self.obj, "inputs", {})

    @property
    def outputs(self) -> Mapping[str, PortType]:
        return getattr(self.obj.cls if isinstance(self.obj, Spawn) else self.obj, "outputs", {})


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    src: PortRef
    dst: PortRef
    kind: PortKind


def normalise(port_type: PortType) -> tuple[bool, tuple[type, ...]]:
    if port_type is Fenced:
        return True, (object,)
    fenced: bool = get_origin(port_type) is Fenced
    inner: PortType = get_args(port_type)[0] if fenced else port_type
    members: tuple = get_args(inner) if get_origin(inner) in (Union, types.UnionType) else (inner,)
    classes: tuple[type, ...] = tuple(get_origin(m) or m for m in members)
    return fenced, tuple(c if isinstance(c, type) else object for c in classes)


class Graph:
    def __init__(self, name: str, about: str = "") -> None:
        self.name: str = name
        self.about: str = about  # one or two sentences: what the graph turns into what (the guide's opening, `dagkit.doc`)
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
        return self._add(NodeSpec(name, sink, "sink", Placement(), ()))

    def edge(self, src: PortRef, dst: PortRef, kind: PortKind = Block()) -> None:
        self.edges.append(EdgeSpec(src, dst, kind))

    def sinks(self, ports: Mapping[str, PortRef], sinks: Mapping[str, Node]) -> None:
        """Add every sink and feed each of its inputs from the producer port named like it."""
        for name, sink in sinks.items():
            ref: NodeRef = self.sink(name, sink)
            for port in sink.inputs:
                self.edge(ports[port], ref[port])

    def order(self) -> list[str]:
        sorter: graphlib.TopologicalSorter[str] = graphlib.TopologicalSorter({name: set() for name in self.nodes})
        for edge in (e for e in self.edges if e.src.node in self.nodes and e.dst.node in self.nodes):
            sorter.add(edge.dst.node, edge.src.node)
        try:
            return list(sorter.static_order())
        except graphlib.CycleError as exc:
            raise GraphError(f"cycle through {sorted(set(exc.args[1]))}") from exc

    def validate(self) -> None:
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
            if min(getattr(edge.kind, "capacity", 1), getattr(edge.kind, "keep", 1)) < 1:
                problems.append(f"{edge.dst.node}.{edge.dst.port}: {edge.kind} needs capacity / keep >= 1")
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
