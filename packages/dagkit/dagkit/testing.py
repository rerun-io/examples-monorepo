"""Test helpers: the node contract on fakes, and the ticks each node of a graph receives."""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from dagkit.graph import Graph, Spawn
from dagkit.node import Emit, Event, Node, PortType, RunContext, Tick
from dagkit.time import TimeNs, check_monotonic

Output = tuple[str, TimeNs, object]


class Recorder:
    """An `Emit` that keeps every put in order."""

    def __init__(self) -> None:
        self.items: list[Output] = []

    def put(self, port: str, t_ns: TimeNs, value: object) -> None:
        self.items.append((port, t_ns, value))

    def watermark(self, t_ns: TimeNs) -> None:
        pass


def check_node_contract(node: Node, ticks: Sequence[Tick], ctx: RunContext | None = None) -> list[Output]:
    """Assert the push-only contract on `node`; return the outputs of one run.

    Checks: outputs go to declared ports with non-decreasing time per port; a second flush emits nothing; a run
    after a double `reset` repeats the first run exactly (reset is idempotent and rebuilds all state).
    """
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
    assert runs[0] == runs[1], "a run after reset differs from the first run (reset is not idempotent)"
    return runs[0]


class _Spy:
    """Forwards the node contract and records every tick (or segment input) the node receives."""

    def __init__(self, node: object, seen: list[object]) -> None:
        self.node: object = node
        self.seen: list[object] = seen
        self.inputs: Mapping[str, PortType] = getattr(node, "inputs", {})
        self.outputs: Mapping[str, PortType] = getattr(node, "outputs", {})
        self.emits_late: bool = getattr(node, "emits_late", False)

    def reset(self, ctx: RunContext) -> None:
        self.node.reset(ctx)  # type: ignore[attr-defined]

    def step(self, tick: Tick, out: Emit) -> None:
        self.seen.append((tick.port, tick.t_ns, tick.value, dict(tick.asof)))
        self.node.step(tick, out)  # type: ignore[attr-defined]

    def flush(self, out: Emit) -> None:
        self.node.flush(out)  # type: ignore[attr-defined]

    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None:
        self.seen.append({port: list(events) for port, events in inputs.items()})
        self.node.run(inputs, out)  # type: ignore[attr-defined]


def collect_edges(graph: Graph, run: Callable[[Graph], object]) -> dict[str, list[object]]:
    """Run `graph` with every node wrapped in a spy; return node -> received (port, t_ns, value, asof) in order."""
    seen: dict[str, list[object]] = defaultdict(list)
    spied = Graph(graph.name)
    spied.resources, spied.edges = dict(graph.resources), list(graph.edges)
    for name, spec in graph.nodes.items():
        keep: bool = spec.kind == "source" or isinstance(spec.obj, Spawn)
        spied.nodes[name] = spec if keep else replace(spec, obj=_Spy(spec.obj, seen[name]))
    run(spied)
    return dict(seen)
