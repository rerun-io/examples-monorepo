"""Runner semantics (SPEC §2.3, §2.3a, §2.5): both runners, unit lane, fakes only."""

import threading
import time
from collections import Counter
from collections.abc import Callable

import pytest
from dagkit_fakes import Add, Batch, Collect, Square, Sum, source
from hypothesis import given, settings
from hypothesis import strategies as st

from dagkit.graph import AsOf, Block, DropOldest, Graph, Placement, Spawn
from dagkit.node import Control, RunContext, Tick
from dagkit.runners import RunReport, run_sequential, run_threaded
from dagkit.testing import collect_edges
from dagkit.time import GraphError

RUNNERS: dict[str, Callable[..., RunReport]] = {
    "sequential": lambda g, **kw: run_sequential(g, **kw),
    "threaded": lambda g, **kw: run_threaded(g, mode="lossless", **kw),
}


def chain(events: list[tuple[int, str, int]], kind: Block | DropOldest = Block(1)) -> tuple[Graph, Collect]:
    graph = Graph("chain")
    src = graph.source("src", source(events))
    add = graph.node("add", Add(1))
    sink = Collect(("y",))
    out = graph.sink("sink", sink)
    graph.edge(src["x"], add["x"], kind)
    graph.edge(add["y"], out["y"])
    return graph, sink


@pytest.mark.parametrize("runner", RUNNERS)
def test_chain_delivers_every_value_in_order(runner: str) -> None:
    graph, sink = chain([(t, "x", t) for t in range(20)])
    report: RunReport = RUNNERS[runner](graph)
    assert sink.items == [("y", t, t + 1) for t in range(20)]
    assert report.status == "complete"
    assert report.nodes["add"].ticks == 20
    assert report.stream_s == pytest.approx(19e-9)


def diamond(events: list[tuple[int, str, int]], late: object) -> tuple[Graph, Collect]:
    """source -> late -> AsOf -> join; source -> join (the flush-only diamond when `late` emits only at flush)."""
    graph = Graph("diamond")
    src = graph.source("src", source(events))
    buffer = graph.node("late", late)  # type: ignore[arg-type]
    join = graph.node("join", Add(0, with_asof=True))
    sink = Collect(("y",))
    out = graph.sink("sink", sink)
    graph.edge(src["x"], buffer["x"], Block(1))
    graph.edge(src["x"], join["x"], Block(1))
    graph.edge(buffer["y"], join["a"], AsOf())
    graph.edge(join["y"], out["y"], Block(1))
    return graph, sink


@pytest.mark.parametrize("runner", RUNNERS)
def test_flush_only_diamond_terminates(runner: str) -> None:
    events = [(t, "x", t) for t in range(1, 30)]
    graph, sink = diamond(events, Batch(size=10_000))
    RUNNERS[runner](graph)
    # every join tick waited for the late producer's flush, so each sees its own value * 10 at the same time
    assert sink.items == [("y", t, t + 10 * t) for t in range(1, 30)]


@pytest.mark.parametrize("runner", RUNNERS)
def test_two_ticks_at_one_time_never_close_it_early(runner: str) -> None:
    graph, sink = diamond([(5, "x", 1), (5, "x", 2), (6, "x", 3)], Add(100))
    RUNNERS[runner](graph)
    # both t=5 ticks see the producer's output of the SECOND t=5 event (newest value at or before 5)
    assert sink.items == [("y", 5, 1 + 102), ("y", 5, 2 + 102), ("y", 6, 3 + 103)]


@pytest.mark.parametrize("runner", RUNNERS)
def test_asof_keeps_the_value_a_silent_producer_left(runner: str) -> None:
    events = [(1, "a", 7)] + [(t, "x", t) for t in range(2, 40)] + [(50, "a", 9), (51, "x", 51)]
    graph = Graph("silent")
    src = graph.source("src", source(events, ports=("a", "x")))
    join = graph.node("join", Add(0, with_asof=True))
    sink = Collect(("y",))
    graph.edge(src["x"], join["x"], Block(1))
    graph.edge(src["a"], join["a"], AsOf())
    graph.edge(join["y"], graph.sink("sink", sink)["y"])
    RUNNERS[runner](graph)
    assert sink.items == [("y", t, t + 7) for t in range(2, 40)] + [("y", 51, 60)]


@given(
    xs=st.lists(st.integers(0, 50), min_size=1, max_size=30),
    a_s=st.lists(st.tuples(st.integers(0, 50), st.integers(-100, 100)), max_size=30),
)
@settings(max_examples=60, deadline=None)
def test_asof_is_latest_at_or_before_tick(xs: list[int], a_s: list[tuple[int, int]]) -> None:
    events: list[tuple[int, str, int]] = sorted([(t, "x", 0) for t in xs] + [(t, "a", v) for t, v in a_s], key=lambda e: e[0])
    seen: list[tuple[int, object]] = []
    graph = Graph("asof")
    src = graph.source("src", source(events, ports=("a", "x")))
    sink = Collect(("x", "a"), on_step=lambda tick: seen.append((tick.t_ns, tick.asof["a"])))
    out = graph.sink("sink", sink)
    graph.edge(src["x"], out["x"], Block(2))
    graph.edge(src["a"], out["a"], AsOf())
    run_sequential(graph)
    expected: list[tuple[int, object]] = []
    for t, port, _ in events:
        if port == "x":
            older: list[tuple[int, str, int]] = [e for e in events if e[1] == "a" and e[0] <= t]
            expected.append((t, older[-1][2] if older else None))
    assert seen == expected


def random_dag(spec: list[tuple[int, int | None, int]], times: list[int]) -> Graph:
    """Node i consumes node `parent` (-1 = source) through Block(cap) and optionally node `asof` through AsOf.

    Node 1 is the emits_late buffer; one multi-producer sink takes every node's output.
    """
    graph = Graph("random")
    src = graph.source("src", source([(t, "x", i) for i, t in enumerate(times)]))
    refs = []
    for i, (parent, asof_choice, capacity) in enumerate(spec):
        asof: int | None = None if i == 1 else asof_choice
        node: object = Batch(size=3) if i == 1 else Add(i, with_asof=asof is not None)
        ref = graph.node(f"n{i}", node)  # type: ignore[arg-type]
        upstream = src["x"] if parent < 0 or parent >= i else refs[parent]["y"]
        graph.edge(upstream, ref["x"], Block(capacity))
        if asof is not None:
            graph.edge(src["x"] if asof >= i else refs[asof]["y"], ref["a"], AsOf())
        refs.append(ref)
    sink = graph.sink("sink", Collect(tuple(f"p{i}" for i in range(len(refs)))))
    for i, ref in enumerate(refs):
        graph.edge(ref["y"], sink[f"p{i}"], Block(1))
    return graph


@given(
    spec=st.lists(st.tuples(st.integers(-1, 5), st.none() | st.integers(0, 5), st.integers(1, 3)), min_size=2, max_size=6),
    deltas=st.lists(st.integers(0, 3), min_size=1, max_size=25),
)
@settings(max_examples=100, deadline=None)
def test_sequential_equals_threaded_on_block_and_asof_edges(spec: list[tuple[int, int | None, int]], deltas: list[int]) -> None:
    times: list[int] = [sum(deltas[: i + 1]) for i in range(len(deltas))]
    sequential: dict[str, list[object]] = collect_edges(random_dag(spec, times), run_sequential)
    threaded: dict[str, list[object]] = collect_edges(random_dag(spec, times), lambda g: run_threaded(g, mode="lossless"))
    for name in sequential:
        if name == "sink":
            assert Counter(map(repr, sequential[name])) == Counter(map(repr, threaded[name]))
        else:
            assert sequential[name] == threaded[name], name


def test_drop_oldest_keeps_the_latest_and_counts_drops() -> None:
    graph = Graph("realtime")
    src = graph.source("src", source([(t, "x", t) for t in range(200)]))
    slow = Collect(("x",), on_step=lambda tick: time.sleep(0.002))
    graph.edge(src["x"], graph.sink("slow", slow)["x"], DropOldest(1))
    report: RunReport = run_threaded(graph, mode="realtime")
    received: list[int] = [value for _, _, value in slow.items]
    assert received == sorted(received) and set(received) <= set(range(200))
    assert received[-1] == 199
    assert report.nodes["slow"].ports["x"].drops == 200 - len(received) > 0


@pytest.mark.parametrize("runner", RUNNERS)
def test_put_back_in_time_fails_the_run(runner: str) -> None:
    class Backwards(Add):
        def step(self, tick: Tick, out) -> None:  # noqa: ANN001
            out.put("y", 10 - tick.t_ns, 0)

    graph = Graph("backwards")
    src = graph.source("src", source([(1, "x", 1), (2, "x", 2)]))
    node = graph.node("node", Backwards(0))
    graph.edge(src["x"], node["x"])
    graph.edge(node["y"], graph.sink("sink", Collect(("y",)))["y"])
    with pytest.raises(GraphError, match="backwards"):
        RUNNERS[runner](graph)


@pytest.mark.parametrize("runner", RUNNERS)
def test_failure_aborts_without_flush_and_reraises(runner: str) -> None:
    class Boom(Add):
        aborted: bool = False

        def step(self, tick: Tick, out) -> None:  # noqa: ANN001
            if tick.t_ns == 3:
                raise KeyError("boom")
            super().step(tick, out)

        def abort(self) -> None:
            self.aborted = True

    boom = Boom(0)
    sink = Collect(("y",))
    graph = Graph("fail")
    src = graph.source("src", source([(t, "x", t) for t in range(10)]))
    node = graph.node("boom", boom)
    graph.edge(src["x"], node["x"])
    graph.edge(node["y"], graph.sink("sink", sink)["y"])
    with pytest.raises(KeyError, match="boom"):
        RUNNERS[runner](graph)
    assert boom.aborted and boom.flushes == 0
    assert sink.stopped_at_flush is None  # the sink never flushed


@pytest.mark.parametrize("runner", RUNNERS)
def test_stop_drains_flushes_and_reports_stopped(runner: str) -> None:
    control = Control()
    sink = Collect(("x",), on_step=lambda tick: control.stop() if tick.t_ns == 5 else None)
    graph = Graph("stop")
    graph.edge(graph.source("src", source([(t, "x", t) for t in range(1000)]))["x"], graph.sink("sink", sink)["x"], Block(1))
    report: RunReport = RUNNERS[runner](graph, control=control)
    assert report.status == "stopped"
    assert sink.stopped_at_flush is True
    assert len(sink.items) < 1000


@pytest.mark.parametrize("runner", RUNNERS)
def test_reset_barrier_bumps_world_once(runner: str) -> None:
    control = Control()
    add = Add(0)
    worlds: list[int] = []
    sink = Collect(("y",))
    graph = Graph("reset")
    src = graph.source("src", source([(t, "x", t) for t in range(10)]))
    node = graph.node("add", add)
    graph.edge(src["x"], node["x"], Block(1))
    graph.edge(node["y"], graph.sink("sink", sink)["y"], Block(1))
    fired = threading.Event()

    def maybe_reset(tick: Tick) -> None:
        if tick.t_ns == 4 and not fired.is_set():
            fired.set()
            control.reset(["add"])

    sink.on_step = lambda tick: (worlds.append(add.world), maybe_reset(tick))  # type: ignore[assignment]
    RUNNERS[runner](graph, control=control)
    assert add.resets == 2 and add.flushes == 2  # start + barrier; barrier + end
    assert worlds[0] == 0 and worlds[-1] == 1 and sorted(worlds) == worlds


@pytest.mark.parametrize("runner", RUNNERS)
def test_segment_node_runs_once_on_everything(runner: str) -> None:
    graph = Graph("segment")
    src = graph.source("src", source([(t, "x", t) for t in range(1, 11)]))
    total = graph.node("sum", Sum())
    sink = Collect(("y",))
    graph.edge(src["x"], total["x"], Block(1))
    graph.edge(total["y"], graph.sink("sink", sink)["y"])
    RUNNERS[runner](graph)
    assert sink.items == [("y", 10, 55)]


def test_process_placement_matches_thread() -> None:
    def build(process: bool) -> tuple[Graph, Collect]:
        graph = Graph("process")
        src = graph.source("src", source([(t, "x", t) for t in range(8)]))
        node = graph.node("sq", Spawn(Square, 1), placement=Placement("process")) if process else graph.node("sq", Square(1))
        sink = Collect(("y",))
        graph.edge(src["x"], node["x"], Block(2))
        graph.edge(node["y"], graph.sink("sink", sink)["y"])
        return graph, sink

    in_thread, thread_sink = build(False)
    in_process, process_sink = build(True)
    run_threaded(in_thread, mode="lossless")
    run_threaded(in_process, mode="lossless")
    assert process_sink.items == thread_sink.items == [("y", t, t * t + 1) for t in range(8)]


@pytest.mark.parametrize("runner", RUNNERS)
def test_reconfigure_applies_params_at_the_barrier(runner: str) -> None:
    control = Control()
    params: list[dict] = []

    class Configurable(Add):
        def reset(self, ctx: RunContext) -> None:
            super().reset(ctx)
            params.append(dict(ctx.params))

    graph = Graph("reconfigure")
    src = graph.source("src", source([(t, "x", t) for t in range(6)]))
    node = graph.node("add", Configurable(0))
    sink = Collect(("y",), on_step=lambda tick: control.reconfigure({"add": {"k": 5}}) if tick.t_ns == 2 else None)
    graph.edge(src["x"], node["x"], Block(1))
    graph.edge(node["y"], graph.sink("sink", sink)["y"], Block(1))
    RUNNERS[runner](graph, control=control)
    assert params == [{}, {"k": 5}]


class Model:
    """A graph resource that records its lifecycle."""

    closed: bool = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("runner", RUNNERS)
def test_resources_are_built_once_shared_and_closed(runner: str) -> None:
    built: list[Model] = []
    seen: list[object] = []

    class Uses(Add):
        def reset(self, ctx: RunContext) -> None:
            seen.append(ctx.resources["model"])

    graph = Graph("resources")
    graph.resource("model", lambda: built.append(Model()) or built[-1])
    src = graph.source("src", source([(1, "x", 1)]))
    a, b = graph.node("a", Uses(0), needs=["model"]), graph.node("b", Uses(0), needs=["model"])
    graph.edge(src["x"], a["x"])
    graph.edge(a["y"], b["x"])
    graph.edge(b["y"], graph.sink("sink", Collect(("y",)))["y"])
    RUNNERS[runner](graph)
    assert len(built) == 1 and seen == [built[0], built[0]] and built[0].closed
