"""One engine for both runners: one lock and one condition."""

import bisect
import heapq
import itertools
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

import numpy as np
import rerun as rr
from serde import serde

from dagkit.doc import GRAPH_ROOT, HEALTH_ROOT, Rgb, log_graph
from dagkit.graph import AsOf, Block, DropOldest, EdgeSpec, Graph, NodeSpec, Spawn, normalise
from dagkit.node import Control, Event, Fenced, Mode, RunContext, Tick
from dagkit.process import ProcessNode
from dagkit.sinks import RerunSink
from dagkit.time import STATIC, TIME_MAX, GraphError, TimeNs, check_monotonic

DEV_MODE: bool = os.environ.get("PIXI_DEV_MODE") == "1"
STEP_WINDOW: int = 4096  # step times a node keeps for the report's p95
Status: TypeAlias = Literal["complete", "stopped", "failed"]


@serde
@dataclass(frozen=True, slots=True)
class PortStats:
    depth_max: int
    drops: int
    pending_max: int


@serde
@dataclass(frozen=True, slots=True)
class NodeStats:
    ticks: int
    step_ms_mean: float
    step_ms_p95: float  # of the last STEP_WINDOW steps
    ports: dict[str, PortStats]


@serde
@dataclass(frozen=True, slots=True)
class RunReport:
    graph: str
    mode: str
    status: Status
    wall_s: float
    stream_s: float  # last minus first non-STATIC source time
    nodes: dict[str, NodeStats]


class _Abort(Exception):
    """Unwinds a worker after another worker failed."""


@dataclass(eq=False, slots=True)
class _Edge:
    spec: EdgeSpec
    unwrap: bool  # Fenced producer, host consumer: sync and unwrap at delivery
    wrap: bool  # host producer (a resumed checkpoint), fenced consumer: deliver Fenced(value, None)
    queue: deque = field(default_factory=deque)  # trigger edges: (seq, t_ns, value)
    history: list = field(default_factory=list)  # AsOf edges: (t_ns, value) in time order
    depth_max: int = 0
    drops: int = 0


@dataclass(eq=False, slots=True)
class _State:
    run: "_Run"
    spec: NodeSpec
    obj: object
    triggers: list[_Edge] = field(default_factory=list)
    asofs: list[_Edge] = field(default_factory=list)
    outs: dict[str, list[_Edge]] = field(default_factory=dict)
    pending: deque = field(default_factory=deque)  # (edge, t_ns, value) parked until AsOf watermarks reach t_ns
    statics: dict[str, object] = field(default_factory=dict)
    last_put: dict[str, TimeNs] = field(default_factory=dict)
    collected: dict[str, list[Event]] = field(default_factory=dict)
    events: Iterator[Event] | None = None  # a source's own iterator: closed with the run (a generator's with / finally)
    wm: TimeNs = STATIC - 1
    started: bool = False
    done: bool = False
    ticks: int = 0
    step_ms: deque = field(default_factory=lambda: deque(maxlen=STEP_WINDOW))  # the latest step times: a live run never grows
    step_ms_sum: float = 0.0
    logged: tuple[int, float] = (0, 0.0)  # ticks and step_ms_sum at the last health log
    pending_max: int = 0
    drops_seen: int = 0  # drops at the last health log

    def put(self, port: str, t_ns: TimeNs, value: object) -> None:
        run: _Run = self.run
        with run.cond:
            if run.failure is not None:
                raise _Abort  # another worker failed: no more outputs
            if port not in self.outs:
                raise GraphError(f"{self.spec.name}: no output port {port!r}")
            if DEV_MODE:
                fenced, classes = normalise(self.spec.outputs[port])
                inner: object = value.value if fenced and isinstance(value, Fenced) else value
                if fenced != isinstance(value, Fenced) or not isinstance(inner, classes):
                    raise GraphError(f"{self.spec.name}.{port}: {type(value).__name__} is not {self.spec.outputs[port]}")
            check_monotonic(self.last_put.get(port), t_ns, f"{self.spec.name}.{port}")
            if t_ns <= self.wm:
                raise GraphError(f"{self.spec.name}.{port}: put at {t_ns} after watermark {self.wm}")
            self.last_put[port] = t_ns
            for edge in self.outs[port]:
                kind = edge.spec.kind
                if isinstance(kind, AsOf):
                    edge.history.append((t_ns, value))
                    if run.mode == "realtime" and len(edge.history) > kind.keep:
                        del edge.history[0]
                    continue
                drop: bool = isinstance(kind, DropOldest) and run.mode == "realtime"
                while run.threaded and run.running and not drop and len(edge.queue) >= kind.capacity:
                    run.wait()
                if drop and len(edge.queue) >= kind.capacity:
                    edge.queue.popleft()
                    edge.drops += 1
                run.seq += 1
                edge.queue.append((run.seq, t_ns, value))
                edge.depth_max = max(edge.depth_max, len(edge.queue))
            run.cond.notify_all()

    def watermark(self, t_ns: TimeNs) -> None:
        with self.run.cond:
            if not getattr(self.obj, "emits_late", False) or t_ns < self.wm:
                raise GraphError(f"{self.spec.name}: watermark({t_ns}) needs emits_late and must not go back (now {self.wm})")
            self.wm = t_ns
            self.run.cond.notify_all()


class _Run:
    def __init__(self, graph: Graph, mode: Mode, control: Control | None, threaded: bool) -> None:
        graph.validate()
        self.graph, self.mode, self.threaded = graph, mode, threaded
        self.control: Control = control or Control()
        self.control.wake = self.notify
        self.cond: threading.Condition = threading.Condition(threading.RLock())
        self.failure: BaseException | None = None
        self.running = False
        self.seq = 0
        self.first, self.last = TIME_MAX, STATIC - 1  # first and last non-STATIC source time
        self.resources: dict[str, object] = {}
        self.threads: list[threading.Thread] = []
        self.states: dict[str, _State] = {}
        for name in graph.order():
            spec: NodeSpec = graph.nodes[name]
            obj: object = ProcessNode(spec.obj, spec.placement.cpus) if isinstance(spec.obj, Spawn) else spec.obj
            self.states[name] = _State(self, spec, obj, outs={p: [] for p in spec.outputs}, collected={p: [] for p in spec.inputs})
        for es in graph.edges:
            src_fenced: bool = normalise(graph.nodes[es.src.node].outputs[es.src.port])[0]
            dst_fenced: bool = normalise(graph.nodes[es.dst.node].inputs[es.dst.port])[0]
            edge = _Edge(es, unwrap=src_fenced and not dst_fenced, wrap=dst_fenced and not src_fenced)
            self.states[es.src.node].outs[es.src.port].append(edge)
            dst: _State = self.states[es.dst.node]
            (dst.asofs if isinstance(es.kind, AsOf) else dst.triggers).append(edge)
        self.sources: list[_State] = [s for s in self.states.values() if s.spec.kind == "source"]
        self.nodes: list[_State] = [s for s in self.states.values() if s.spec.kind != "source"]

    def notify(self) -> None:
        with self.cond:
            self.cond.notify_all()

    def fail(self, exc: BaseException) -> None:
        with self.cond:
            first: bool = self.failure is None
            self.failure = self.failure or exc
        self.notify()
        for s in (s for s in self.nodes if first and (s.started or s.spec.kind == "sink") and hasattr(s.obj, "abort")):
            self._guard(s.obj.abort, note=getattr(exc, "add_note", print))  # type: ignore[attr-defined]  # also wakes the caller-thread node

    def wait(self, timeout: float | None = None) -> None:
        if self.failure is None:
            self.cond.wait(timeout)
        if self.failure is not None:
            raise _Abort

    def _ready(self, s: _State, t_ns: TimeNs) -> bool:
        if s.spec.kind == "node" and len(s.triggers) > 1 and self.states[s.triggers[0].spec.src.node].wm < t_ns:
            return False  # several trigger ports (one producer): wait until nothing earlier can still arrive
        return self.mode == "realtime" or s.spec.kind == "segment" or all(self.states[e.spec.src.node].wm >= t_ns for e in s.asofs)

    def _next(self, s: _State) -> tuple[_Edge, TimeNs, object] | None:
        """The earliest (t_ns, port declaration) item, taken from its queue only when ready, so Block capacities hold the
        producer back. Queues spill into ``pending`` only while one is full and the head waits (a burst at one time, a late
        AsOf producer, a flush-only one): the producer may be waiting on that queue."""
        rank: dict[str, int] = {port: i for i, port in enumerate(s.spec.inputs)}
        heads: list[tuple[TimeNs, int, int, _Edge | None]] = [(e.queue[0][1], rank[e.spec.dst.port], 1, e) for e in s.triggers if e.queue]
        heads += [(s.pending[0][1], rank[s.pending[0][0].spec.dst.port], 0, None)] if s.pending else []  # pending first: it came first
        if not heads:
            return None
        t_ns, _, _, edge = min(heads, key=lambda head: head[:3])
        if self._ready(s, t_ns):
            if edge is None:
                return s.pending.popleft()
            self.cond.notify_all()
            return (edge, *edge.queue.popleft()[1:])
        if any(len(e.queue) >= cast(Block | DropOldest, e.spec.kind).capacity for e in s.triggers):
            s.pending.extend((e, t, v) for e in s.triggers for _, t, v in e.queue)
            for e in s.triggers:
                e.queue.clear()
            s.pending = deque(sorted(s.pending, key=lambda p: (p[1], rank[p[0].spec.dst.port])))
            s.pending_max = max(s.pending_max, len(s.pending))
            self.cond.notify_all()
        return None

    def _can_finish(self, s: _State) -> bool:
        if s.pending or any(e.queue or not self.states[e.spec.src.node].done for e in s.triggers):
            return False
        if s.spec.kind == "segment" and any(self.states[e.spec.src.node].wm < TIME_MAX for e in s.asofs):
            return False
        return s.spec.kind != "sink" or all(o.done for o in self.states.values() if o.spec.kind != "sink")

    def _convert(self, edge: _Edge, value: object) -> object:
        if edge.unwrap and isinstance(value, Fenced):
            if value.fence is not None:
                value.fence.synchronize()
            return value.value
        return Fenced(value, None) if edge.wrap and not isinstance(value, Fenced) else value

    def _deliver(self, s: _State, item: tuple[_Edge, TimeNs, object]) -> None:
        edge, t_ns, raw = item
        port: str = edge.spec.dst.port
        value: object = self._convert(edge, raw)
        with self.cond:
            if s.spec.kind == "segment":
                s.collected[port].append(Event(t_ns, port, value))
                return
            if not getattr(s.obj, "emits_late", False) and t_ns - 1 > s.wm:
                s.wm = t_ns - 1
                self.cond.notify_all()
            raw_asof: list[tuple[_Edge, object | None]] = []
            queued: list[TimeNs] = [p[1] for p in s.pending] + [e.queue[0][1] for e in s.triggers if e.queue]
            low: TimeNs = min([t_ns, *queued, *(self.states[e.spec.src.node].wm + 1 for e in s.triggers)])  # earliest tick still to come
            for e in s.asofs:
                found: int = bisect.bisect_right(e.history, t_ns, key=lambda h: h[0])
                raw_asof.append((e, e.history[found - 1][1] if found else None))
                del e.history[: max(bisect.bisect_right(e.history, low, key=lambda h: h[0]) - 1, 0)]  # retain low's predecessor
        asof: dict[str, object | None] = {e.spec.dst.port: None if v is None else self._convert(e, v) for e, v in raw_asof}
        if self.failure is not None:  # the commit point: no step begins after a failure (abort() may only meet a running one)
            raise _Abort
        start: float = time.perf_counter()
        s.obj.step(Tick(t_ns, port, value, asof), s)  # type: ignore[attr-defined]
        with self.cond:
            s.ticks += 1
            s.step_ms.append(ms := (time.perf_counter() - start) * 1e3)
            s.step_ms_sum += ms

    def _close(self, s: _State) -> None:
        with self.cond:
            s.wm, s.done = TIME_MAX, True
        self.notify()

    def _reset(self, s: _State) -> None:
        resources: dict[str, object] = {need: self.resources[need] for need in s.spec.needs}
        s.started = True  # a reset that fails half way still gets abort()
        s.obj.reset(RunContext(self.graph.name, s.spec.name, self.mode, resources, dict(s.statics), self.control))  # type: ignore[attr-defined]

    def _start_source(self, s: _State) -> Iterator[Event]:
        s.events = events = iter(s.obj.events())  # type: ignore[attr-defined]
        for event in events:
            if event.t_ns != STATIC:
                if self.health is not None:  # the picture's colours on the data timeline, from its first time (not 0: Unix-time data)
                    self.health.set_time(s.obj.clock, duration=np.timedelta64(event.t_ns, "ns"))  # type: ignore[attr-defined]
                    self.health.log(GRAPH_ROOT + self.graph.name, rr.GraphNodes.from_fields(colors=self.picture[1]))
                    self.health.reset_time()
                return itertools.chain([event], events)
            s.put(event.port, STATIC, event.value)
            for edge in (e for e in s.outs[event.port] if isinstance(e.spec.kind, AsOf)):
                self.states[edge.spec.dst.node].statics[edge.spec.dst.port] = self._convert(edge, event.value)
        return iter(())

    def _emit(self, s: _State, event: Event) -> None:
        with self.cond:
            s.wm, s.ticks = max(s.wm, event.t_ns - 1), s.ticks + 1
            self.first, self.last = min(self.first, event.t_ns), max(self.last, event.t_ns)
        s.put(event.port, event.t_ns, event.value)

    def _drain(self) -> None:
        while any([self._node_loop(s) for s in self.nodes]):  # visit every node, without short-circuiting
            pass

    def sequential(self, streams: list[Iterator[Event]]) -> None:
        heap: list[tuple[TimeNs, int, Event]] = [(e.t_ns, i, e) for i, it in enumerate(streams) if (e := next(it, None)) is not None]
        heapq.heapify(heap)
        while heap and not self.control.stopped:
            self._drain()
            _, i, event = heapq.heappop(heap)
            self._emit(self.sources[i], event)
            if (following := next(streams[i], None)) is not None:
                heapq.heappush(heap, (following.t_ns, i, following))
            else:
                self._close(self.sources[i])
            self._drain()
            self._health()
        for s in self.sources:
            self._close(s)
        self._drain()

    def _guard(self, body: Callable[..., object], *args: object, note: Callable[[str], object] | None = None) -> None:
        try:
            body(*args)
        except BaseException as exc:
            if note is not None:  # cleanup errors ride along on the original failure, which the caller re-raises
                note(f"{getattr(body, '__qualname__', body)} failed during cleanup: {exc!r}")
            elif not isinstance(exc, _Abort):  # worker failures reach the caller; _Abort only unwinds a failed run
                self.fail(exc)

    def _source_loop(self, s: _State, events: Iterator[Event]) -> None:
        for event in events:
            if self.control.stopped:
                break
            self._emit(s, event)
        self._close(s)

    def _node_loop(self, s: _State) -> bool:
        if self.threaded and s.spec.placement.cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, s.spec.placement.cpus)
        progress: bool = False
        while not s.done:
            with self.cond:
                while (item := self._next(s)) is None and not self._can_finish(s):
                    if not self.threaded:
                        return progress
                    self.wait()
                if self.failure is not None:
                    raise _Abort  # never step or flush after a failure
            if item is not None:
                self._deliver(s, item)
            else:
                history: dict[str, list[Event]] = {
                    e.spec.dst.port: [Event(t, e.spec.dst.port, self._convert(e, v)) for t, v in e.history] for e in s.asofs if s.spec.kind == "segment"
                }
                if self.failure is not None:  # a fence sync above may have outlived a failure
                    raise _Abort
                if s.spec.kind == "segment":
                    s.obj.run({**s.collected, **history}, s)  # type: ignore[attr-defined]
                else:
                    s.obj.flush(s)  # type: ignore[attr-defined]
                self._close(s)
            progress = True
        return progress

    def _coordinate(self) -> None:
        with self.cond:
            while not all(s.done for s in self.states.values()) and self.failure is None:
                self.cond.wait(self.health_every_s if self.health is not None else None)
                self._health()

    def threaded_body(self, streams: list[Iterator[Event]]) -> None:
        main: _State | None = next((s for s in self.nodes if s.spec.placement.where == "main"), None)
        workers: list[tuple[Callable[..., object], tuple]] = [(self._source_loop, (s, it)) for s, it in zip(self.sources, streams, strict=True)]
        workers += [(self._node_loop, (s,)) for s in self.nodes if s is not main] + [(self._coordinate, ())]
        self.threads = [threading.Thread(target=self._guard, args=(body, *args), daemon=True) for body, args in workers]
        for thread in self.threads:
            thread.start()
        if main is not None:
            self._guard(self._node_loop, main)
        self.threads[-1].join()
        if self.failure is not None:
            raise self.failure
        for thread in self.threads:
            thread.join()

    def _health(self, *, force: bool = False) -> None:
        now: float = time.perf_counter()
        if self.health is None or (not force and now - self.health_logged < self.health_every_s):
            return
        self.health_logged = now
        self.health.set_time("wall_time", duration=now - (self.started_at if force else self.health_started_at))
        for s in self.nodes:
            root: str = HEALTH_ROOT + s.spec.name
            if s.ticks > s.logged[0]:  # the mean step since the last log
                self.health.log(f"{root}/step_ms", rr.Scalars((s.step_ms_sum - s.logged[1]) / (s.ticks - s.logged[0])))
                s.logged = (s.ticks, s.step_ms_sum)
            for edge in s.triggers:
                self.health.log(f"{root}/queue_depth/{edge.spec.dst.port}", rr.Scalars(float(len(edge.queue))))
                self.health.log(f"{root}/drops/{edge.spec.dst.port}", rr.Scalars(float(edge.drops)))
        ids, colors = self.picture
        colors = [self._light(self.states[i]) if i in self.states and self.states[i].spec.kind != "source" else c for i, c in zip(ids, colors, strict=True)]
        self.health.log(GRAPH_ROOT + self.graph.name, rr.GraphNodes.from_fields(colors=colors))

    def _light(self, s: _State) -> Rgb:
        """grey: no input yet; red: drops since the last health log; yellow: a Block queue at least half full; else green."""
        drops: int = sum(e.drops for e in s.triggers)
        dropped, s.drops_seen = drops > s.drops_seen, drops
        if not (s.ticks or any(s.collected.values())):
            return (130, 130, 130)
        if dropped:
            return (255, 80, 80)
        return (240, 210, 60) if any(isinstance(e.spec.kind, Block) and 2 * len(e.queue) >= e.spec.kind.capacity for e in s.triggers) else (110, 220, 130)

    def execute(self, health: rr.RecordingStream | None, every_s: float) -> RunReport:
        self.started_at: float = time.perf_counter()
        self.health: rr.RecordingStream | None = health
        self.health_every_s: float = every_s
        try:
            self.picture: tuple[list[str], list[Rgb]] = log_graph(health, self.graph, health=True) if health is not None else ([], [])
            for name, factory in self.graph.resources.items():
                self.resources[name] = factory()
            streams: list[Iterator[Event]] = [self._start_source(s) for s in self.sources]
            for s in self.nodes:
                self._reset(s)
            self.running = True
            self.health_started_at: float = time.perf_counter()
            self.health_logged: float = self.health_started_at
            (self.threaded_body if self.threaded else self.sequential)(streams)
        except BaseException as exc:
            self.fail(exc)
            first: BaseException = self.failure or exc
            deadline: float = time.monotonic() + 10.0
            for thread in self.threads:
                thread.join(max(deadline - time.monotonic(), 0.0))
            if stuck := [t.name for t in self.threads if t.is_alive()]:
                first.add_note(f"threads {stuck} did not stop; resources leaked")
                raise first from None
            self._unwind(first)
            raise first from None
        if health is not None:  # the guide's "See it" links: what each viewer sink drew
            log_graph(health, self.graph, drawn={s.spec.name: s.obj.drawn for s in self.nodes if isinstance(s.obj, RerunSink) and s.obj.recording is health}, health=True)
        self._health(force=True)
        nodes: dict[str, NodeStats] = {}
        for name, s in self.states.items():
            ms: list[float] = sorted(s.step_ms) or [0.0]
            ports: dict[str, PortStats] = {e.spec.dst.port: PortStats(e.depth_max, e.drops, s.pending_max) for e in s.triggers}
            nodes[name] = NodeStats(s.ticks, s.step_ms_sum / s.ticks if s.step_ms else 0.0, ms[int(0.95 * (len(ms) - 1))], ports)
        if (error := self._unwind(None)) is not None:
            raise error
        wall_s: float = time.perf_counter() - self.started_at
        return RunReport(self.graph.name, self.mode, "stopped" if self.control.stopped else "complete", wall_s, max(self.last - self.first, 0) / 1e9, nodes)

    def _unwind(self, first: BaseException | None) -> BaseException | None:
        """The one cleanup of a finished or failed run: close every source iterator, source and process node, then the
        resources in reverse order, each although an earlier close failed; then break the state/control cycles, so
        resources free on return without cyclic GC, even while a UI keeps Control. Returns the run's failure, else the
        first close error; later close errors ride along on it as notes."""
        owned: list[object] = [s.events for s in self.sources] + [s.obj for s in self.states.values() if s.spec.kind == "source" or isinstance(s.obj, ProcessNode)]
        for close in (c for o in owned + list(reversed(self.resources.values())) if (c := getattr(o, "close", None)) is not None):
            try:
                close()
            except BaseException as exc:
                if first is None:
                    first = exc
                else:
                    first.add_note(f"{getattr(close, '__qualname__', close)} failed during cleanup: {exc!r}")
        self.control.wake = lambda: None
        self.states, self.sources, self.nodes, self.threads, self.resources = {}, [], [], [], {}
        return first


def run_sequential(graph: Graph, *, mode: Mode = "lossless", control: Control | None = None, health: rr.RecordingStream | None = None) -> RunReport:
    if mode != "lossless":
        raise GraphError("run_sequential is lossless only")
    return _Run(graph, mode, control, threaded=False).execute(health, 1.0)


def run_threaded(
    graph: Graph, *, mode: Mode = "realtime", control: Control | None = None, health: rr.RecordingStream | None = None, health_every_s: float = 1.0
) -> RunReport:
    return _Run(graph, mode, control, threaded=True).execute(health, health_every_s)
