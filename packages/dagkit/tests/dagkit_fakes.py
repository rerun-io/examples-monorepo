"""Pure fake nodes for the dagkit unit lane (importable by a spawned child process)."""

import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import pyarrow as pa
import rerun as rr
from serde import serde

from dagkit.node import Emit, Event, RunContext, Tick
from dagkit.sources import IterSource
from dagkit.time import TimeNs
from dagkit.types import Log, row_times


def source(events: Sequence[tuple[TimeNs, str, int]], ports: Sequence[str] = ("x",), clock: str = "video_time") -> IterSource:
    """A source emitting `(t_ns, port, value)` in order."""

    def gen() -> Iterator[Event]:
        return (Event(t, port, value) for t, port, value in events)

    return IterSource(dict.fromkeys(ports, int), gen, clock=clock)


class Add:
    """y = x + k (+ the AsOf `a` value when that port is declared)."""

    outputs: Mapping[str, type] = {"y": int}

    def __init__(self, k: int, with_asof: bool = False) -> None:
        self.k: int = k
        self.inputs: dict[str, type] = {"x": int, "a": int} if with_asof else {"x": int}
        self.resets: int = 0
        self.flushes: int = 0

    def reset(self, ctx: RunContext) -> None:
        self.resets += 1
        self.world: int = ctx.world

    def step(self, tick: Tick, out: Emit) -> None:
        extra: object = tick.asof.get("a")
        out.put("y", tick.t_ns, int(tick.value) + self.k + (extra if isinstance(extra, int) else 0))  # type: ignore[call-overload]

    def flush(self, out: Emit) -> None:
        self.flushes += 1


class Batch:
    """emits_late buffer: holds inputs and emits them `size` at a time (the rest at flush)."""

    inputs: Mapping[str, type] = {"x": int}
    outputs: Mapping[str, type] = {"y": int}
    emits_late: bool = True

    def __init__(self, size: int) -> None:
        self.size: int = size
        self.held: list[tuple[TimeNs, int]] = []

    def reset(self, ctx: RunContext) -> None:
        self.held = []

    def step(self, tick: Tick, out: Emit) -> None:
        self.held.append((tick.t_ns, int(tick.value)))  # type: ignore[call-overload]
        if len(self.held) >= self.size:
            self._emit(out)
            out.watermark(tick.t_ns - 1)

    def flush(self, out: Emit) -> None:
        self._emit(out)

    def _emit(self, out: Emit) -> None:
        for t_ns, value in self.held:
            out.put("y", t_ns, value * 10)
        self.held = []


class Collect:
    """Sink recording (port, t_ns, value) of any number of int ports."""

    outputs: Mapping[str, type] = {}

    def __init__(self, ports: Sequence[str] = ("x",), on_step: Callable[[Tick], None] | None = None) -> None:
        self.inputs: dict[str, type] = dict.fromkeys(ports, int)
        self.items: list[tuple[str, TimeNs, int]] = []
        self.on_step: Callable[[Tick], None] | None = on_step
        self.stopped_at_flush: bool | None = None
        self.ctx: RunContext | None = None

    def reset(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def step(self, tick: Tick, out: Emit) -> None:
        self.items.append((tick.port, tick.t_ns, int(tick.value)))  # type: ignore[call-overload]
        if self.on_step is not None:
            self.on_step(tick)

    def flush(self, out: Emit) -> None:
        self.stopped_at_flush = self.ctx is not None and self.ctx.control is not None and self.ctx.control.stopped


class Sum:
    """SegmentNode: one output, the sum of all trigger values, at the last time."""

    inputs: Mapping[str, type] = {"x": int}
    outputs: Mapping[str, type] = {"y": int}

    def reset(self, ctx: RunContext) -> None:
        pass

    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None:
        events: Sequence[Event] = inputs["x"]
        if events:
            out.put("y", events[-1].t_ns, sum(int(e.value) for e in events))  # type: ignore[call-overload]


class Square:
    """Picklable pure node for process placement: y = x * x (config = unused offset)."""

    inputs: Mapping[str, type] = {"x": int}
    outputs: Mapping[str, type] = {"y": int}

    def __init__(self, config: int) -> None:
        self.offset: int = config

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        out.put("y", tick.t_ns, int(tick.value) ** 2 + self.offset)  # type: ignore[call-overload]

    def flush(self, out: Emit) -> None:
        pass


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Point:
    """A minimal host Type: one 3D point and a label, logged as Points3D."""

    xyz: tuple[float, float, float]
    label: str

    def to_rerun(self, root: str) -> Iterable[Log]:
        yield Log(f"{root}/point", rr.Points3D([self.xyz], labels=[self.label]))

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/point"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "Point"]]:
        positions: list = table[f"{root}/point:Points3D:positions"].to_pylist()
        labels: list = table[f"{root}/point:Points3D:labels"].to_pylist()
        for t_ns, xyz, label in zip(row_times(table, timeline), positions, labels, strict=True):
            yield t_ns, cls((float(xyz[0][0]), float(xyz[0][1]), float(xyz[0][2])), label[0])


def verify_run(case: object, phase: str, recording: rr.RecordingStream) -> None:
    """Fake `pkg.verify:run` for dagkit.verify tests: logs one point."""
    recording.log("/out", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)


def verify_score(case: object, golden: object) -> dict[str, float]:
    """Fake `pkg.verify:score`: a constant metric."""
    return {"ate": 1.0}


class Affinity(Square):
    """Process-placement probe: y = the CPUs this process may run on, as a bit mask."""

    def step(self, tick: Tick, out: Emit) -> None:
        out.put("y", tick.t_ns, sum(1 << cpu for cpu in os.sched_getaffinity(0)))
