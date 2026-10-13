import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

from dagkit.time import TimeNs

Mode: TypeAlias = Literal["lossless", "realtime"]  # lossless: eval and layers; realtime: live devices
PortType: TypeAlias = Any  # a class, `Fenced[T]` or a union `A | B`; `dagkit.graph.normalise` reads it
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Event:
    t_ns: TimeNs
    port: str
    value: object


@dataclass(frozen=True, slots=True)
class Tick:
    t_ns: TimeNs
    port: str
    value: object  # synced and unwrapped for host consumers of a `Fenced` port
    asof: Mapping[str, object | None]  # AsOf port -> newest value with time <= t_ns, or None


class Control:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stopped: threading.Event = threading.Event()
        self._requests: list[tuple[frozenset[str], dict[str, dict[str, object]], bool]] = []
        self.wake: Callable[[], None] = lambda: None  # set by the runner: wakes its waits on a request

    def stop(self) -> None:
        self._stopped.set()
        self.wake()

    def reset(self, nodes: Sequence[str] = ()) -> None:
        self._request(frozenset(nodes), {}, True)

    def reconfigure(self, params: Mapping[str, Mapping[str, object]]) -> None:
        self._request(frozenset(params), {name: dict(p) for name, p in params.items()}, False)

    def _request(self, targets: frozenset[str], params: dict[str, dict[str, object]], bump: bool) -> None:
        with self._lock:
            self._requests.append((targets, params, bump))
        self.wake()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def take_requests(self) -> list[tuple[frozenset[str], dict[str, dict[str, object]], bool]]:
        with self._lock:
            requests, self._requests = self._requests, []
            return requests


@dataclass(frozen=True, slots=True)
class RunContext:
    graph: str
    node: str
    mode: Mode
    resources: Mapping[str, object] = field(default_factory=dict)  # graph-owned objects this node declared in `needs`
    params: Mapping[str, object] = field(default_factory=dict)  # profile/preset params, as applied by `Control.reconfigure`
    world: int = 0  # bumped by `Control.reset`; resettable estimators copy it onto their outputs
    statics: Mapping[str, object] = field(default_factory=dict)  # AsOf port -> the Source's leading STATIC value (startup)
    control: Control | None = None  # None inside a process child; sinks read `control.stopped` at flush


@runtime_checkable
class Emit(Protocol):
    def put(self, port: str, t_ns: TimeNs, value: object) -> None: ...  # time non-decreasing per port; may be < the tick
    def watermark(self, t_ns: TimeNs) -> None: ...  # `emits_late` nodes only: no more outputs with time <= t_ns


@runtime_checkable
class Node(Protocol):  # optional: `emits_late`; `abort()` on failure, which may meet a step begun before the failure
    @property
    def inputs(self) -> Mapping[str, PortType]: ...
    @property
    def outputs(self) -> Mapping[str, PortType]: ...
    def reset(self, ctx: RunContext) -> None: ...
    def step(self, tick: Tick, out: Emit) -> None: ...
    def flush(self, out: Emit) -> None: ...


@runtime_checkable
class SegmentNode(Protocol):
    @property
    def inputs(self) -> Mapping[str, PortType]: ...
    @property
    def outputs(self) -> Mapping[str, PortType]: ...
    def reset(self, ctx: RunContext) -> None: ...
    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None: ...


@runtime_checkable
class Source(Protocol):
    @property
    def outputs(self) -> Mapping[str, PortType]: ...
    @property
    def clock(self) -> str: ...
    def events(self) -> Iterator[Event]: ...
    def close(self) -> None: ...


@runtime_checkable
class Fence(Protocol):
    def synchronize(self) -> None: ...
    def wait(self, stream: object = None) -> None: ...


@dataclass(frozen=True, slots=True)
class Fenced(Generic[T]):
    value: T
    fence: Fence | None  # None for host-side or resumed values
