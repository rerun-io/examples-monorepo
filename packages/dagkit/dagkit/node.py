"""The push-only node contract (SPEC §2.3), the control plane and fenced GPU values (§2.8)."""

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

from dagkit.time import TimeNs

Mode: TypeAlias = Literal["lossless", "realtime"]
PortType: TypeAlias = Any
"""A class, `Fenced[T]` or a union `A | B`; `dagkit.graph.normalise` reads it."""
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Event:
    """One item of a Source's single time-ordered stream."""

    t_ns: TimeNs
    """Time on the graph clock; `STATIC` for whole-stream values."""
    port: str
    """Output port the value leaves on."""
    value: object
    """The typed value."""


@dataclass(frozen=True, slots=True)
class Tick:
    """What `Node.step` receives: one trigger event plus the AsOf snapshot at its time."""

    t_ns: TimeNs
    """Time of the trigger event."""
    port: str
    """Input port the trigger arrived on."""
    value: object
    """The trigger value (unwrapped for host consumers of a `Fenced` port)."""
    asof: Mapping[str, object | None]
    """AsOf port -> newest value with time <= t_ns, or None."""


class Control:
    """Thread-safe control plane, separate from data edges: stop, reset barrier, reconfigure."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stopped: bool = False
        self._requests: list[tuple[frozenset[str], dict[str, dict[str, object]], bool]] = []
        self.wake: Callable[[], None] = lambda: None
        """Set by the runner: wakes its waits when a request arrives."""

    def stop(self) -> None:
        """Sources stop after their current event; queues drain, nodes flush, the run returns `stopped`."""
        with self._lock:
            self._stopped = True
        self.wake()

    def reset(self, nodes: Sequence[str] = ()) -> None:
        """At the reset barrier: flush, `world += 1`, `reset(ctx)` on `nodes` (every node if empty)."""
        with self._lock:
            self._requests.append((frozenset(nodes), {}, True))
        self.wake()

    def reconfigure(self, params: Mapping[str, Mapping[str, object]]) -> None:
        """Profile/preset switch: at the barrier the named nodes flush and reset with the new params."""
        with self._lock:
            self._requests.append((frozenset(params), {name: dict(p) for name, p in params.items()}, False))
        self.wake()

    @property
    def stopped(self) -> bool:
        """True once `stop` was called."""
        with self._lock:
            return self._stopped

    def take_requests(self) -> list[tuple[frozenset[str], dict[str, dict[str, object]], bool]]:
        """Runner side: pop pending (targets, params, bump_world) barrier requests."""
        with self._lock:
            requests, self._requests = self._requests, []
            return requests


@dataclass(frozen=True, slots=True)
class RunContext:
    """Everything a node may know about its run; rebuilt for every `reset`."""

    graph: str
    """Graph name."""
    node: str
    """This node's name."""
    mode: Mode
    """lossless (eval, layers) or realtime (live)."""
    resources: Mapping[str, object] = field(default_factory=dict)
    """Graph-owned objects this node declared in `needs`."""
    params: Mapping[str, object] = field(default_factory=dict)
    """Profile/preset params for this node, as applied by `Control.reconfigure`."""
    world: int = 0
    """Bumped by `Control.reset`; resettable estimators copy it onto their outputs."""
    statics: Mapping[str, object] = field(default_factory=dict)
    """AsOf port -> the Source's leading STATIC value on that edge (startup phase)."""
    control: Control | None = None
    """The run's control plane (None inside a process child); sinks read `control.stopped` at flush."""


@runtime_checkable
class Emit(Protocol):
    """Output side of a node, given to `step`, `flush` and `run`."""

    def put(self, port: str, t_ns: TimeNs, value: object) -> None:
        """Emit `value` on `port`; time is non-decreasing per port and may be older than the tick."""
        ...

    def watermark(self, t_ns: TimeNs) -> None:
        """Only for `emits_late` nodes: no more outputs with time <= t_ns."""
        ...


@runtime_checkable
class Node(Protocol):
    """Push-only node: `reset` before the first tick, one `step` per trigger event, `flush` at end of stream.

    Optional attributes, read with getattr: `emits_late: bool = False`, `abort(self) -> None` (failure cleanup).
    """

    @property
    def inputs(self) -> Mapping[str, PortType]: ...
    @property
    def outputs(self) -> Mapping[str, PortType]: ...
    def reset(self, ctx: RunContext) -> None: ...
    def step(self, tick: Tick, out: Emit) -> None: ...
    def flush(self, out: Emit) -> None: ...


@runtime_checkable
class SegmentNode(Protocol):
    """Whole-segment batch logic: `run` is called once, after every input closed."""

    @property
    def inputs(self) -> Mapping[str, PortType]: ...
    @property
    def outputs(self) -> Mapping[str, PortType]: ...
    def reset(self, ctx: RunContext) -> None: ...
    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None: ...


@runtime_checkable
class Source(Protocol):
    """The only pull point: STATIC events first, then non-decreasing `t_ns` on one stream."""

    @property
    def outputs(self) -> Mapping[str, PortType]: ...
    @property
    def clock(self) -> str: ...
    def events(self) -> Iterator[Event]: ...
    def close(self) -> None: ...


@runtime_checkable
class Fence(Protocol):
    """A recorded GPU event (`torch.cuda.Event` satisfies it); dagkit never imports torch."""

    def synchronize(self) -> None: ...
    def wait(self, stream: object = None) -> None: ...


@dataclass(frozen=True, slots=True)
class Fenced(Generic[T]):
    """A GPU value plus the event that marks it ready; immutable once put (fan-out safe)."""

    value: T
    """The device value (tensors stay on the device)."""
    fence: Fence | None
    """Recorded after the producer's last write; None for host-side or resumed values."""
