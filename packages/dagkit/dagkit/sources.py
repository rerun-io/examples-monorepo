"""Sources: the only pull points; each emits ONE time-ordered stream (SPEC §2.6)."""

import heapq
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from rerun.catalog import CatalogClient, DatasetEntry
from serde import serde

from dagkit.node import Event, PortType
from dagkit.time import TimeNs, check_monotonic
from dagkit.types import Type, read_column


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SegmentRef:
    """One segment of a catalog dataset, optionally a short window of it."""

    dataset: str
    """Catalog dataset name."""
    segment: str
    """Segment id."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog server."""
    timeline: str = "video_time"
    """The graph clock."""
    window_ns: tuple[int, int] | None = None
    """Short clips: [start, stop) on the timeline. Windowed runs never register a layer."""

    def entry(self) -> DatasetEntry:
        return CatalogClient(self.catalog_url).get_dataset(self.dataset)


@dataclass(frozen=True, slots=True)
class Column:
    """One typed port read from a layer: the Type and the entity root that identifies it."""

    type: type[Type]
    """Value type; its `from_layer` reads the table."""
    root: str
    """Entity root (identifies a checkpoint, since catalog reads select entities, not layers)."""
    static: bool = False
    """Read without an index; values come back at STATIC."""


def merge_events(*streams: Iterator[Event]) -> Iterator[Event]:
    """Heap merge by (t_ns, stream index); raises `GraphError` on a non-monotonic stream."""
    heap: list[tuple[TimeNs, int, Event]] = [(e.t_ns, i, e) for i, s in enumerate(streams) if (e := next(s, None)) is not None]
    heapq.heapify(heap)
    last: list[TimeNs | None] = [None] * len(streams)
    while heap:
        t_ns, i, event = heapq.heappop(heap)
        check_monotonic(last[i], t_ns, f"stream {i}")
        last[i] = t_ns
        yield event
        if (following := next(streams[i], None)) is not None:
            heapq.heappush(heap, (following.t_ns, i, following))


def _column_events(dataset: DatasetEntry, segment: str, ports: Mapping[str, Column], timeline: str, window_ns: tuple[int, int] | None) -> list[Event]:
    statics: list[Event] = []
    streams: list[Iterator[Event]] = []
    for port, column in ports.items():
        table: pa.Table = read_column(dataset, segment, column.type, column.root, timeline, column.static)
        if window_ns is not None and not column.static:
            times: np.ndarray = table[timeline].cast(pa.int64()).to_numpy()  # int64 ns
            table = table.filter(pa.array((times >= window_ns[0]) & (times < window_ns[1])))
        events: list[Event] = [Event(t_ns, port, value) for t_ns, value in column.type.from_layer(table, column.root, timeline)]
        if column.static:
            statics += events
        else:
            streams.append(iter(events))
    return statics + list(merge_events(*streams))


class CatalogSegmentSource:
    """Typed non-video columns of one catalog segment (poses, GT, prior layers, checkpoints)."""

    def __init__(self, segment: SegmentRef, ports: Mapping[str, Column]) -> None:
        self.segment: SegmentRef = segment
        self.ports: dict[str, Column] = dict(ports)
        self.outputs: dict[str, PortType] = {port: column.type for port, column in ports.items()}
        self.clock: str = segment.timeline

    def events(self) -> Iterator[Event]:
        dataset: DatasetEntry = self.segment.entry()
        yield from _column_events(dataset, self.segment.segment, self.ports, self.clock, self.segment.window_ns)

    def close(self) -> None:
        pass


class RecordingSource:
    """A local .rrd read through an in-process server: same reader, same columns as the catalog."""

    def __init__(self, path: Path, ports: Mapping[str, Column], timeline: str = "video_time") -> None:
        self.path: Path = path
        self.ports: dict[str, Column] = dict(ports)
        self.outputs: dict[str, PortType] = {port: column.type for port, column in ports.items()}
        self.clock: str = timeline

    def events(self) -> Iterator[Event]:
        with rr.server.Server(host="127.0.0.1", datasets={"recording": [str(self.path)]}) as server:
            dataset: DatasetEntry = server.client().get_dataset("recording")
            events: list[Event] = _column_events(dataset, dataset.segment_ids()[0], self.ports, self.clock, None)
        yield from events

    def close(self) -> None:
        pass


class IterSource:
    """Files and live devices: any generator of Events (follow live sources with DropOldest edges)."""

    def __init__(self, outputs: Mapping[str, PortType], events: Callable[[], Iterator[Event]], *, clock: str = "video_time") -> None:
        self.outputs: dict[str, PortType] = dict(outputs)
        self.clock: str = clock
        self._events: Callable[[], Iterator[Event]] = events

    def events(self) -> Iterator[Event]:
        return self._events()

    def close(self) -> None:
        pass
