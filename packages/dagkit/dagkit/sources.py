import heapq
import itertools
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
    dataset: str  # catalog dataset name
    segment: str  # segment id
    catalog_url: str = "rerun+http://127.0.0.1:51235"  # catalog server
    timeline: str = "video_time"  # the graph clock
    window_ns: tuple[int, int] | None = None  # short clips: [start, stop) on the timeline. Windowed runs never register a layer

    def entry(self) -> DatasetEntry:
        return CatalogClient(self.catalog_url).get_dataset(self.dataset)


@dataclass(frozen=True, slots=True)
class Column:
    type: type[Type]  # value type; its `from_layer` reads the table
    root: str  # entity root (identifies a checkpoint, since catalog reads select entities, not layers)
    static: bool = False  # read without an index; values come back at STATIC


def merge_events(*streams: Iterator[Event]) -> Iterator[Event]:
    last: list[TimeNs | None] = [None] * len(streams)
    indexed: list[Iterator[tuple[int, Event]]] = [zip(itertools.repeat(i), stream) for i, stream in enumerate(streams)]
    for i, event in heapq.merge(*indexed, key=lambda pair: pair[1].t_ns):
        check_monotonic(last[i], event.t_ns, f"stream {i}")
        last[i] = event.t_ns
        yield event


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
    return [*statics, *merge_events(*streams)]


class IterSource:
    """Hand in values from the App."""

    def __init__(self, outputs: Mapping[str, PortType], events: Callable[[], Iterator[Event]], *, clock: str = "video_time") -> None:
        self.outputs: dict[str, PortType] = dict(outputs)
        self.clock: str = clock
        self._events: Callable[[], Iterator[Event]] = events

    def events(self) -> Iterator[Event]:
        return self._events()

    def close(self) -> None:
        pass


class CatalogSegmentSource(IterSource):
    """Read stored layers from the catalog."""

    def __init__(self, segment: SegmentRef, ports: Mapping[str, Column]) -> None:
        def events() -> Iterator[Event]:
            dataset: DatasetEntry = segment.entry()  # the entry must outlive its queries
            return iter(_column_events(dataset, segment.segment, ports, segment.timeline, segment.window_ns))

        super().__init__({port: column.type for port, column in ports.items()}, events, clock=segment.timeline)


class RecordingSource(IterSource):
    """Read stored values from an rrd file."""

    def __init__(self, path: Path, ports: Mapping[str, Column], timeline: str = "video_time") -> None:
        def events() -> Iterator[Event]:
            with rr.server.Server(host="127.0.0.1", datasets={"recording": [str(path)]}) as server:
                dataset: DatasetEntry = server.client().get_dataset("recording")
                return iter(_column_events(dataset, dataset.segment_ids()[0], ports, timeline, None))

        super().__init__({port: column.type for port, column in ports.items()}, events, clock=timeline)
