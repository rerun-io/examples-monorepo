"""The base contract of every edge value: a Rerun mapping both ways, plus the producer record (SPEC §2.2)."""

import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
import pyarrow as pa
import rerun as rr
from rerun.catalog import DatasetEntry
from serde import serde, to_dict

from dagkit.time import STATIC, TimeNs


@dataclass(frozen=True, slots=True)
class Log:
    """One entry of a Type's Rerun mapping."""

    path: str  # absolute entity path (root + suffix)
    data: rr.AsComponents
    static: bool = False


@runtime_checkable
class Type(Protocol):
    """Every edge value. Host Types are also `@serde` frozen dataclasses (jaxtyping arrays with explicit dtype)."""

    def to_rerun(self, root: str) -> Iterable[Log]: ...
    @classmethod
    def contents(cls, root: str) -> list[str]: ...  # entity filter for `from_layer` reads
    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "Type"]]: ...  # exactly what to_rerun wrote


T = TypeVar("T", bound=Type)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Producer:
    """Who wrote a layer; logged as a STATIC entity, never as recording properties."""

    package: str
    version: str  # package version + git sha7
    graph: str
    profile: str  # hardware class / preset the run used
    phase: str = "run"  # `run`, or a verify phase (`pre`, `pre2`, `post`)
    params_sha: str = ""  # sha256[:12] of the resolved node params + SegmentRef window; resume compares it


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LayerFile:
    """A finished layer rrd written outside dagkit (e.g. by a Rust core)."""

    path: Path


def log_producer(recording: rr.RecordingStream, layer: str, producer: Producer) -> None:
    """Log `producer` as static `AnyValues` at `/producers/<layer>`."""
    recording.log(f"/producers/{layer}", rr.AnyValues(**to_dict(producer)), static=True)


def row_times(table: pa.Table, timeline: str) -> list[TimeNs]:
    """Index times (ns) of a reader table; a static read (no timeline column) gives STATIC for each row."""
    if timeline not in table.column_names:
        return [STATIC] * table.num_rows
    return table[timeline].cast(pa.int64()).to_pylist()


def read_column(dataset: DatasetEntry, segment: str, value_type: type[Type], root: str, timeline: str, static: bool) -> pa.Table:
    """Reader table of one Type under `root` in one segment (static columns are read without an index)."""
    view = dataset.filter_segments([segment]).filter_contents(value_type.contents(root))
    return view.reader(index=None if static else timeline).to_arrow_table()


def rerun_roundtrip(value: T, root: str, timeline: str, t_ns: TimeNs) -> list[tuple[TimeNs, T]]:
    """Write `value` to a temp rrd, serve it in-process and read it back (a catalog: integration lane only)."""
    static: bool = t_ns == STATIC
    with tempfile.TemporaryDirectory() as tmp:
        path: Path = Path(tmp) / "roundtrip.rrd"
        recording = rr.RecordingStream("dagkit-roundtrip", recording_id="roundtrip", send_properties=False)
        recording.save(path)
        if not static:
            recording.set_time(timeline, duration=np.timedelta64(t_ns, "ns"))
        for log in value.to_rerun(root):
            recording.log(log.path, log.data, static=log.static or static)
        recording.flush()
        recording.disconnect()
        with rr.server.Server(host="127.0.0.1", datasets={"roundtrip": [str(path)]}) as server:
            table: pa.Table = read_column(server.client().get_dataset("roundtrip"), "roundtrip", type(value), root, timeline, static)
    return list(type(value).from_layer(table, root, timeline))  # type: ignore[return-value]
