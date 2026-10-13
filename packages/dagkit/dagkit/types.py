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
    path: str  # absolute entity path (root + suffix)
    data: rr.AsComponents
    static: bool = False


@runtime_checkable
class Type(Protocol):
    def to_rerun(self, root: str) -> Iterable[Log]: ...
    @classmethod
    def contents(cls, root: str) -> list[str]: ...  # entity filter for `from_layer` reads
    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "Type"]]: ...  # exactly what to_rerun wrote


T = TypeVar("T", bound=Type)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Producer:
    package: str
    version: str  # package version + git sha7
    graph: str
    profile: str  # hardware class / preset the run used
    phase: str = "run"  # `run`, or a verify phase (`pre`, `pre2`, `post`)
    params_sha: str = ""  # sha256[:12] of the resolved node params + SegmentRef window; resume compares it


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LayerFile:
    path: Path


def log_producer(recording: rr.RecordingStream, layer: str, producer: Producer) -> None:
    recording.log(f"/producers/{layer}", rr.AnyValues(**to_dict(producer)), static=True)


def log_value(recording: rr.RecordingStream, timeline: str, t_ns: TimeNs, value: Type, root: str) -> None:
    if t_ns != STATIC:
        recording.set_time(timeline, duration=np.timedelta64(t_ns, "ns"))
    for log in value.to_rerun(root):
        recording.log(log.path, log.data, static=log.static or t_ns == STATIC)


def row_times(table: pa.Table, timeline: str) -> list[TimeNs]:
    return table[timeline].cast(pa.int64()).to_pylist() if timeline in table.column_names else [STATIC] * table.num_rows


def read_column(dataset: DatasetEntry, segment: str, value_type: type[Type], root: str, timeline: str, static: bool) -> pa.Table:
    view = dataset.filter_segments([segment]).filter_contents(value_type.contents(root))
    return view.reader(index=None if static else timeline).to_arrow_table()


def rerun_roundtrip(value: T, root: str, timeline: str, t_ns: TimeNs) -> list[tuple[TimeNs, T]]:
    static: bool = t_ns == STATIC
    with tempfile.TemporaryDirectory() as tmp:
        path: Path = Path(tmp) / "roundtrip.rrd"
        recording = rr.RecordingStream("dagkit-roundtrip", recording_id="roundtrip", send_properties=False)
        recording.save(path)
        log_value(recording, timeline, t_ns, value, root)
        recording.flush()
        recording.disconnect()
        with rr.server.Server(host="127.0.0.1", datasets={"roundtrip": [str(path)]}) as server:
            table: pa.Table = read_column(server.client().get_dataset("roundtrip"), "roundtrip", type(value), root, timeline, static)
    return list(type(value).from_layer(table, root, timeline))  # type: ignore[return-value]
