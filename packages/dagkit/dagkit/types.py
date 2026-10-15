import functools
import importlib.metadata
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import rerun as rr
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


@functools.cache
def git(*args: str) -> str:
    """``git <args>`` in dagkit's checkout ('' on a host without git or outside a checkout, e.g. a deployed device)."""
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent)
    except OSError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Producer:
    package: str
    version: str  # package version + git sha7
    graph: str
    profile: str  # hardware class / preset the run used
    phase: str = "run"  # `run`, or a verify phase (`pre`, `pre2`, `post`)
    params_sha: str = ""  # sha256[:12] of the resolved node params + SegmentRef window; resume compares it

    @classmethod
    def of(cls, package: str, graph: str, profile: str, *, phase: str = "run", params_sha: str = "") -> "Producer":
        """Stamped with the installed ``package`` version and the checkout's git sha7 (none without a checkout)."""
        sha: str = git("rev-parse", "--short=7", "HEAD")
        version: str = importlib.metadata.version(package)
        return cls(package, f"{version}+{sha}" if sha else version, graph, profile, phase, params_sha)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LayerFile:
    """A finished layer file (an rrd) that a sink publishes."""

    path: Path


def log_producer(recording: rr.RecordingStream, layer: str, producer: Producer) -> None:
    recording.log(f"/producers/{layer}", rr.AnyValues(**to_dict(producer)), static=True)


def log_value(recording: rr.RecordingStream, timeline: str, t_ns: TimeNs, value: Type, root: str) -> list[str]:
    if t_ns != STATIC:
        recording.set_time(timeline, duration=np.timedelta64(t_ns, "ns"))
    logs: list[Log] = list(value.to_rerun(root))
    for log in logs:
        recording.log(log.path, log.data, static=log.static or t_ns == STATIC)
    return [log.path for log in logs]


def row_times(table: pa.Table, timeline: str) -> list[TimeNs]:
    return table[timeline].cast(pa.int64()).to_pylist() if timeline in table.column_names else [STATIC] * table.num_rows


def static_row(table: pa.Table, entity: str) -> dict[str, object]:
    """``entity``'s components in the first row of ``table`` (a static read) by component name, each cell taken out of
    Rerun's per-row list when it holds one value (a logged array stays a list); {} without a row."""
    prefix: str = f"{entity}:"
    cells: dict[str, object] = {c.removeprefix(prefix): table.column(c)[0].as_py() for c in table.column_names if c.startswith(prefix)} if table.num_rows else {}
    return {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in cells.items() if v is not None}


def column(table: pa.Table, name: str) -> list:
    """One column's cells as Python values, one per row (``None`` in every row when the table lacks it)."""
    return table[name].to_pylist() if name in table.column_names else [None] * table.num_rows


def cells(table: pa.Table, name: str, timeline: str) -> dict[TimeNs, pa.Scalar]:
    """The non-null cells of one column keyed by their time (STATIC for a table read without an index)."""
    if name not in table.column_names:
        return {}
    return {t: cell for t, cell in zip(row_times(table, timeline), table[name], strict=True) if cell.is_valid}


def flat(values: pa.Scalar | pa.Array, dtype: type) -> np.ndarray:
    """A cell's (or a column's non-null) component values as ``dtype``, unwrapping Rerun's nesting: one list per row plus
    the component's arity (``Transform3D:translation`` is ``list<fixed_size_list<3>>``)."""
    array = values.values if isinstance(values, pa.Scalar) else values.drop_null()
    while pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or pa.types.is_fixed_size_list(array.type):
        array = array.flatten()
    return array.to_numpy(zero_copy_only=False).astype(dtype)
