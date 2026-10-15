"""Catalog reads: the rows a ``Type.from_layer`` reads, the Producer a layer carries, one rrd on an in-process server.

Nodes and Apps use these; Types, Ops and Models never query a catalog (``dagkit.types`` holds what they need).
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import rerun as rr
from rerun.catalog import DatasetEntry
from serde import SerdeError, from_dict

from dagkit.types import Producer, Type, static_row


def read_column(
    dataset: DatasetEntry, segment: str, value_type: type[Type], root: str, timeline: str, static: bool, window_ns: tuple[int, int] | None = None
) -> pa.Table:
    """The rows ``value_type.from_layer`` reads, only ``window_ns`` [start, stop) of a temporal column."""
    view = dataset.filter_segments([segment]).filter_contents(value_type.contents(root))
    reader = view.reader(index=None if static else timeline)
    if window_ns is not None and not static:
        from datafusion import col, lit  # the catalog reader's expression language (rerun-sdk's catalog extra)

        time = col(timeline).cast(pa.int64())
        reader = reader.filter((time >= lit(window_ns[0])) & (time < lit(window_ns[1])))
    return reader.to_arrow_table()


def read_producer(dataset: DatasetEntry, segment: str, layer: str) -> Producer | None:
    """The Producer ``log_producer`` wrote for ``layer`` into ``segment`` (None when it has none)."""
    entity: str = f"/producers/{layer}"
    row: dict[str, object] = static_row(dataset.filter_segments([segment]).filter_contents([entity]).reader(index=None).to_arrow_table(), entity)
    try:
        return from_dict(Producer, row) if row else None
    except SerdeError as exc:
        raise ValueError(f"{segment} {entity}: {exc}") from exc


@contextlib.contextmanager
def served(path: Path) -> Iterator[DatasetEntry]:
    """One rrd on an in-process catalog server: query the entry inside the ``with`` (an entry must outlive its queries)."""
    with rr.server.Server(host="127.0.0.1", datasets={"rrd": [str(path)]}) as server:
        yield server.client().get_dataset("rrd")
