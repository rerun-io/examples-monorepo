from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
import rerun as rr
from numpy import ndarray

T = TypeVar("T")  # simplecv still supports Python 3.10, so no PEP 695 type parameters here.


def _normalize_content_expr(expr: str) -> str:
    if expr.startswith("/"):
        return expr
    if expr.startswith("-/"):
        return expr
    if expr.startswith("-"):
        return f"-/{expr[1:].lstrip('/')}"
    return f"/{expr.lstrip('/')}"


def _normalize_selectors(selectors: Sequence[str], *, index: str | None = None) -> list[str]:
    normalized: list[str] = []
    for selector in selectors:
        if selector == index or selector.startswith("/") or ":" not in selector:
            normalized.append(selector)
        else:
            normalized.append(f"/{selector.lstrip('/')}")
    return normalized


def unwrap_singleton_lists(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value


def first_valid_value(
    column: pa.ChunkedArray | pa.Array,
    *,
    allow_none: bool = False,
    component_name: str | None = None,
) -> Any:
    values = column.combine_chunks().to_pylist() if isinstance(column, pa.ChunkedArray) else column.to_pylist()
    for value in values:
        if value is None:
            continue
        return unwrap_singleton_lists(value)
    if allow_none:
        return None
    column_name = component_name or "(unknown component)"
    raise ValueError(f"Expected at least one non-null value in column '{column_name}'")


def first_valid_value_as(column: pa.ChunkedArray | pa.Array, kind: type[T], *, component_name: str | None = None) -> T:
    """:func:`first_valid_value` narrowed to ``kind``, so callers get a precise type instead of ``Any``.

    Raises:
        TypeError: If the first non-null value is not an instance of ``kind``.
    """
    value = first_valid_value(column, component_name=component_name)
    if not isinstance(value, kind):
        raise TypeError(f"column '{component_name or '(unknown component)'}' holds {type(value).__name__}, expected {kind.__name__}")
    return value


def series_to_int64_ns(series: pd.Series) -> ndarray:
    values = series.to_numpy()
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[ns]").astype(np.int64)
    return np.asarray(values, dtype=np.int64)


@dataclass(slots=True)
class RRDQuerySession:
    """Lazily hosts a single RRD in the OSS server for catalog queries."""

    rrd_path: Path | str
    _server: Any | None = field(default=None, init=False, repr=False)
    _dataset: Any | None = field(default=None, init=False, repr=False)
    _dataset_name: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        resolved = Path(self.rrd_path).expanduser().resolve()
        self.rrd_path = resolved
        digest = hashlib.sha1(str(resolved).encode(), usedforsecurity=False).hexdigest()[:12]
        self._dataset_name = f"simplecv_rrd_{digest}"

    def close(self) -> None:
        if self._server is None:
            return
        close = getattr(self._server, "close", None)
        if callable(close):
            close()
        self._server = None
        self._dataset = None

    def _dataset_entry(self) -> Any:
        if self._dataset is None:
            self._server = rr.server.Server(datasets={self._dataset_name: [str(self.rrd_path)]})
            self._dataset = self._server.client().get_dataset(self._dataset_name)
        return self._dataset

    def _dataset_view(self, contents: str | Sequence[str]) -> Any:
        exprs = [contents] if isinstance(contents, str) else list(contents)
        normalized_exprs = [_normalize_content_expr(expr) for expr in exprs]
        return self._dataset_entry().filter_contents(normalized_exprs)

    def read_arrow(
        self,
        *,
        contents: str | Sequence[str],
        selectors: Sequence[str],
        index: str | None,
        allow_missing: bool = False,
    ) -> pa.Table:
        dataset_view = self._dataset_view(contents)
        projection = _normalize_selectors(selectors, index=index)
        if index is not None:
            projection = [index, *projection]
        available_columns = set(dataset_view.arrow_schema().names)
        missing = [column for column in projection if column not in available_columns]
        if missing:
            if allow_missing:
                return pa.table({})
            raise ValueError(f"Missing columns for query on {self.rrd_path.name}: {missing}")
        reader = dataset_view.reader(index=index)
        return reader.select(*projection).to_arrow_table()

    def read_pandas(
        self,
        *,
        contents: str | Sequence[str],
        selectors: Sequence[str],
        index: str,
    ) -> pd.DataFrame:
        dataset_view = self._dataset_view(contents)
        projection = [index, *_normalize_selectors(selectors, index=index)]
        return dataset_view.reader(index=index).select(*projection).to_pandas()
