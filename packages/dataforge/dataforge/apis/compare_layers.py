"""Compare RRD data by entity, component and source clocks, independent of chunks.

Generated row IDs, log_time/log_tick, store IDs and blueprints are excluded.
The build-time wall clock (/__properties, RecordingInfo:start_time) is also
ignored by default; --ignore accepts extra entity/component pairs.
All data components (including properties and video bytes) are compared. Float
values use absolute tolerance 1e-6, with NaNs equal. Row counts remain exact.
"""

from collections import defaultdict
from pathlib import Path
from typing import Annotated, TypeAlias

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun.chunk as rrc
import tyro
from jaxtyping import Float
from numpy import ndarray

ColumnKey: TypeAlias = tuple[str, str, tuple[str, ...]]


def component_rows(path: Path) -> dict[ColumnKey, pa.Table]:
    """Read per-component rows, retaining empty cells and removing absent cells."""
    parts: dict[ColumnKey, list[pa.Table]] = defaultdict(list)
    reader: rrc.RrdReader = rrc.RrdReader(path)
    if len(reader.recordings()) != 1:
        raise ValueError(f"{path}: expected one data recording")
    for chunk in reader.stream():
        batch: pa.RecordBatch = chunk.to_record_batch()
        timelines: tuple[str, ...] = tuple(sorted(name for name in chunk.timeline_names if name not in {"log_time", "log_tick"}))
        for field in batch.schema:
            if (field.metadata or {}).get(b"rerun:kind") != b"data":
                continue
            key: ColumnKey = (str(chunk.entity_path), field.name, timelines)
            table: pa.Table = pa.Table.from_batches([batch.select([*timelines, field.name])]).replace_schema_metadata(None)
            parts[key].append(table.filter(table[field.name].is_valid()))
    result: dict[ColumnKey, pa.Table] = {}
    for key, tables in parts.items():
        combined: pa.Table = pa.concat_tables(tables)
        result[key] = combined.sort_by([(name, "ascending") for name in key[2]]) if key[2] else combined
    return result


def assert_values_equal(left: pa.Array, right: pa.Array, *, atol: float) -> None:
    """Compare Arrow nesting, nulls and exact non-floats without expanding video blobs."""
    assert left.type == right.type, f"types differ: {left.type} != {right.type}"
    assert len(left) == len(right), f"length differs: {len(left)} != {len(right)}"
    assert left.is_null().equals(right.is_null()), "null masks differ"
    if pa.types.is_list(left.type) or pa.types.is_large_list(left.type) or pa.types.is_fixed_size_list(left.type):
        assert pc.call_function("list_value_length", [left]).equals(pc.call_function("list_value_length", [right])), "cell lengths differ"
        assert_values_equal(pc.call_function("list_flatten", [left]), pc.call_function("list_flatten", [right]), atol=atol)
    elif pa.types.is_struct(left.type):
        for index in range(left.type.num_fields):
            assert_values_equal(left.field(index), right.field(index), atol=atol)
    elif pa.types.is_floating(left.type):
        np.testing.assert_allclose(left.to_numpy(zero_copy_only=False), right.to_numpy(zero_copy_only=False), rtol=0.0, atol=atol, equal_nan=True)
    else:
        assert left.equals(right), "values differ"


def max_float_difference(left: pa.Array, right: pa.Array) -> float | None:
    """Return the maximum comparable float delta, or None for non-float data."""
    if left.type != right.type or len(left) != len(right):
        return None
    if pa.types.is_list(left.type) or pa.types.is_large_list(left.type) or pa.types.is_fixed_size_list(left.type):
        if not pc.call_function("list_value_length", [left]).equals(pc.call_function("list_value_length", [right])):
            return None
        return max_float_difference(pc.call_function("list_flatten", [left]), pc.call_function("list_flatten", [right]))
    if pa.types.is_struct(left.type):
        differences: list[float] = [
            value for index in range(left.type.num_fields)
            if (value := max_float_difference(left.field(index), right.field(index))) is not None
        ]
        return max(differences, default=None)
    if pa.types.is_floating(left.type):
        with np.errstate(invalid="ignore"):
            deltas: Float[ndarray, "n"] = np.abs(left.to_numpy(zero_copy_only=False) - right.to_numpy(zero_copy_only=False))
        return float(np.nanmax(deltas)) if np.any(~np.isnan(deltas)) else None
    return None


def compare_layers(a: Path, b: Path, *, atol: float = 1e-6, ignore: list[tuple[str, str]] | None = None) -> None:
    """Print every track/column mismatch, then raise with the mismatch count.

    Row indices are zero-based in source-clock order. Float deltas cover all
    comparable values in the column; n/a means no aligned float values exist.
    """
    ignored: set[tuple[str, str]] = {("/__properties", "RecordingInfo:start_time"), *(ignore or [])}
    left: dict[ColumnKey, pa.Table] = {key: table for key, table in component_rows(a).items() if key[:2] not in ignored}
    right: dict[ColumnKey, pa.Table] = {key: table for key, table in component_rows(b).items() if key[:2] not in ignored}
    mismatches: list[str] = []
    for key in sorted(left.keys() - right.keys()):
        mismatches.append(f"{key}: missing in b")
    for key in sorted(right.keys() - left.keys()):
        mismatches.append(f"{key}: extra in b")
    for key in sorted(left.keys() & right.keys()):
        table: pa.Table = left[key]
        other: pa.Table = right[key]
        if table.num_rows != other.num_rows:
            mismatches.append(f"{key}: row counts differ: {table.num_rows} != {other.num_rows}")
        count: int = min(table.num_rows, other.num_rows)
        for name in table.column_names:
            first: pa.Array = table[name].combine_chunks().slice(0, count)
            second: pa.Array = other[name].combine_chunks().slice(0, count)
            tolerance: float = atol if name == key[1] else 0.0
            try:
                assert_values_equal(first, second, atol=tolerance)
            except AssertionError:
                # Search prefixes to locate the first differing row without expanding blobs.
                low: int = 0
                high: int = max(0, count - 1)
                while low < high:
                    middle: int = (low + high) // 2
                    try:
                        assert_values_equal(first.slice(0, middle + 1), second.slice(0, middle + 1), atol=tolerance)
                        low = middle + 1
                    except AssertionError:
                        high = middle
                times_a: dict[str, object] = {clock: table[clock][low].as_py() for clock in key[2]} if count else {}
                times_b: dict[str, object] = {clock: other[clock][low].as_py() for clock in key[2]} if count else {}
                delta: float | None = max_float_difference(first, second)
                mismatches.append(
                    f"{key}: {name} differs; first differing row {low}, time a={times_a}, b={times_b}; "
                    f"max abs float difference={delta if delta is not None else 'n/a'}"
                )
    if mismatches:
        for mismatch in mismatches:
            print(mismatch)
        raise AssertionError(f"{len(mismatches)} mismatch(es): " + "; ".join(mismatches))
    print(f"equal: {a} == {b} ({len(left)} component tracks, atol={atol})")


def main(
    a: Annotated[Path, tyro.conf.Positional], b: Annotated[Path, tyro.conf.Positional], *,
    atol: float = 1e-6, ignore: list[tuple[str, str]] | None = None,
) -> None:
    """Compare two layer RRDs; --ignore takes extra entity/component pairs."""
    try:
        compare_layers(a, b, atol=atol, ignore=ignore)
    except AssertionError as error:
        raise SystemExit(str(error).split(":", 1)[0]) from None
