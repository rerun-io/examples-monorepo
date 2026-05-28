"""Semantic comparison helpers for Robocap RRD recordings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import rerun as rr

CONTROL_COLUMN_NAMES: frozenset[str] = frozenset({"rerun.controls.RowId", "log_tick", "log_time"})
MEDIA_COLUMN_MARKERS: tuple[str, ...] = (
    ":AssetVideo:",
    ":Blob:",
    ":DepthImage:",
    ":EncodedImage:",
    ":Image:",
    ":SegmentationImage:",
    ":Tensor:",
    ":VideoFrameReference:",
    ":VideoStream:",
)


class RrdComparisonError(ValueError):
    """Raised when two RRD recordings differ outside the configured tolerance."""


@dataclass(frozen=True, slots=True)
class RrdCompareConfig:
    """Configuration for comparing a candidate Robocap RRD against a reference RRD."""

    reference_rrd: Path = Path("/tmp/robocap_cuvslam_0_2_0_baseline.rrd")
    """Known-good RRD produced with the previous cuVSLAM package."""
    candidate_rrd: Path = Path("/tmp/robocap_cuvslam_15_candidate.rrd")
    """Newly generated RRD to validate."""
    index: str = "video_time"
    """Rerun timeline used as the comparison index."""
    rtol: float = 1e-2
    """Relative tolerance for floating point component values."""
    atol: float = 2.5e-1
    """Absolute tolerance for floating point component values."""
    max_rows: int | None = None
    """Optional number of leading timeline rows to compare."""
    include_column_prefixes: tuple[str, ...] = ("/world/rig:Transform3D",)
    """Comparable RRD column prefixes to include in the semantic comparison."""


@dataclass(frozen=True, slots=True)
class RrdCompareReport:
    """Summary of a successful RRD comparison."""

    rows: int
    """Number of rows compared on the selected index."""
    columns: tuple[str, ...]
    """Comparable component columns that were compared."""
    skipped_columns: tuple[str, ...]
    """Schema columns skipped because they are media, metadata, or unsupported types."""


def _is_media_or_metadata_column(column_name: str) -> bool:
    if column_name.startswith("property:"):
        return True
    return any(marker in column_name for marker in MEDIA_COLUMN_MARKERS)


def _is_comparable_type(data_type: pa.DataType) -> bool:
    if (
        pa.types.is_null(data_type)
        or pa.types.is_boolean(data_type)
        or pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_duration(data_type)
        or pa.types.is_timestamp(data_type)
        or pa.types.is_date(data_type)
        or pa.types.is_time(data_type)
    ):
        return True
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type):
        return _is_comparable_type(data_type.value_type)
    if pa.types.is_struct(data_type):
        return all(_is_comparable_type(field.type) for field in data_type)
    return False


def _has_float_type(data_type: pa.DataType) -> bool:
    if pa.types.is_floating(data_type):
        return True
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type):
        return _has_float_type(data_type.value_type)
    if pa.types.is_struct(data_type):
        return any(_has_float_type(field.type) for field in data_type)
    return False


def _matches_column_prefixes(column_name: str, include_column_prefixes: tuple[str, ...] | None) -> bool:
    if include_column_prefixes is None:
        return True
    return any(column_name.startswith(prefix) for prefix in include_column_prefixes)


def comparable_schema_columns(
    schema: pa.Schema,
    *,
    index: str,
    include_column_prefixes: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return sorted non-media component columns that can be compared mechanically.

    Args:
        schema: Arrow schema returned by a Rerun dataset view.
        index: Timeline column used as the comparison index.

    Returns:
        Comparable schema column names.
    """
    columns: list[str] = []
    for field in schema:
        column_name: str = field.name
        if column_name == index or column_name in CONTROL_COLUMN_NAMES:
            continue
        if _is_media_or_metadata_column(column_name):
            continue
        if not _matches_column_prefixes(column_name, include_column_prefixes):
            continue
        if not _is_comparable_type(field.type):
            continue
        columns.append(column_name)
    return tuple(sorted(columns))


def skipped_schema_columns(
    schema: pa.Schema,
    *,
    index: str,
    include_column_prefixes: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return sorted schema columns not included in the mechanical comparison."""
    comparable_columns: set[str] = set(
        comparable_schema_columns(schema, index=index, include_column_prefixes=include_column_prefixes)
    )
    skipped_columns: list[str] = []
    for field in schema:
        column_name: str = field.name
        if column_name == index or column_name in CONTROL_COLUMN_NAMES:
            continue
        if column_name not in comparable_columns:
            skipped_columns.append(column_name)
    return tuple(sorted(skipped_columns))


def _column_values(table: pa.Table, column_name: str) -> list[Any]:
    column: pa.ChunkedArray = table.column(column_name)
    values: list[Any] = []
    for chunk in column.chunks:
        values.extend(chunk.to_pylist())
    return values


def _null_mask(table: pa.Table, column_name: str) -> list[bool]:
    column: pa.ChunkedArray = table.column(column_name)
    mask: list[bool] = []
    for chunk in column.chunks:
        mask.extend(chunk.is_null().to_pylist())
    return mask


def _raise_value_mismatch(column_name: str, location: str, reference_value: Any, candidate_value: Any) -> None:
    raise RrdComparisonError(
        f"Value differs for column '{column_name}' at {location}: "
        f"reference={reference_value!r}, candidate={candidate_value!r}"
    )


def _compare_nested_values(
    *,
    column_name: str,
    location: str,
    reference_value: Any,
    candidate_value: Any,
    rtol: float,
    atol: float,
) -> None:
    if reference_value is None or candidate_value is None:
        if reference_value != candidate_value:
            _raise_value_mismatch(column_name, location, reference_value, candidate_value)
        return
    if isinstance(reference_value, dict) or isinstance(candidate_value, dict):
        if not isinstance(reference_value, dict) or not isinstance(candidate_value, dict):
            _raise_value_mismatch(column_name, location, reference_value, candidate_value)
        if reference_value.keys() != candidate_value.keys():
            _raise_value_mismatch(column_name, location, reference_value, candidate_value)
        for key in sorted(reference_value):
            _compare_nested_values(
                column_name=column_name,
                location=f"{location}.{key}",
                reference_value=reference_value[key],
                candidate_value=candidate_value[key],
                rtol=rtol,
                atol=atol,
            )
        return
    if isinstance(reference_value, list) or isinstance(candidate_value, list):
        if not isinstance(reference_value, list) or not isinstance(candidate_value, list):
            _raise_value_mismatch(column_name, location, reference_value, candidate_value)
        if len(reference_value) != len(candidate_value):
            _raise_value_mismatch(column_name, location, reference_value, candidate_value)
        for item_index, (reference_item, candidate_item) in enumerate(zip(reference_value, candidate_value, strict=True)):
            _compare_nested_values(
                column_name=column_name,
                location=f"{location}[{item_index}]",
                reference_value=reference_item,
                candidate_value=candidate_item,
                rtol=rtol,
                atol=atol,
            )
        return
    if isinstance(reference_value, bool) or isinstance(candidate_value, bool):
        if reference_value != candidate_value:
            _raise_value_mismatch(column_name, location, reference_value, candidate_value)
        return
    if isinstance(reference_value, int | float | np.integer | np.floating) and isinstance(candidate_value, int | float | np.integer | np.floating):
        if not bool(np.isclose(reference_value, candidate_value, rtol=rtol, atol=atol, equal_nan=True)):
            _raise_value_mismatch(column_name, location, reference_value, candidate_value)
        return
    if reference_value != candidate_value:
        _raise_value_mismatch(column_name, location, reference_value, candidate_value)


def compare_arrow_tables(
    *,
    reference_table: pa.Table,
    candidate_table: pa.Table,
    index: str,
    columns: tuple[str, ...],
    rtol: float,
    atol: float,
) -> tuple[str, ...]:
    """Compare selected Arrow table columns using exact masks and tolerant floats.

    Args:
        reference_table: Known-good Rerun query table.
        candidate_table: Candidate Rerun query table.
        index: Index column that must match exactly.
        columns: Component columns to compare.
        rtol: Relative tolerance for floating point values.
        atol: Absolute tolerance for floating point values.

    Returns:
        Sorted column names that were compared.

    Raises:
        RrdComparisonError: If indexes, schemas, null masks, or values differ.
    """
    reference_names: set[str] = set(reference_table.column_names)
    candidate_names: set[str] = set(candidate_table.column_names)
    required_names: set[str] = {index, *columns}
    missing_reference: set[str] = required_names - reference_names
    missing_candidate: set[str] = required_names - candidate_names
    if missing_reference:
        raise RrdComparisonError(f"Reference table is missing columns: {sorted(missing_reference)}")
    if missing_candidate:
        raise RrdComparisonError(f"Candidate table is missing columns: {sorted(missing_candidate)}")

    reference_index: list[Any] = _column_values(reference_table, index)
    candidate_index: list[Any] = _column_values(candidate_table, index)
    if reference_index != candidate_index:
        raise RrdComparisonError("Index values differ between reference and candidate RRDs.")

    compared_columns: tuple[str, ...] = tuple(sorted(columns))
    for column_name in compared_columns:
        reference_type: pa.DataType = reference_table.schema.field(column_name).type
        candidate_type: pa.DataType = candidate_table.schema.field(column_name).type
        if reference_type != candidate_type:
            raise RrdComparisonError(
                f"Schema type differs for column '{column_name}': reference={reference_type}, candidate={candidate_type}"
            )

        reference_nulls: list[bool] = _null_mask(reference_table, column_name)
        candidate_nulls: list[bool] = _null_mask(candidate_table, column_name)
        if reference_nulls != candidate_nulls:
            raise RrdComparisonError(f"Null mask differs for column '{column_name}'.")

        reference_values: list[Any] = _column_values(reference_table, column_name)
        candidate_values: list[Any] = _column_values(candidate_table, column_name)
        if _has_float_type(reference_type):
            for row_index, (reference_value, candidate_value) in enumerate(zip(reference_values, candidate_values, strict=True)):
                _compare_nested_values(
                    column_name=column_name,
                    location=f"row {row_index}",
                    reference_value=reference_value,
                    candidate_value=candidate_value,
                    rtol=rtol,
                    atol=atol,
                )
        elif reference_values != candidate_values:
            raise RrdComparisonError(f"Value differs for exact column '{column_name}'.")

    return compared_columns


def _query_table(dataset: Any, *, index: str, columns: tuple[str, ...]) -> pa.Table:
    from datafusion import col

    dataset_view: Any = dataset.filter_contents(["/**"])
    projection: list[Any] = [col(index), *[col(column_name) for column_name in columns]]
    return dataset_view.reader(index=index, fill_latest_at=False).select(*projection).to_arrow_table()


def _query_schema(dataset: Any, *, index: str) -> pa.Schema:
    dataset_view: Any = dataset.filter_contents(["/**"])
    return dataset_view.reader(index=index, fill_latest_at=False).schema()


def compare_rrd_reference(config: RrdCompareConfig) -> RrdCompareReport:
    """Compare a candidate Robocap RRD against a reference RRD.

    Args:
        config: Reference, candidate, index, and tolerance configuration.

    Returns:
        Successful comparison summary.

    Raises:
        RrdComparisonError: If the candidate differs from the reference.
    """
    reference_path: Path = config.reference_rrd.expanduser().resolve()
    candidate_path: Path = config.candidate_rrd.expanduser().resolve()
    if not reference_path.exists():
        raise RrdComparisonError(f"Reference RRD does not exist: {reference_path}")
    if not candidate_path.exists():
        raise RrdComparisonError(f"Candidate RRD does not exist: {candidate_path}")

    with rr.server.Server(datasets={"reference": [str(reference_path)], "candidate": [str(candidate_path)]}) as server:
        reference_dataset: Any = server.client().get_dataset("reference")
        candidate_dataset: Any = server.client().get_dataset("candidate")
        reference_schema: pa.Schema = _query_schema(reference_dataset, index=config.index)
        candidate_schema: pa.Schema = _query_schema(candidate_dataset, index=config.index)
        reference_columns: tuple[str, ...] = comparable_schema_columns(
            reference_schema,
            index=config.index,
            include_column_prefixes=config.include_column_prefixes,
        )
        candidate_columns: tuple[str, ...] = comparable_schema_columns(
            candidate_schema,
            index=config.index,
            include_column_prefixes=config.include_column_prefixes,
        )

        missing_columns: set[str] = set(reference_columns) - set(candidate_columns)
        extra_columns: set[str] = set(candidate_columns) - set(reference_columns)
        if missing_columns or extra_columns:
            raise RrdComparisonError(
                "Comparable schema columns differ. "
                f"missing={sorted(missing_columns)}, extra={sorted(extra_columns)}"
            )

        reference_table: pa.Table = _query_table(reference_dataset, index=config.index, columns=reference_columns)
        candidate_table: pa.Table = _query_table(candidate_dataset, index=config.index, columns=candidate_columns)
        if config.max_rows is not None:
            if config.max_rows <= 0:
                raise RrdComparisonError("max_rows must be positive when provided.")
            reference_table = reference_table.slice(0, config.max_rows)
            candidate_table = candidate_table.slice(0, config.max_rows)
        compared_columns: tuple[str, ...] = compare_arrow_tables(
            reference_table=reference_table,
            candidate_table=candidate_table,
            index=config.index,
            columns=reference_columns,
            rtol=config.rtol,
            atol=config.atol,
        )
        skipped_columns: tuple[str, ...] = skipped_schema_columns(
            reference_schema,
            index=config.index,
            include_column_prefixes=config.include_column_prefixes,
        )

    return RrdCompareReport(
        rows=reference_table.num_rows,
        columns=compared_columns,
        skipped_columns=skipped_columns,
    )
