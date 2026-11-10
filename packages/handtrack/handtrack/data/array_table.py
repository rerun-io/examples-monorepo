"""Named arrays that share their first axis (one row per sample), as one Parquet table: the RKNN conversion hand-off, nothing else.

This is the monorepo's one non-Rerun data file (AGENTS.md, Serialization). ``apis/export_detnet.py`` and ``apis/export_keynet.py``
write their PyTorch references here, and ``robocap-live/tools/rknn_convert.py`` reads them with pyarrow alone in Rockchip's
RKNN-Toolkit2 environment, which cannot load rerun-sdk: rknn-toolkit2 2.3.2 (and 2.5.0, the latest) pins numpy <=1.26.4, while
rerun-sdk >=0.24 needs numpy >=2. Revisit when Rockchip supports numpy 2; give no other data this format.

Each array becomes a column of fixed-size lists holding one row's values; the per-row shape and the dtype live in the column's
field metadata, so a reader gets back exactly the arrays it wrote.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy import ndarray

SHAPE_KEY: bytes = b"array_shape"
DTYPE_KEY: bytes = b"array_dtype"


def write_arrays(path: Path, arrays: dict[str, ndarray], table_metadata: dict[bytes, bytes] | None = None) -> None:
    """One column per array; every array's first axis is the row axis. ``table_metadata`` goes to the schema (``pq.read_schema``)."""
    rows: set[int] = {len(value) for value in arrays.values()}
    if len(rows) != 1:
        raise ValueError(f"arrays must share their rows, got {sorted(rows)}")
    columns: list[pa.Array] = []
    fields: list[pa.Field] = []
    for name, value in arrays.items():
        flat: ndarray = np.ascontiguousarray(value).reshape(len(value), -1)
        width: int = flat.shape[1]
        column: pa.Array = pa.FixedSizeListArray.from_arrays(pa.array(flat.reshape(-1)), width) if value.ndim > 1 else pa.array(value)
        columns.append(column)
        metadata: dict[bytes, bytes] = {SHAPE_KEY: json.dumps(list(value.shape[1:])).encode(), DTYPE_KEY: str(value.dtype).encode()}
        fields.append(pa.field(name, column.type, metadata=metadata))
    pq.write_table(pa.Table.from_arrays(columns, schema=pa.schema(fields, metadata=table_metadata)), path)


def read_arrays(path: Path) -> dict[str, ndarray]:
    """The arrays ``write_arrays`` wrote, in column order."""
    table: pa.Table = pq.read_table(path)
    out: dict[str, ndarray] = {}
    for field in table.schema:
        metadata: dict[bytes, bytes] = field.metadata or {}
        shape: list[int] = json.loads(metadata[SHAPE_KEY])
        dtype: np.dtype = np.dtype(metadata[DTYPE_KEY].decode())
        column: pa.Array = table[field.name].combine_chunks()
        values: ndarray = (column.flatten() if shape else column).to_numpy(zero_copy_only=False)
        out[field.name] = values.astype(dtype, copy=False).reshape(len(column), *shape)
    return out
