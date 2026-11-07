"""The frozen CUDA-kernel references of the Mojo backend: one .rrd of static records, read back bit-exactly.

Each record is one entity (`/corr/<case>`, `/patchify/<case>`, `/reproject/<case>`, `/ba/<case>`, ...)
logged as static `rr.AnyValues`. Its tensors are packed into one flat array per dtype (fields
`float16`, `float32`, `int64`); the `#index` text field (JSON) maps each tensor name to
[dtype, offset, shape]. The layout is lietorch-mojo's `tests/references.py` plus float16 (the
correlation runs in half precision under autocast).

Provenance: `kernel_references.rrd` was written from real DPVO inputs (pool.mp4) by DPVO's former CUDA
extensions `_cuda_corr` / `_cuda_ba` with `tests/freeze_kernel_references.py` at commit 60164d1cf, and
`metric_prior_references.rrd` (MetricTrail's BA with the depth prior) by `tests/freeze_metric_prior_references.py`
at commit 2526a20bf. The extensions and both scripts were deleted when the Mojo kernels replaced them; the
fixtures are the frozen record of the extensions' outputs. The parity tests read them with `rerun.chunk.RrdReader`.
"""

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from rerun.chunk import RrdReader
from torch import Tensor

REFERENCE_FILE: Path = Path(__file__).parent / "fixtures" / "kernel_references.rrd"
INDEX_FIELD: str = "#index"
TENSOR_DTYPES: dict[str, torch.dtype] = {"float16": torch.float16, "float32": torch.float32, "int64": torch.int64}


def encode(values: Mapping[str, Tensor | str]) -> dict[str, np.ndarray | str]:
    """AnyValues fields for one record: the tensors packed into one flat array per dtype, the `#index`, and the strings."""
    dtype_names: dict[torch.dtype, str] = {v: k for k, v in TENSOR_DTYPES.items()}
    fields: dict[str, np.ndarray | str] = {}
    packed: dict[str, list[np.ndarray]] = {}
    offsets: dict[str, int] = {}
    index: dict[str, tuple[str, int, list[int]]] = {}
    for name, value in values.items():
        assert name != INDEX_FIELD and name not in TENSOR_DTYPES, name
        if isinstance(value, str):
            fields[name] = value
            continue
        assert value.dtype in dtype_names, (name, value.dtype)
        dtype: str = dtype_names[value.dtype]
        flat: np.ndarray = value.detach().cpu().contiguous().numpy().reshape(-1)
        index[name] = (dtype, offsets.get(dtype, 0), list(value.shape))
        packed.setdefault(dtype, []).append(flat)
        offsets[dtype] = offsets.get(dtype, 0) + flat.size
    fields |= {dtype: np.concatenate(parts) for dtype, parts in packed.items()}
    fields[INDEX_FIELD] = json.dumps(index)
    return fields


@dataclass(frozen=True, slots=True)
class Record:
    """One frozen record: CPU tensors and strings by field name."""

    tensors: dict[str, Tensor]
    """Tensor fields, reshaped, in the dtype they were written with."""
    texts: dict[str, str]
    """Text fields."""

    def tensor(self, name: str) -> Tensor:
        return self.tensors[name]


def _decode(batch: pa.RecordBatch) -> Record:
    columns: dict[str, pa.Array] = {name: batch.column(name).flatten() for name in batch.schema.names if not name.startswith("rerun.")}
    index_text: pa.Array | None = columns.pop(INDEX_FIELD, None)
    index: dict[str, list] = json.loads(index_text[0].as_py()) if index_text is not None else {}
    packed: dict[str, Tensor] = {}
    for dtype, dt in TENSOR_DTYPES.items():
        if dtype in columns:
            packed[dtype] = torch.from_numpy(columns.pop(dtype).to_numpy(zero_copy_only=False).copy())
            assert packed[dtype].dtype == dt, (dtype, packed[dtype].dtype)
    tensors: dict[str, Tensor] = {}
    for name, (dtype, offset, shape) in index.items():
        n: int = int(np.prod(shape))
        tensors[name] = packed[dtype][offset : offset + n].reshape(shape).clone()
    texts: dict[str, str] = {name: values[0].as_py() for name, values in columns.items()}
    return Record(tensors=tensors, texts=texts)


@functools.cache
def load(path: Path = REFERENCE_FILE) -> dict[str, Record]:
    """Every record in the reference file by entity path (recording properties excluded)."""
    records: dict[str, Record] = {}
    for chunk in RrdReader(path).stream():
        if chunk.entity_path.startswith("/__properties"):
            continue
        assert chunk.is_static and chunk.num_rows == 1, chunk.entity_path
        assert chunk.entity_path not in records, f"two chunks for {chunk.entity_path}"
        records[chunk.entity_path] = _decode(chunk.to_record_batch())
    return records


def entities(prefix: str) -> list[str]:
    """Entity paths below `prefix` in the committed reference file, in file order."""
    return [e for e in load() if e.startswith(prefix + "/")]
