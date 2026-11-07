"""The frozen CUDA-kernel references of the Mojo backend: one .rrd of static records, read back bit-exactly.

Each record is one entity (`/iter_proj/<case>`, `/refine_matches/<case>`, `/gn_rays/<case>`, `/gn_calib/<case>`)
logged as static `rr.AnyValues`. Its tensors are packed into one flat array per dtype (fields `bool`, `float16`,
`float32`, `int64`); the `#index` text field (JSON) maps each tensor name to [dtype, offset, shape]. The layout is
dpvo's and lietorch-mojo's `tests/references.py` plus bool.

Provenance: `kernel_references.rrd` was written from real MASt3R-SLAM inputs (normal-apt-tour.mp4, the 30-frame test
configuration) by the former CUDA extension `mast3r_slam._backends` with `tests/freeze_kernel_references.py` at commit
74a125725: one late `iter_proj`, `refine_matches` and `gauss_newton_rays` call (every 6th query point, the first 3
keyframes of the GN problem), and the extension's calib solve on that problem with the points constrained to the
rays of a pinhole K. The extension and the script were deleted when the Mojo kernels replaced them; the fixture is
the frozen record of the extension's outputs. The parity tests read it with `rerun.chunk.RrdReader`.
"""

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from rerun.chunk import RrdReader
from torch import Tensor

REFERENCE_FILE: Path = Path(__file__).parent / "fixtures" / "kernel_references.rrd"
INDEX_FIELD: str = "#index"
TENSOR_DTYPES: dict[str, torch.dtype] = {"bool": torch.bool, "float16": torch.float16, "float32": torch.float32, "int64": torch.int64}


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

