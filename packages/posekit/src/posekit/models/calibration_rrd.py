"""Activation calibration records stored as static Rerun recordings.

Layout, one .rrd per model, application_id ``posekit-calibration``, recording_id the model
name. Every chunk is static and holds one row; there is no timeline.

- Recording property ``calibration``: ``model`` plus each scalar field of the record
  (``revision``, ``crop_size``, crop counts, ``segment``).
- ``/ranges/<field>``: parallel ``module: list[str]`` and ``amax: list[float64]`` columns,
  e.g. ``/ranges/input_amax`` and ``/ranges/attention_amax``.
- ``/split/<name>``: parallel ``source: list[str]``, ``index: list[int64]`` and, for decoded
  frames, ``sha256: list[str]`` columns, e.g. ``/split/calibration`` and ``/split/evaluation``.

Ranges are float64, so every value round-trips bit-exactly. Reading needs no viewer or server.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pyarrow as pa
import rerun as rr
from jaxtyping import UInt8
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from torch import Tensor

APPLICATION_ID: str = "posekit-calibration"
"""Application id shared by every posekit calibration recording."""
PROPERTY: str = "calibration"
"""Recording property holding the model name and the record's scalar fields."""


def frame_sha256(frame: UInt8[Tensor, "h w 3"]) -> str:
    """SHA256 of decoded RGB pixels in row-major order: the identity calibration and evaluation splits compare."""
    return hashlib.sha256(frame.cpu().numpy().tobytes()).hexdigest()


COLUMN_TYPES: dict[str, pa.DataType] = {
    "module": pa.string(), "amax": pa.float64(), "source": pa.string(), "index": pa.int64(), "sha256": pa.string(),
}
"""Arrow type of every table column the layout allows."""

CalibrationScalar: TypeAlias = str | int | float
CalibrationColumn: TypeAlias = list[str] | list[int] | list[float]


@dataclass(frozen=True, slots=True)
class CalibrationRecording:
    """The static contents of one calibration .rrd, before the model's record checks them."""

    properties: dict[str, CalibrationScalar]
    """Scalar fields of the ``calibration`` property, including ``model``."""
    tables: dict[str, dict[str, CalibrationColumn]]
    """Parallel columns keyed by entity path, then by column name."""

    def ranges(self, field: str) -> dict[str, float]:
        """The ``/ranges/<field>`` table as module name to maximum, in stored order; KeyError if it is missing."""
        table: dict[str, CalibrationColumn] = self.tables[f"/ranges/{field}"]
        return {str(module): float(amax) for module, amax in zip(table["module"], table["amax"], strict=True)}


def range_tables(input_amax: dict[str, float], attention_amax: dict[str, float]) -> dict[str, dict[str, CalibrationColumn]]:
    """The ``/ranges/input_amax`` and ``/ranges/attention_amax`` tables of an activation calibration."""
    return {f"/ranges/{name}": {"module": list(amax), "amax": list(amax.values())} for name, amax in (("input_amax", input_amax), ("attention_amax", attention_amax))}


def write_calibration_rrd(path: Path, recording: CalibrationRecording) -> None:
    """Write every property and table as one static chunk each."""
    scalars: rr.AnyValues = rr.AnyValues()
    for name, value in recording.properties.items():
        scalars = scalars.with_component_from_data(name, value)
    chunks: list[Chunk] = [Chunk.from_property(PROPERTY, scalars)]
    for entity, columns in recording.tables.items():
        arrays: dict[str, pa.Array] = {name: pa.array(values, COLUMN_TYPES[name]) for name, values in columns.items()}
        lengths: set[int] = {len(array) for array in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"{entity}: parallel columns must have equal lengths.")
        rows: int = lengths.pop()
        # One row whose cells are whole lists: a static entity keeps only its latest row.
        chunks.append(Chunk.from_columns(entity, indexes=[], columns=[column.partition([rows]) for column in rr.AnyValues.columns(**arrays)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    LazyChunkStream.from_iter(chunks).write_rrd(path, application_id=APPLICATION_ID, recording_id=str(recording.properties["model"]))


def read_calibration_rrd(path: Path, model: str) -> CalibrationRecording:
    """Read one model's calibration recording; raise ValueError naming ``path`` otherwise."""
    reader = RrdReader(path)
    stores: list[tuple[str, str]] = [(entry.application_id, entry.recording_id) for entry in reader.recordings()]
    if stores != [(APPLICATION_ID, model)]:
        raise ValueError(f"{path}: expected one {APPLICATION_ID} recording of {model}, found {stores}.")
    properties: dict[str, CalibrationScalar] = {}
    tables: dict[str, dict[str, CalibrationColumn]] = {}
    for chunk in reader.stream().to_chunks():
        if not chunk.is_static or chunk.num_rows != 1:
            raise ValueError(f"{path}: {chunk.entity_path} must be one static row.")
        batch: pa.RecordBatch = chunk.to_record_batch()
        columns: dict[str, CalibrationColumn] = {
            field.name: batch.column(position)[0].as_py()
            for position, field in enumerate(batch.schema) if field.metadata.get(b"rerun:kind") == b"data"
        }
        # Chunk.from_property writes the property under Rerun's /__properties/<name> entity.
        if chunk.entity_path == f"/__properties/{PROPERTY}":
            properties.update({name: values[0] for name, values in columns.items()})
        else:
            tables.setdefault(chunk.entity_path, {}).update(columns)
    if properties.get("model") != model:
        raise ValueError(f"{path}: the {PROPERTY} property names model {properties.get('model')!r}, not {model}.")
    return CalibrationRecording(properties, tables)
