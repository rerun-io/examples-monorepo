"""Typed record boundaries shared by dataset readers: one decode door that names the source on failure.

Records go through pyserde. A numeric CSV table (an MPS trajectory, a timesync file) is a stream of rows, so
``read_csv_columns`` parses only the named columns into typed Arrow arrays instead (CSV has no pyserde codec, and
``DictReader`` + ``from_dict`` per row is ~80x slower on a 250k-row trajectory)."""

from pathlib import Path
from types import GenericAlias
from typing import Literal, TypeVar
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.csv as pacsv
from serde import SerdeError
from serde.json import from_json
from serde.yaml import from_yaml
from yaml import YAMLError

SourceT = TypeVar("SourceT")


def decode(  # noqa: UP047 — beartype requires legacy generics
    cls: type[SourceT] | GenericAlias, text: str, *, source: str, fmt: Literal["json", "yaml"] = "json", redact: str | None = None
) -> SourceT:
    """Decode one record, naming ``source`` on parser, schema and ``__post_init__`` (``ValueError``) errors.

    With ``redact`` set the error names only that description and the error type: pyserde's message quotes the
    rejected value, which must not leak when the document holds secrets (signed URLs).
    """
    try:
        return from_json(cls, text) if fmt == "json" else from_yaml(cls, text)
    except (SerdeError, ValueError, YAMLError) as error:  # ValueError covers json.JSONDecodeError
        if redact is not None:
            raise ValueError(f"{source}: {redact} ({type(error).__name__})") from None
        raise ValueError(f"{source}: {error}") from error


def read_json(path: Path, cls: type[SourceT] | GenericAlias, *, text: str | None = None) -> SourceT:  # noqa: UP047 — beartype requires legacy generics
    """Decode a JSON record file (or its already-read ``text``), naming the file on errors."""
    return decode(cls, path.read_text() if text is None else text, source=str(path))


def read_member(archive: ZipFile, member: str, cls: type[SourceT], *, fmt: Literal["json", "yaml"] = "yaml") -> SourceT:  # noqa: UP047 — beartype requires legacy generics
    """Decode one archive member, naming ``<archive>:<member>`` on errors."""
    return decode(cls, archive.read(member).decode(), source=f"{archive.filename}:{member}", fmt=fmt)


def read_csv_columns(path: Path, columns: dict[str, pa.DataType]) -> pa.Table:
    """Only ``columns`` of a headed CSV, each parsed to its Arrow type, in the order of ``columns``.

    Empty and ``nan`` float cells read as NaN. A missing column, a cell that does not parse as its type (``0.5`` in an
    int64 column) and an empty integer cell raise ``ValueError`` naming ``path``.
    """
    try:
        table: pa.Table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(include_columns=list(columns), column_types=columns))
    except (pa.ArrowInvalid, pa.ArrowKeyError) as error:
        raise ValueError(f"{path}: {error}") from error
    for name, kind in columns.items():
        if pa.types.is_integer(kind) and table.column(name).null_count:
            raise ValueError(f"{path}: {name} has empty cells")
    return table
