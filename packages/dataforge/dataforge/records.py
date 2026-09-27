"""Typed record boundaries shared by dataset readers: one decode door that names the source on failure."""

from pathlib import Path
from types import GenericAlias
from typing import Literal, TypeVar
from zipfile import ZipFile

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
