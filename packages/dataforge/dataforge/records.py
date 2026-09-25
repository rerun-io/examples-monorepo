"""Typed JSON record boundaries shared by dataset readers."""

import json
from pathlib import Path
from types import GenericAlias
from typing import TypeVar

from serde import SerdeError
from serde.json import from_json

SourceT = TypeVar("SourceT")


def read_json(path: Path, cls: type[SourceT] | GenericAlias, *, text: str | None = None) -> SourceT:  # noqa: UP047 — beartype requires legacy generics
    """Decode a third-party record, naming the file on schema/parser errors."""
    try:
        return from_json(cls, path.read_text() if text is None else text)
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error
