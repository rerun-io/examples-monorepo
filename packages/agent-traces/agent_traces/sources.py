"""Resolved recording inputs shared by discovery, hashing, and parsing."""

import hashlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import orjson
from serde import SerdeError

from agent_traces.events import Session

DecodedRecord = TypeVar("DecodedRecord")


def iter_jsonl(path: Path, decode: Callable[[dict[str, object], int], DecodedRecord]) -> Iterator[DecodedRecord]:  # noqa: UP047 - Runtime TypeVar annotations.
    """Stream strictly decoded records with their source line in any error."""
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            try:
                raw: object = orjson.loads(line)
            except orjson.JSONDecodeError:
                raise ValueError(f"{path}:{index + 1}: invalid JSON") from None
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{index + 1}: expected a JSON object")
            try:
                yield decode(raw, index)
            except (SerdeError, ValueError) as error:
                raise ValueError(f"{path}:{index + 1}: invalid record structure ({type(error).__name__})") from None


@dataclass(frozen=True, slots=True)
class SessionSource:
    """One recording's identity, dependencies, and provider parser."""

    session_id: str
    """Canonical recording and manifest identity."""
    main: Path
    """Resolved main transcript."""
    inputs: tuple[Path, ...]
    """Resolved files, in fingerprint order, with the main transcript first."""
    parse: Callable[[], Session]
    """Parse the inventoried files without discovering them again."""
    project: str = ""
    """Provider working directory or encoded project directory for filtering."""


@dataclass(frozen=True, slots=True)
class Discovery:
    """Sessions and path-addressed exclusions from one home."""

    sessions: list[SessionSource] = field(default_factory=list)
    """Recording owners."""
    failed: dict[Path, str] = field(default_factory=dict)
    """Paths whose inventory could not be read."""


def fingerprint(inputs: tuple[Path, ...]) -> str:
    """Hash framed relative paths, sizes, and bytes from one source inventory."""
    digest = hashlib.sha256()
    for source in inputs:
        digest.update(os.path.relpath(source, inputs[0].parent).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(source.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with source.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()
