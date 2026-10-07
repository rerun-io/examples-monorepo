"""Resolved recording inputs shared by discovery, hashing, and parsing."""

import hashlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeVar

import orjson
from serde import SerdeError

from agent_traces.events import Session

MISSING_INPUT: str = "missing"
"""Fingerprint marker shared by extra-input collection and incremental checks."""


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


FOLDED_SUBAGENT: str = "folded-subagent"
"""A child included in its owner's recording."""
PARENT_SKIPPED: str = "parent-skipped"
"""A child excluded by its owner's eligibility policy."""
PARENT_FAILED: str = "parent-failed"
"""A child whose owner could not be converted."""


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
    folded: tuple[Path, ...] = ()
    """Children counted as folded only after their owner produces a recording."""


@dataclass(frozen=True, slots=True)
class Discovery:
    """Sessions and path-addressed exclusions from one home."""

    sessions: list[SessionSource] = field(default_factory=list)
    """Recording owners."""
    skipped: dict[Path, str] = field(default_factory=dict)
    """Excluded paths and their policy reasons."""
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


def input_digest(path: Path) -> str:
    """Hash a known extra input before parsing, including an absent-file marker."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError:
        return MISSING_INPUT


def fingerprint_with_extras(transcript_hash: str, extras: dict[str, str]) -> str:
    """Combine the pre-parse transcript hash with hashes of consumed extra inputs."""
    if not extras:
        return transcript_hash
    return hashlib.sha256(transcript_hash.encode() + orjson.dumps(sorted(extras.items()))).hexdigest()


def parse_fingerprinted(source: SessionSource, transcript_hash: str) -> Session:
    """Parse the inventoried bytes and add hashes of extra inputs actually consumed."""
    session: Session = source.parse()
    return replace(session, source_sha256=fingerprint_with_extras(transcript_hash, session.extra_inputs))
