"""Resolved recording inputs shared by discovery, hashing, and parsing."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from serde import serde
from serde.json import from_json

from agent_traces.events import Session


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


@runtime_checkable
class Provider(Protocol):
    """Provider operations used by both commands."""

    def discover(self, home: Path) -> Discovery: ...
    def session_source(self, path: Path) -> SessionSource: ...


@serde
@dataclass(frozen=True, slots=True)
class _FirstRecord:
    """Minimal provider discriminator, independent of either record schema."""

    type: str = ""
    """First record tag."""


def provider_for(path: Path) -> Provider:
    """Detect a home by layout or a transcript by its first record."""
    from agent_traces import claude, codex

    if path.is_dir():
        return codex if not (path / "projects").is_dir() and any((path / name).is_dir() for name in ("sessions", "archived_sessions")) else claude
    with path.open("rb") as stream:
        return codex if from_json(_FirstRecord, stream.readline()).type == "session_meta" else claude
