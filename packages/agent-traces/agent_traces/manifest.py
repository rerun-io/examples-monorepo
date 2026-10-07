"""Strict conversion progress with atomic publication."""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import orjson
from serde import SerdeError, serde
from serde.json import from_json, to_json

from agent_traces.writing import atomic_write

CONVERSION_REVISION: int = 19  # Bump when the recording layout or content changes.


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One completed session conversion."""

    source_path: str
    """Absolute main transcript path."""
    source_sha256: str
    """Framed hash of session input paths, sizes, and bytes."""
    rrd: str
    """Recording path relative to the profile directory."""
    converted_at: str
    """UTC conversion time in ISO-8601 format."""
    host: str
    """Effective hostname written to the recording."""
    revision: int
    """Conversion content revision, independent of the manifest schema."""
    n_rows: int
    """Temporal rows in the saved recording, excluding properties."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Manifest:
    """Completed conversions for one profile."""

    version: Literal[2] = 2
    """Manifest schema version."""
    sessions: dict[str, ManifestEntry] = field(default_factory=dict)
    """Completed entries keyed by session id."""


def load_manifest(path: Path) -> Manifest:
    """Load a strict manifest, naming the source on decode errors.

    Args:
        path: Manifest JSON path.

    Returns:
        Saved progress, or an empty manifest if absent.

    Raises:
        ValueError: The manifest cannot be decoded.
    """
    if not path.exists():
        return Manifest()
    try:
        content: bytes = path.read_bytes()
        return from_json(Manifest, content)
    except (SerdeError, orjson.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}; delete it to convert everything again") from error


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Atomically replace saved progress in the same directory.

    Args:
        manifest: Completed session entries.
        path: Destination JSON path.
    """
    with atomic_write(path) as temporary:
        temporary.write_text(to_json(manifest))


@contextmanager
def manifest_lock(profile: Path) -> Iterator[None]:
    """Hold one exclusive profile transaction, waiting for concurrent writers."""
    with (profile / "manifest.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Waiting for manifest lock: {profile / 'manifest.lock'}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
