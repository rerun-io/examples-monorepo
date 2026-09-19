"""Strict conversion progress with atomic publication."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import orjson
from serde import SerdeError, serde
from serde.json import from_json, to_json


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
    n_rows: int
    """Temporal rows in the saved recording, excluding properties."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Manifest:
    """Completed conversions for one profile."""

    version: int = 1
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
        return from_json(Manifest, path.read_bytes())
    except (SerdeError, orjson.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Atomically replace saved progress in the same directory.

    Args:
        manifest: Completed session entries.
        path: Destination JSON path.
    """
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(to_json(manifest))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


