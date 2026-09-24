"""Strict conversion progress with atomic publication."""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import orjson
from serde import SerdeError, serde
from serde.json import from_json, to_json

from agent_traces.writing import atomic_write

CONVERSION_REVISION: int = 1  # Bump when the recording layout or content changes.


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

    version: int = 2
    """Manifest schema version."""
    sessions: dict[str, ManifestEntry] = field(default_factory=dict)
    """Completed entries keyed by session id."""


@serde
@dataclass(frozen=True, slots=True)
class _ManifestVersion:
    """Read the schema before decoding version-specific entries."""

    version: int
    """Required schema version."""


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
        version: int = from_json(_ManifestVersion, content).version
        if version != 2:
            raise ValueError(f"{path}: unsupported manifest version {version}; delete it to convert everything again")
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
