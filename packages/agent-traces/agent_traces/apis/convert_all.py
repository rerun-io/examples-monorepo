"""Convert a Claude home incrementally using a content-hash manifest."""

import hashlib
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

import orjson
from rerun.experimental import RrdReader
from serde import SerdeError, serde
from serde.json import from_json, to_json

from agent_traces.claude import ClaudeSession, parse_session
from agent_traces.rerun_log import write_session_rrd


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One completed session conversion."""

    source_path: str
    """Absolute main transcript path."""
    source_sha256: str
    """Hash of main bytes followed by sorted child transcript bytes."""
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


@dataclass(frozen=True, slots=True)
class Config:
    """Batch conversion arguments."""

    home: Path
    """Claude home directory; a leading tilde is expanded."""
    out: Path
    """Required output directory."""
    profile: str | None = None
    """Override the home directory name without its leading dot."""
    project: str | None = None
    """Only project directory names containing this substring."""
    session_id: str | None = None
    """Only convert this session id."""
    since: str | None = None
    """Only main transcripts modified on or after this ISO date."""


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


def main(config: Config) -> None:
    """Convert sessions and save progress after each successful recording.

    Args:
        config: Input home, output root, and optional selection filters.
    """
    home: Path = config.home.expanduser()
    profile: str = config.profile if config.profile is not None else home.name.lstrip(".")
    out: Path = config.out.expanduser() / profile
    out.mkdir(parents=True, exist_ok=True)
    manifest_path: Path = out / "manifest.json"
    manifest: Manifest = load_manifest(manifest_path)
    converted: int = 0
    skipped: int = 0
    failed: int = 0
    since: float | None = (
        datetime.combine(date.fromisoformat(config.since), datetime.min.time(), UTC).timestamp() if config.since is not None else None
    )
    for path in sorted((home / "projects").glob("*/*.jsonl")):
        if path.name.startswith("agent-"):
            continue
        if config.project is not None and config.project not in path.parent.name:
            continue
        if config.session_id is not None and config.session_id != path.stem:
            continue
        if since is not None and path.stat().st_mtime < since:
            continue
        started: float = perf_counter()
        digest = hashlib.sha256()
        for source in [path, *sorted((path.with_suffix("") / "subagents").glob("agent-*.jsonl"))]:
            with source.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
        source_hash: str = digest.hexdigest()
        session_id: str = path.stem
        entry: ManifestEntry | None = manifest.sessions.get(session_id)
        if entry is not None and entry.source_sha256 == source_hash and (out / entry.rrd).is_file():
            skipped += 1
            print(f"skipped {session_id} rows={entry.n_rows} seconds={perf_counter() - started:.3f}")
            continue
        try:
            session: ClaudeSession = parse_session(path)
        except ValueError as error:
            failed += 1
            print(f"FAILED {path}: {error} session_id={session_id} rows=0 seconds={perf_counter() - started:.3f}")
            continue
        session = replace(session, profile=profile)
        rrd: Path = write_session_rrd(session, out / f"{session_id}.rrd")
        n_rows: int = sum(
            chunk.to_record_batch().num_rows
            for chunk in RrdReader(rrd).stream().to_chunks()
            if not str(chunk.entity_path).lstrip("/").startswith("__properties")
        )
        manifest.sessions[session_id] = ManifestEntry(str(path.resolve()), source_hash, rrd.name, datetime.now(UTC).isoformat(), n_rows)
        save_manifest(manifest, manifest_path)
        converted += 1
        print(f"converted {session_id} rows={n_rows} seconds={perf_counter() - started:.3f}")
    print(f"converted={converted} skipped={skipped} failed={failed} out={out}")
