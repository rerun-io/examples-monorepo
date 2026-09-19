"""Convert a Claude home incrementally using a content-hash manifest."""

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

from rerun.experimental import RrdReader

from agent_traces.claude import ClaudeSession, parse_session, session_sources
from agent_traces.manifest import Manifest, ManifestEntry, load_manifest, save_manifest
from agent_traces.rerun_log import write_session_rrd


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
        for source in session_sources(path):
            digest.update(source.relative_to(path.parent).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(source.stat().st_size).encode("ascii"))
            digest.update(b"\0")
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
            chunk.num_rows
            for chunk in RrdReader(rrd).stream()
            if not str(chunk.entity_path).lstrip("/").startswith("__properties")
        )
        manifest.sessions[session_id] = ManifestEntry(str(path.resolve()), source_hash, rrd.name, datetime.now(UTC).isoformat(), n_rows)
        save_manifest(manifest, manifest_path)
        converted += 1
        print(f"converted {session_id} rows={n_rows} seconds={perf_counter() - started:.3f}")
    print(f"converted={converted} skipped={skipped} failed={failed} out={out}")
