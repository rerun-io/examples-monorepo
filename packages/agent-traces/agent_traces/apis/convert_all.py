"""Convert a Claude or Codex home incrementally using a content-hash manifest."""

import hashlib
import os
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

import orjson
from rerun.chunk import RrdReader

from agent_traces import codex
from agent_traces.claude import parse_session, session_sources
from agent_traces.codex_records import SessionMeta
from agent_traces.events import Session
from agent_traces.manifest import Manifest, ManifestEntry, load_manifest, save_manifest
from agent_traces.rerun_log import write_session_rrd


@dataclass(frozen=True, slots=True)
class Config:
    """Batch conversion arguments."""

    home: Path
    """Claude or Codex home directory; a leading tilde is expanded."""
    out: Path
    """Required output directory."""
    profile: str | None = None
    """Override the home directory name without its leading dot."""
    host: str | None = None
    """Machine the sessions ran on, for transcripts copied from another host; defaults to this hostname."""
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
    is_codex: bool = not (home / "projects").is_dir() and ((home / "sessions").is_dir() or (home / "archived_sessions").is_dir())
    discovery_errors: dict[Path, str] = {}
    discovery_skips: dict[Path, str] = {}
    index: dict[Path, SessionMeta] = codex.discover_rollouts(home, discovery_errors, discovery_skips) if is_codex else {}
    for ordinal, error in enumerate(discovery_errors.values()):
        failed += 1
        print(f"FAILED rollout_index={ordinal}: {error}")
    thread_ids: set[str] = {meta.id or meta.session_id for meta in index.values()}
    reasons: Counter[str] = Counter(discovery_skips.values())
    skipped += len(discovery_skips)
    paths: list[Path] = sorted(index) if is_codex else sorted((home / "projects").glob("*/*.jsonl"))
    for path in paths:
        if path.name.startswith("agent-") or path.name.startswith("._"):  # subagent files; AppleDouble sidecars from macOS copies
            continue
        if config.project is not None and config.project not in path.parent.name:
            continue
        session_id: str = (index[path].id or index[path].session_id) if is_codex else path.stem
        if config.session_id is not None and config.session_id != session_id:
            continue
        if since is not None and path.stat().st_mtime < since:
            continue
        started: float = perf_counter()
        try:
            if is_codex:
                codex.check_version(index[path])
                if index[path].parent_thread_id in thread_ids:
                    raise codex.SkipRollout("folded-subagent")
            sources: list[Path] = codex.session_sources(path, index) if is_codex else session_sources(path)
        except codex.SkipRollout as error:
            skipped += 1
            reasons[str(error)] += 1
            continue
        except ValueError as error:
            failed += 1
            print(f"FAILED session_id={session_id}: {error}")
            continue
        digest = hashlib.sha256()
        for source in sources:
            digest.update(os.path.relpath(source, path.parent).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(source.stat().st_size).encode("ascii"))
            digest.update(b"\0")
            with source.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
        source_hash: str = digest.hexdigest()
        entry: ManifestEntry | None = manifest.sessions.get(session_id)
        if entry is not None and entry.source_sha256 == source_hash and (out / entry.rrd).is_file():
            skipped += 1
            reasons["unchanged"] += 1
            print(f"skipped {session_id} rows={entry.n_rows} seconds={perf_counter() - started:.3f}")
            continue
        try:
            session: Session = codex.parse_rollout(path, index) if is_codex else parse_session(path)
        except codex.SkipRollout as error:
            skipped += 1
            reasons[str(error)] += 1
            continue
        except ValueError as error:
            failed += 1
            label: str = f"session_id={session_id}" if is_codex else str(path)
            print(f"FAILED {label}: {error} session_id={session_id} rows=0 seconds={perf_counter() - started:.3f}")
            continue
        session = replace(session, profile=profile)
        rrd: Path = write_session_rrd(session, out / f"{session_id}.rrd", host=config.host)
        n_rows: int = sum(chunk.num_rows for chunk in RrdReader(rrd).stream() if not str(chunk.entity_path).lstrip("/").startswith("__properties"))
        manifest.sessions[session_id] = ManifestEntry(str(path.resolve()), source_hash, rrd.name, datetime.now(UTC).isoformat(), n_rows)
        save_manifest(manifest, manifest_path)
        converted += 1
        print(f"converted {session_id} rows={n_rows} seconds={perf_counter() - started:.3f}")
    if is_codex:
        print(f"skip_reasons={orjson.dumps(dict(sorted(reasons.items()))).decode()}")
    print(f"converted={converted} skipped={skipped} failed={failed} out={out}")
