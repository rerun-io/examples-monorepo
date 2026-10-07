"""Convert a Claude home incrementally using a content-hash manifest."""

import socket
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

import orjson

from agent_traces import manifest as manifest_contract
from agent_traces.claude import discover
from agent_traces.events import Session
from agent_traces.manifest import (
    Manifest,
    ManifestEntry,
    load_manifest,
    manifest_lock,
    save_manifest,
)
from agent_traces.rerun_log import WrittenRecording, write_session_rrd
from agent_traces.sources import (
    Discovery,
    fingerprint,
)


@dataclass(frozen=True, slots=True)
class Config:
    """Batch conversion arguments."""

    home: Path
    """Claude home directory; a leading tilde is expanded."""
    out: Path
    """Required output directory."""
    profile: str | None = None
    """Override the home directory name without its leading dot."""
    host: str | None = None
    """Machine the sessions ran on, for transcripts copied from another host; defaults to this hostname."""
    project: str | None = None
    """Only sessions whose working directory contains this substring (Claude uses the project directory name)."""
    session_id: str | None = None
    """Only convert this session id."""
    since: str | None = None
    """Only main transcripts modified on or after this ISO date."""


@dataclass(frozen=True, slots=True)
class Summary:
    """Completed batch counts and process status."""

    converted: int
    """Recordings published."""
    skipped: int
    """Unchanged inputs."""
    failed: int
    """Sessions that failed conversion."""

    @property
    def exit_code(self) -> int:
        """Fail the process when any conversion failed."""
        return int(self.failed > 0)


def main(config: Config) -> Summary:
    """Convert sessions and save progress after each successful recording.

    Args:
        config: Input home, output root, and optional selection filters.
    """
    home: Path = config.home.expanduser()
    host: str = config.host if config.host is not None else socket.gethostname()
    profile: str = config.profile if config.profile is not None else home.name.lstrip(".")
    out: Path = config.out.expanduser() / profile
    try:
        discovery: Discovery = discover(home)
    except (ValueError, OSError) as error:
        print(f"FAILED {home}: {error}")
        return Summary(0, 0, 1)
    out.mkdir(parents=True, exist_ok=True)
    with manifest_lock(out):
        manifest_path: Path = out / "manifest.json"
        manifest: Manifest = load_manifest(manifest_path)
        converted: int = 0
        failed: int = 0
        since: float | None = (
            datetime.combine(date.fromisoformat(config.since), datetime.min.time(), UTC).timestamp() if config.since is not None else None
        )
        for path, error in discovery.failed.items():
            failed += 1
            print(f"FAILED {path}: {error}")
        reasons: Counter[str] = Counter()
        for source in discovery.sessions:
            path: Path = source.main
            session_id: str = source.session_id
            if config.project is not None and config.project not in source.project:
                continue
            if config.session_id is not None and config.session_id != session_id:
                continue
            if since is not None and path.stat().st_mtime < since:
                continue
            started: float = perf_counter()
            try:
                transcript_hash: str = fingerprint(source.inputs)
                entry: ManifestEntry | None = manifest.sessions.get(session_id)
                if (
                    entry is not None
                    and entry.source_sha256 == transcript_hash
                    and entry.host == host
                    and entry.revision == manifest_contract.CONVERSION_REVISION
                    and (out / entry.rrd).is_file()
                ):
                    reasons["unchanged"] += 1
                    print(f"skipped {session_id} rows={entry.n_rows} seconds={perf_counter() - started:.3f}")
                    continue
                session: Session = replace(source.parse(), source_sha256=transcript_hash, profile=profile)
                written: WrittenRecording = write_session_rrd(session, out / f"{session_id}.rrd", host=host)
                n_rows: int = sum(written.entity_rows.values())
                manifest.sessions[session_id] = ManifestEntry(
                    source_path=str(path), source_sha256=session.source_sha256, rrd=written.path.name,
                    converted_at=datetime.now(UTC).isoformat(), n_rows=n_rows, host=host, revision=manifest_contract.CONVERSION_REVISION,
                )
                save_manifest(manifest, manifest_path)
            except (ValueError, OSError, RuntimeError) as error:
                failed += 1
                print(f"FAILED {path}: {error} session_id={session_id} rows=0 seconds={perf_counter() - started:.3f}")
                continue
            converted += 1
            print(f"converted {session_id} rows={n_rows} seconds={perf_counter() - started:.3f}")
            del session  # Release this recording tree before parsing the next one.
        print(f"skip_reasons={orjson.dumps(dict(sorted(reasons.items()))).decode()}")
        print(f"converted={converted} skipped={sum(reasons.values())} failed={failed} out={out}")
        return Summary(converted, sum(reasons.values()), failed)
