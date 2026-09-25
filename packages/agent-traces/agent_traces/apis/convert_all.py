"""Convert a Claude or Codex home incrementally using a content-hash manifest."""

import socket
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

import orjson

from agent_traces import manifest as manifest_contract
from agent_traces.codex import SkipRollout
from agent_traces.events import Session
from agent_traces.manifest import Manifest, ManifestEntry, fingerprint, fingerprint_with_extras, input_digest, load_manifest, save_manifest
from agent_traces.rerun_log import WrittenRecording, write_session_rrd
from agent_traces.sources import Discovery, provider_for


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
    host: str = config.host if config.host is not None else socket.gethostname()
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
    discovery: Discovery = provider_for(home).discover(home)
    for path, error in discovery.failed.items():
        failed += 1
        print(f"FAILED {path}: {error}")
    reasons: Counter[str] = Counter(discovery.skipped.values())
    skipped += len(discovery.skipped)
    for source in discovery.sessions:
        path: Path = source.main
        session_id: str = source.session_id
        if config.project is not None and config.project not in path.parent.name:
            continue
        if config.session_id is not None and config.session_id != session_id:
            continue
        if since is not None and path.stat().st_mtime < since:
            continue
        started: float = perf_counter()
        try:
            transcript_hash: str = fingerprint(source.inputs)
            entry: ManifestEntry | None = manifest.sessions.get(session_id)
            source_hash: str = fingerprint_with_extras(
                transcript_hash, {image: input_digest(Path(image)) for image in entry.extra_inputs} if entry is not None else {},
            )
            if (
                entry is not None
                and entry.source_sha256 == source_hash
                and entry.host == host
                and entry.revision == manifest_contract.CONVERSION_REVISION
                and (out / entry.rrd).is_file()
            ):
                skipped += 1 + len(source.folded)
                reasons["unchanged"] += 1
                reasons.update(["folded-subagent"] * len(source.folded))
                print(f"skipped {session_id} rows={entry.n_rows} seconds={perf_counter() - started:.3f}")
                continue
            session: Session = source.parse()
        except SkipRollout as error:
            skipped += 1 + len(source.folded)
            reasons[str(error)] += 1
            reasons.update(["parent-skipped"] * len(source.folded))
            continue
        except (ValueError, OSError) as error:
            failed += 1
            skipped += len(source.folded)
            reasons.update(["parent-failed"] * len(source.folded))
            print(f"FAILED {path}: {error} session_id={session_id} rows=0 seconds={perf_counter() - started:.3f}")
            continue
        source_hash = fingerprint_with_extras(transcript_hash, session.extra_inputs)
        session = replace(session, profile=profile, source_sha256=source_hash)
        written: WrittenRecording = write_session_rrd(session, out / f"{session_id}.rrd", host=host)
        n_rows: int = sum(written.entity_rows.values())
        manifest.sessions[session_id] = ManifestEntry(
            source_path=str(path), source_sha256=source_hash, rrd=written.path.name, extra_inputs=tuple(sorted(session.extra_inputs)),
            converted_at=datetime.now(UTC).isoformat(), n_rows=n_rows, host=host, revision=manifest_contract.CONVERSION_REVISION,
        )
        save_manifest(manifest, manifest_path)
        converted += 1
        skipped += len(source.folded)
        reasons.update(["folded-subagent"] * len(source.folded))
        print(f"converted {session_id} rows={n_rows} seconds={perf_counter() - started:.3f}")
        del session  # Release this recording tree before parsing the next one.
    print(f"skip_reasons={orjson.dumps(dict(sorted(reasons.items()))).decode()}")
    print(f"converted={converted} skipped={skipped} failed={failed} out={out}")
