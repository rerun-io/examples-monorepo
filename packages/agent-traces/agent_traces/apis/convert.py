"""Convert one Claude or Codex session to a recording."""

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import orjson

from agent_traces.codex import SkipRollout
from agent_traces.events import Session
from agent_traces.manifest import fingerprint
from agent_traces.rerun_log import WrittenRecording, write_session_rrd
from agent_traces.sources import SessionSource, provider_for


@dataclass(frozen=True, slots=True)
class Config:
    """One-session conversion arguments."""

    session: Path
    """Main Claude transcript or Codex rollout JSONL path."""
    out: Path
    """Destination RRD path."""
    profile: str | None = None
    """Override the profile inferred from the agent home directory."""
    host: str | None = None
    """Machine the sessions ran on, for transcripts copied from another host; defaults to this hostname."""


def main(config: Config) -> None:
    """Write a recording and print its entity and row counts.

    Args:
        config: Input path, output path, and optional profile override.
    """
    try:
        source: SessionSource = provider_for(config.session.expanduser()).session_source(config.session)
        session: Session = replace(source.parse(), source_sha256=fingerprint(source.inputs))
    except SkipRollout as error:
        print(f"skipped reason={error}")
        return
    if config.profile is not None:
        session = replace(session, profile=config.profile)
    written: WrittenRecording = write_session_rrd(session, config.out, host=config.host)
    counts: Counter[str] = Counter(written.entity_rows)
    families: Counter[str] = Counter()
    for entity, rows in counts.items():
        parts: list[str] = entity.split("/")
        family: str = parts[2] if parts[0] == "agents" else parts[0]
        families[family] += rows
    print(f"entities={len(counts)} rows={sum(counts.values())}")
    for family, count in sorted(families.items()):
        print(f"{family}: {count}")
    print(f"n_turns={counts['turns']}")
    print(f"images={families['media']} inlined_outputs={session.n_inlined_outputs}")
    print(f"skipped={orjson.dumps(session.skipped, option=orjson.OPT_SORT_KEYS).decode()}")
    print(written.path)
