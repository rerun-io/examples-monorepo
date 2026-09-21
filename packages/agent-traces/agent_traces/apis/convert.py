"""Convert one Claude Code session to a recording."""

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import orjson
from rerun.chunk import RrdReader

from agent_traces.claude import ClaudeSession, parse_session
from agent_traces.rerun_log import write_session_rrd


@dataclass(frozen=True, slots=True)
class Config:
    """One-session conversion arguments."""

    session: Path
    """Main Claude Code session JSONL path."""
    out: Path
    """Destination RRD path."""
    profile: str | None = None
    """Override the profile inferred from the Claude home directory."""
    host: str | None = None
    """Machine the sessions ran on, for transcripts copied from another host; defaults to this hostname."""


def main(config: Config) -> None:
    """Write a recording and print its entity and row counts.

    Args:
        config: Input path, output path, and optional profile override.
    """
    session: ClaudeSession = parse_session(config.session)
    if config.profile is not None:
        session = replace(session, profile=config.profile)
    out: Path = write_session_rrd(session, config.out, host=config.host)
    counts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    for chunk in RrdReader(out).stream().to_chunks():
        entity: str = str(chunk.entity_path).lstrip("/")
        if entity.startswith("__properties"):
            continue
        rows: int = chunk.to_record_batch().num_rows
        counts[entity] += rows
        parts: list[str] = entity.split("/")
        family: str = parts[2] if parts[0] == "agents" else parts[0]
        families[family] += rows
    print(f"entities={len(counts)} rows={sum(counts.values())}")
    for family, count in sorted(families.items()):
        print(f"{family}: {count}")
    print(f"n_turns={counts['turns']}")
    print(f"images={families['media']} inlined_outputs={session.n_inlined_outputs}")
    print(f"skipped={orjson.dumps(session.skipped, option=orjson.OPT_SORT_KEYS).decode()}")
    print(out)
