"""Convert one Claude or Codex session to a recording."""

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import orjson
from rerun.chunk import RrdReader

from agent_traces.claude import parse_session
from agent_traces.codex import SkipRollout, parse_rollout
from agent_traces.events import Session
from agent_traces.rerun_log import write_session_rrd


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
    with config.session.open("rb") as source:
        first: object = orjson.loads(source.readline())
    codex: bool = isinstance(first, dict) and first.get("type") == "session_meta"
    try:
        session: Session = parse_rollout(config.session) if codex else parse_session(config.session)
    except SkipRollout as error:
        print(f"skipped reason={error}")
        return
    if config.profile is not None:
        session = replace(session, profile=config.profile)
    out: Path = write_session_rrd(session, config.out, host=config.host)
    counts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    for chunk in RrdReader(out).stream():
        entity: str = str(chunk.entity_path).lstrip("/")
        if entity.startswith("__properties"):
            continue
        rows: int = chunk.num_rows
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
