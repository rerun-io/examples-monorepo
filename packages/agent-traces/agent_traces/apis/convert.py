"""Convert one Claude session to a recording."""

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import orjson

from agent_traces.claude import session_source
from agent_traces.events import Session
from agent_traces.rerun_log import WrittenRecording, write_session_rrd
from agent_traces.sources import SessionSource, fingerprint


@dataclass(frozen=True, slots=True)
class Config:
    """One-session conversion arguments."""

    session: Path
    """Main Claude transcript JSONL path."""
    out: Path
    """Destination RRD path."""
    profile: str | None = None
    """Override the profile inferred from the agent home directory."""
    host: str | None = None
    """Machine the sessions ran on, for transcripts copied from another host; defaults to this hostname."""


@dataclass(frozen=True, slots=True)
class Summary:
    """Outcome of one conversion, including failures before a recording exists."""

    failed: int = 0
    """Whether the input could not be converted."""

    @property
    def exit_code(self) -> int:
        """Map a failed conversion to process status one."""
        return int(self.failed > 0)


def main(config: Config) -> Summary:
    """Write a recording and print its entity and row counts.

    Args:
        config: Input path, output path, and optional profile override.
    """
    try:
        source: SessionSource = session_source(config.session)
        transcript_hash: str = fingerprint(source.inputs)
        session: Session = replace(source.parse(), source_sha256=transcript_hash)
    except (ValueError, OSError) as error:
        print(f"FAILED {config.session}: {error}")
        return Summary(failed=1)
    if config.profile is not None:
        session = replace(session, profile=config.profile)
    written: WrittenRecording = write_session_rrd(session, config.out, host=config.host)
    counts: Counter[str] = Counter(written.entity_rows)
    families: Counter[str] = Counter()
    for entity, rows in counts.items():
        family: str = entity.split("/", 1)[0]
        families[family] += rows
    print(f"entities={len(counts)} rows={sum(counts.values())}")
    for family, count in sorted(families.items()):
        print(f"{family}: {count}")
    print(f"n_turns={counts['turns']}")
    print(f"images={families['media']} inlined_outputs={session.properties.get('n_inlined_outputs', 0)}")
    print(f"skipped={orjson.dumps(session.skipped, option=orjson.OPT_SORT_KEYS).decode()}")
    print(written.path)
    return Summary()
