"""Stream Claude JSONL records into one typed session."""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import orjson
from serde import SerdeError, from_dict

from agent_traces.claude_records import ContentBlock, Record, ResultContent


@dataclass(frozen=True, slots=True)
class TimedRecord:
    """Record positioned on wall time and in its source file."""

    record: Record
    """Typed Claude record."""
    timestamp_ns: int
    """Nanoseconds since the Unix epoch."""
    file_index: int
    """Zero-based source line index."""


@dataclass(frozen=True, slots=True)
class ClaudeSession:
    """One session and its child agent transcripts."""

    session_id: str
    """Main transcript filename stem."""
    profile: str
    """Claude home name without the leading dot."""
    source_path: Path
    """Absolute main transcript path."""
    main: list[TimedRecord]
    """Main transcript records in file order."""
    subagents: dict[str, list[TimedRecord]]
    """Child transcripts keyed by agent id."""
    skipped: dict[str, int]
    """Counts of records omitted from temporal data."""
    tool_results_dir: Path
    """Allowed directory for persisted tool output."""

    n_inlined_outputs: int = 0
    """Number of tool result blocks expanded from local files."""

    cwd: str = ""
    """Last recorded main-session working directory."""
    git_branch: str = ""
    """Last recorded main-session branch."""
    cli_versions: set[str] = field(default_factory=set)
    """CLI versions found across all transcripts."""
    models: set[str] = field(default_factory=set)
    """Assistant models found across all transcripts."""
    title: str = ""
    """Last title in main-file order."""
    total_cost_usd: float = float("nan")
    """Last reported main-session total cost."""
    source_sha256: str = ""
    """SHA-256 of the main JSONL bytes."""


def iter_records(path: Path) -> Iterator[Record]:
    """Decode records one line at a time through the typed boundary.

    Args:
        path: Source JSONL file.

    Yields:
        Typed records in source order, without reading ahead.

    Raises:
        ValueError: A line is malformed; the message names its file and line.
    """
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                decoded: object = orjson.loads(line)
                if not isinstance(decoded, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                raw: dict[str, object] = decoded
                if raw.get("type") == "system":
                    raw["raw_json"] = orjson.dumps(raw).decode()
                if raw.get("type") == "attachment":
                    raw["attachment_json"] = orjson.dumps(raw.get("attachment")).decode()
                raw["tool_use_result_json"] = orjson.dumps(raw.get("toolUseResult")).decode()
                if not isinstance(raw.get("toolUseResult"), dict):
                    raw["toolUseResult"] = None
                yield from_dict(Record, raw)
            except (orjson.JSONDecodeError, SerdeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


KEPT_ATTACHMENTS: frozenset[str] = frozenset({"queued_command", "command_permissions", "hook_success", "edited_text_file", "auto_mode"})


def parse_session(session_path: Path) -> ClaudeSession:
    """Read the main transcript and child files, counting omitted records.

    Args:
        session_path: Main JSONL path under a Claude home's projects directory.

    Returns:
        One typed session with child records and recording metadata.

    Raises:
        ValueError: A record or timestamp is malformed.
    """
    source_path: Path = session_path.resolve()
    tool_results_dir: Path = source_path.with_suffix("") / "tool-results"
    n_inlined_outputs: int = 0
    skipped: dict[str, int] = {}
    cwd: str = ""
    git_branch: str = ""
    cli_versions: set[str] = set()
    models: set[str] = set()
    title: str = ""
    total_cost_usd: float = float("nan")
    with source_path.open("rb") as source:
        source_sha256: str = hashlib.file_digest(source, "sha256").hexdigest()
    transcripts: dict[str, list[TimedRecord]] = {}
    paths: dict[str, Path] = {"": source_path}
    paths.update({path.stem.removeprefix("agent-"): path for path in sorted((source_path.with_suffix("") / "subagents").glob("agent-*.jsonl"))})
    for agent_id, path in paths.items():
        rows: list[TimedRecord] = []
        record: Record
        for file_index, record in enumerate(iter_records(path)):
            if record.version:
                cli_versions.add(record.version)
            if record.type == "assistant" and record.message is not None and record.message.model:
                models.add(record.message.model)
            if not agent_id:
                cwd = record.cwd or cwd
                git_branch = record.gitBranch or git_branch
                if record.type in {"custom-title", "ai-title"}:
                    title = record.customTitle or record.aiTitle or record.content or ""
                if record.type == "cost-state":
                    total_cost_usd = float(record.totalCostUSD) if record.totalCostUSD is not None else float("nan")
            kind: str = record.type
            keep: bool = kind in {"user", "assistant", "system", "pr-link"}
            if kind == "attachment" and record.attachment is not None:
                kind = record.attachment.type
                keep = kind in KEPT_ATTACHMENTS
            if record.timestamp is None or not keep:
                key: str = record.type if record.timestamp is None else kind
                skipped[key] = skipped.get(key, 0) + 1
                continue
            if record.message is not None and isinstance(record.message.content, list):
                blocks: list[ContentBlock] = []
                persisted: str | None = record.toolUseResult.persistedOutputPath if record.toolUseResult else None
                block: ContentBlock
                for block in record.message.content:
                    if block.type == "tool_result":
                        reference: str | None = persisted
                        if not reference:
                            text: str = (
                                block.content
                                if isinstance(block.content, str)
                                else "".join(part.text for part in block.content or [] if part.type == "text")
                            )
                            match: re.Match[str] | None = re.search(r"(?:[Oo]utput saved to|[Ss]aved to(?: file)?):?\s*([^\n]+)", text)
                            reference = match.group(1).strip(" `") if match else None
                        if reference:
                            candidate: Path = Path(reference)
                            if not candidate.is_absolute():
                                candidate = tool_results_dir / candidate
                            candidate = candidate.resolve()
                            if candidate.is_relative_to(tool_results_dir.resolve()) and candidate.is_file():
                                full_text: str = candidate.read_text()
                                replacement: str | list[ResultContent] = (
                                    [ResultContent(type="text", text=full_text), *(part for part in block.content if part.type != "text")]
                                    if isinstance(block.content, list)
                                    else full_text
                                )
                                block = replace(block, content=replacement)
                                n_inlined_outputs += 1
                    blocks.append(block)
                record = replace(record, message=replace(record.message, content=blocks))
            assert record.timestamp is not None
            try:
                stamp: datetime = datetime.fromisoformat(record.timestamp)
                if stamp.tzinfo is None:
                    raise ValueError("timestamp must include a timezone")
            except ValueError as error:
                raise ValueError(f"{path}:{file_index + 1}: {error}") from error
            delta: timedelta = stamp - datetime(1970, 1, 1, tzinfo=UTC)
            fraction: re.Match[str] | None = re.search(r"T\d{2}:\d{2}:\d{2}[.,](\d+)", record.timestamp)
            fractional_ns: int = int(fraction.group(1)[:9].ljust(9, "0")) if fraction else 0
            timestamp_ns: int = (delta.days * 86400 + delta.seconds) * 1_000_000_000 + fractional_ns
            rows.append(TimedRecord(record, timestamp_ns, file_index))
        transcripts[agent_id] = rows
    main: list[TimedRecord] = transcripts.pop("")
    return ClaudeSession(
        source_path.stem,
        source_path.parents[2].name.lstrip("."),
        source_path,
        main,
        transcripts,
        skipped,
        tool_results_dir,
        n_inlined_outputs,
        cwd,
        git_branch,
        cli_versions,
        models,
        title,
        total_cost_usd,
        source_sha256,
    )
