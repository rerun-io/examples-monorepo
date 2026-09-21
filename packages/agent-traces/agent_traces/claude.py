"""Stream Claude JSONL records into one typed session."""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import orjson
from serde import SerdeError, from_dict

from agent_traces.claude_records import Block, ImageSource, Record, ResultContent, ToolResultBlock


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Typed record and uninterpreted source metadata."""

    record: Record
    """Decoded Claude record."""
    tool_use_result_json: str
    """Raw tool metadata JSON, empty when absent or null."""
    raw_json: str
    """Whole source line for system and attachment records."""


@dataclass(frozen=True, slots=True)
class TimedRecord:
    """Record positioned on wall time and in its source file."""

    record: Record
    """Typed Claude record."""
    timestamp_ns: int
    """Nanoseconds since the Unix epoch."""
    file_index: int
    """Zero-based source line index."""
    tool_use_result_json: str
    """Raw tool metadata JSON, empty when absent or null."""
    raw_json: str
    """Whole source line for system and attachment records."""


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


def iter_records(path: Path) -> Iterator[SourceRecord]:
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
                raw_json: str = line.decode("utf-8").rstrip("\r\n") if raw.get("type") in {"system", "attachment"} else ""
                tool_metadata: object = raw.get("toolUseResult")
                tool_use_result_json: str = orjson.dumps(tool_metadata).decode() if tool_metadata is not None else ""
                if not isinstance(tool_metadata, dict):
                    raw["toolUseResult"] = None
                yield SourceRecord(record=from_dict(Record, raw), tool_use_result_json=tool_use_result_json, raw_json=raw_json)
            except (orjson.JSONDecodeError, SerdeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def result_text(block: ToolResultBlock) -> str:
    """Return the text the model saw in a tool response."""
    if isinstance(block.content, str):
        return block.content
    return "".join(part.text for part in block.content or [] if part.type == "text")


def result_images(block: ToolResultBlock) -> list[ImageSource]:
    """Return image sources from a tool response."""
    if not isinstance(block.content, list):
        return []
    return [part.source for part in block.content if part.type == "image" and part.source is not None]


def inline_offloaded_output(block: ToolResultBlock, persisted_path: str | None, tool_results_dir: Path) -> ToolResultBlock:
    """Expand a local tool response only within its session's output directory."""
    reference: str | None = persisted_path
    if not reference:
        marker: re.Match[str] | None = re.search(r"(?:[Oo]utput saved to|[Ss]aved to(?: file)?):?\s*([^\n]+)", result_text(block))
        reference = marker.group(1).strip(" `") if marker else None
    if not reference:
        return block
    candidate: Path = Path(reference)
    if not candidate.is_absolute():
        candidate = tool_results_dir / candidate
    candidate = candidate.resolve()
    try:
        is_output_file: bool = candidate.is_relative_to(tool_results_dir.resolve()) and candidate.is_file()
    except OSError:  # the marker matched prose, not a path (e.g. "saved to" followed by a paragraph): name too long
        return block
    if not is_output_file:
        return block
    full_text: str = candidate.read_text(encoding="utf-8", errors="replace")
    replacement: str | list[ResultContent] = (
        [ResultContent(type="text", text=full_text), *(part for part in block.content if part.type != "text")]
        if isinstance(block.content, list)
        else full_text
    )
    return replace(block, content=replacement)


KEPT_ATTACHMENTS: frozenset[str] = frozenset({"queued_command", "command_permissions", "hook_success", "edited_text_file", "auto_mode"})


TIMESTAMP_PATTERN: re.Pattern[str] = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:[.,]([0-9]{1,9}))?(Z|[+-][0-9]{2}:[0-9]{2})"
)


def parse_timestamp_ns(text: str) -> int:
    """Parse a zoned ISO timestamp with one to nine optional fractional digits."""
    matched: re.Match[str] | None = TIMESTAMP_PATTERN.fullmatch(text)
    if matched is None:
        raise ValueError(f"invalid timestamp: {text!r}")
    stamp: datetime = datetime(
        int(matched.group(1)),
        int(matched.group(2)),
        int(matched.group(3)),
        int(matched.group(4)),
        int(matched.group(5)),
        int(matched.group(6)),
    )
    zone: str = matched.group(8)
    offset_seconds: int = 0
    if zone != "Z":
        hours: int = int(zone[1:3])
        minutes: int = int(zone[4:6])
        if hours > 23 or minutes > 59:
            raise ValueError(f"invalid timestamp offset: {zone!r}")
        offset_seconds = (hours * 3600 + minutes * 60) * (1 if zone[0] == "+" else -1)
    delta: timedelta = stamp - datetime(1970, 1, 1)
    fraction_ns: int = int((matched.group(7) or "").ljust(9, "0"))
    return (delta.days * 86400 + delta.seconds - offset_seconds) * 1_000_000_000 + fraction_ns


def session_sources(session_path: Path) -> list[Path]:
    """List the main transcript, sorted children, then sorted offloaded files."""
    session_dir: Path = session_path.with_suffix("")
    return [
        session_path,
        *sorted(path for path in (session_dir / "subagents").glob("agent-*.jsonl") if path.is_file()),
        *sorted(path for path in (session_dir / "tool-results").rglob("*") if path.is_file()),
    ]


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
    paths.update({
        path.stem.removeprefix("agent-"): path
        for path in session_sources(source_path)[1:]
        if path.parent == source_path.with_suffix("") / "subagents"
    })
    for agent_id, path in paths.items():
        rows: list[TimedRecord] = []
        source_record: SourceRecord
        for file_index, source_record in enumerate(iter_records(path)):
            record: Record = source_record.record
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
            if record.message is not None:
                blocks: list[Block] = []
                persisted: str | None = record.toolUseResult.persistedOutputPath if record.toolUseResult else None
                block: Block
                for block in record.message.content:
                    match block:
                        case ToolResultBlock():
                            expanded: ToolResultBlock = inline_offloaded_output(block, persisted, tool_results_dir)
                            n_inlined_outputs += int(expanded is not block)
                            block = expanded
                    blocks.append(block)
                record = replace(record, message=replace(record.message, content=blocks))
            assert record.timestamp is not None
            try:
                timestamp_ns: int = parse_timestamp_ns(record.timestamp)
            except ValueError as error:
                raise ValueError(f"{path}:{file_index + 1}: {error}") from error
            rows.append(
                TimedRecord(
                    record=record,
                    timestamp_ns=timestamp_ns,
                    file_index=file_index,
                    tool_use_result_json=source_record.tool_use_result_json,
                    raw_json=source_record.raw_json,
                )
            )
        transcripts[agent_id] = rows
    main: list[TimedRecord] = transcripts.pop("")
    return ClaudeSession(
        session_id=source_path.stem,
        profile=source_path.parents[2].name.lstrip("."),
        source_path=source_path,
        main=main,
        subagents=transcripts,
        skipped=skipped,
        n_inlined_outputs=n_inlined_outputs,
        cwd=cwd,
        git_branch=git_branch,
        cli_versions=cli_versions,
        models=models,
        title=title,
        total_cost_usd=total_cost_usd,
        source_sha256=source_sha256,
    )
