"""Stream Claude JSONL records into one typed session."""

import base64
import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import orjson
from serde import SerdeError, from_dict

from agent_traces import events as ev
from agent_traces.claude_records import (
    Block,
    CacheCreation,
    ImageBlock,
    ImageSource,
    OutputTokensDetails,
    Record,
    ResultContent,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from agent_traces.timestamps import parse_timestamp_ns as parse_timestamp_ns


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    """Typed record and uninterpreted source metadata."""

    file_index: int
    """Zero-based source line index."""
    record: Record
    """Decoded Claude record."""
    tool_use_result_json: str
    """Raw tool metadata JSON, empty when absent or null."""
    raw_json: str
    """Whole source line for system and attachment records."""


def iter_records(path: Path) -> Iterator[_SourceRecord]:
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
                yield _SourceRecord(file_index=line_number - 1, record=from_dict(Record, raw), tool_use_result_json=tool_use_result_json, raw_json=raw_json)
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


def session_sources(session_path: Path) -> list[Path]:
    """List the main transcript, sorted children, then sorted offloaded files."""
    session_dir: Path = session_path.with_suffix("")
    return [
        session_path,
        *sorted(path for path in (session_dir / "subagents").glob("agent-*.jsonl") if path.is_file()),
        *sorted(path for path in (session_dir / "tool-results").rglob("*") if path.is_file()),
    ]


def parse_session(session_path: Path) -> ev.Session:
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
    transcripts: dict[str, list[ev.TimedRecord]] = {}
    paths: dict[str, Path] = {"": source_path}
    paths.update(
        {
            path.stem.removeprefix("agent-"): path
            for path in session_sources(source_path)[1:]
            if path.parent == source_path.with_suffix("") / "subagents"
        }
    )
    for agent_id, path in paths.items():
        rows: list[tuple[_SourceRecord, int]] = []
        source_record: _SourceRecord
        for source_record in iter_records(path):
            file_index: int = source_record.file_index
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
            rows.append((replace(source_record, record=record), timestamp_ns))
        transcripts[agent_id] = interpret_records(rows)
    main: list[ev.TimedRecord] = transcripts.pop("")
    return ev.Session(
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


def tool_kind(name: str) -> ev.ToolKind:
    """Map Claude native names to the shared tool vocabulary."""
    if name.startswith("mcp__"):
        return "mcp"
    kinds: dict[str, ev.ToolKind] = {
        "Bash": "shell",
        "Read": "file_read",
        "Edit": "file_edit",
        "Write": "file_edit",
        "WebFetch": "web_search",
        "WebSearch": "web_search",
        "Agent": "subagent",
        "Workflow": "subagent",
    }
    return kinds.get(name, "other")


def tool_path(name: str) -> str:
    """Map a Claude MCP name to its server and tool entity path."""
    parts: list[str] = name.split("__", 2)
    return f"mcp/{parts[1]}/{parts[2]}" if len(parts) == 3 and parts[0] == "mcp" else name


def interpret_records(records: list[tuple[_SourceRecord, int]]) -> list[ev.TimedRecord]:
    """Interpret source blocks into flat events with transcript-local usage deduplication."""
    events: list[ev.TimedRecord] = []
    turn_id: str = ""
    calls: dict[str, tuple[int, str]] = {}
    seen: set[str] = set()
    for source_record, timestamp_ns in records:
        if source_record.record.message is not None:
            for block in source_record.record.message.content:
                if isinstance(block, ToolUseBlock):
                    calls[block.id] = (timestamp_ns, block.name)
    for source_record, timestamp_ns in records:
        record: Record = source_record.record
        values: dict[str, ev.Scalar] = {}
        payloads: list[tuple[ev.Payload, dict[str, ev.Scalar]]] = []
        if record.type == "system":
            payloads.append(
                (
                    ev.Lifecycle("system", record.content if record.content is not None else record.subtype, (record.level or "INFO").upper()),
                    {"subtype": record.subtype, "extra_json": source_record.raw_json},
                )
            )
        elif record.type == "attachment" and record.attachment is not None:
            attachment_json: str = orjson.dumps(orjson.loads(source_record.raw_json).get("attachment")).decode()
            description: str = (
                record.attachment.text
                or (record.attachment.content if isinstance(record.attachment.content, str) else "")
                or record.attachment.command
                or record.attachment.message
                or attachment_json
            )
            payloads.append(
                (
                    ev.Lifecycle("attachments", f"{record.attachment.type}: {description}"),
                    {"subtype": record.attachment.type, "attachment_json": attachment_json},
                )
            )
        elif record.type == "pr-link":
            payloads.append((ev.Lifecycle("pr_links", record.prUrl), {"pr_number": record.prNumber, "pr_repository": record.prRepository}))
        if record.message is not None:
            if record.type == "assistant" and record.message.id and record.message.id not in seen:
                usage: Usage = record.message.usage or Usage()
                cache: CacheCreation = usage.cache_creation or CacheCreation()
                details: OutputTokensDetails = usage.output_tokens_details or OutputTokensDetails()
                counters: ev.Usage = ev.Usage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_input_tokens,
                    cache_creation_tokens=usage.cache_creation_input_tokens,
                    cache_creation_5m_tokens=cache.ephemeral_5m_input_tokens,
                    cache_creation_1h_tokens=cache.ephemeral_1h_input_tokens,
                    thinking_tokens=details.thinking_tokens,
                )
                payloads.append((ev.UsageSample(counters), {}))
                seen.add(record.message.id)
            is_prompt: bool = (
                record.type == "user"
                and not record.isCompactSummary
                and not any(isinstance(block, ToolResultBlock) for block in record.message.content)
            )
            if is_prompt and any(isinstance(block, TextBlock) for block in record.message.content):
                turn_id = record.uuid
                payloads.insert(0, (ev.TurnBoundary("start"), {}))
            for block in record.message.content:
                sources: list[ImageSource] = []
                call_id: str = ""
                origin: Literal["tool_result", "user"] = "user"
                match block:
                    case ToolUseBlock():
                        payloads.append(
                            (ev.ToolCall(tool_path(block.name), block.id, orjson.dumps(block.input).decode(), tool_kind(block.name), block.name), {})
                        )
                    case ToolResultBlock():
                        call: tuple[int, str] | None = calls.get(block.tool_use_id)
                        name: str = call[1] if call else "unknown"
                        elapsed: float = (timestamp_ns - call[0]) / 1_000_000 if call else float("nan")
                        payloads.append(
                            (
                                ev.ToolResult(
                                    tool_path(name),
                                    block.tool_use_id,
                                    result_text(block),
                                    source_record.tool_use_result_json,
                                    tool_kind(name),
                                    elapsed,
                                    block.is_error,
                                    (record.toolUseResult.agentId or "") if record.toolUseResult else "",
                                ),
                                {},
                            )
                        )
                        sources = result_images(block)
                        call_id = block.tool_use_id
                        origin = "tool_result"
                    case ImageBlock(source=source) if record.type == "user" and source is not None:
                        sources = [source]
                    case TextBlock(text=text) | ThinkingBlock(thinking=text):
                        values = {"uuid": record.uuid, "parent_uuid": record.parentUuid or "", "prompt_id": record.promptId or ""}
                        if record.type == "assistant":
                            values.update(message_id=record.message.id, request_id=record.requestId or "", model=record.message.model)
                        payload: ev.Payload = (
                            ev.Thinking(text)
                            if isinstance(block, ThinkingBlock)
                            else (ev.Prompt(text, is_prompt, record.isCompactSummary) if record.type == "user" else ev.AssistantText(text))
                        )
                        payloads.append((payload, values))
                for source in sources:
                    if source.type == "base64":
                        payloads.append((ev.Image(base64.b64decode(source.data), source.media_type, call_id, origin), {}))
        for payload, values in payloads:
            events.append(
                ev.TimedRecord(
                    payload,
                    timestamp_ns,
                    source_record.file_index,
                    values,
                    turn_id=turn_id,
                    prompt_id=record.promptId or "",
                    message_id=record.message.id if record.type == "assistant" and record.message else "",
                    model=record.message.model if record.type == "assistant" and record.message else "",
                    effort=record.effort if record.type == "assistant" else "",
                )
            )
    return events
