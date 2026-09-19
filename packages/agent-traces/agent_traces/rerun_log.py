"""Write typed agent records as deterministic Rerun columns."""

import base64
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import orjson
import pyarrow as pa
import rerun as rr

from agent_traces.blueprint import session_blueprint
from agent_traces.claude import ClaudeSession, TimedRecord
from agent_traces.claude_records import CacheCreation, ContentBlock, ImageSource, Message, OutputTokensDetails, Record, ResultContent, Usage

# TODO(codex): A Codex parser will target the same agent_traces record types.
Scalar: TypeAlias = str | int | float | bool
ROLE_COLORS: dict[str, int] = {"user": 0x8AB4F8FF, "assistant": 0xE8EAEDFF, "thinking": 0x9AA0A6FF, "compaction": 0xF5A623FF}
"""TextLog row colour (RGBA) per conversation entity, so roles read apart without the entity column."""
CALL_COLOR: int = 0xF9D67AFF
"""Tool call rows; result rows keep the neutral assistant colour."""


@dataclass(frozen=True, slots=True)
class TextRow:
    """One text row and its flat metadata."""

    timestamp_ns: int
    """Wall timestamp in nanoseconds."""
    text: str
    """Full text without truncation."""
    level: str = "INFO"
    """Text severity."""
    values: dict[str, Scalar] = field(default_factory=dict)
    """Flat AnyValues components."""
    color: int = ROLE_COLORS["assistant"]
    """RGBA row colour."""


@dataclass(frozen=True, slots=True)
class ScalarRow:
    """One scalar measurement."""

    timestamp_ns: int
    """Wall timestamp in nanoseconds."""
    value: float
    """Measurement value."""
    file_index: int
    """Zero-based source line index."""


@dataclass(frozen=True, slots=True)
class ImageRow:
    """One encoded image with provenance."""

    timestamp_ns: int
    """Wall timestamp in nanoseconds."""
    blob: bytes
    """Decoded image file bytes."""
    media_type: str
    """Image MIME type."""
    file_index: int
    """Zero-based source line index."""
    tool_use_id: str
    """Source tool call, or empty for user images."""
    source: Literal["tool_result", "user"]
    """Whether the image came from a tool_result or user."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A call used to name and time its result."""

    timestamp_ns: int
    """Call wall timestamp in nanoseconds."""
    name: str
    """Entity suffix of the tool."""


def tool_path(name: str) -> str:
    """Map an MCP tool name to its server and tool path.

    Args:
        name: Claude tool name, including any MCP prefix.

    Returns:
        Tool entity suffix, unchanged for non-MCP tools.
    """
    parts: list[str] = name.split("__", 2)
    return f"mcp/{parts[1]}/{parts[2]}" if len(parts) == 3 and parts[0] == "mcp" else name


def write_session_rrd(session: ClaudeSession, out: Path) -> Path:
    """Save one session, including its subagents, to an RRD file.

    Args:
        session: Typed records from the parser.
        out: Destination recording path.

    Returns:
        The output path after the recording has been flushed and closed.
    """
    texts: dict[str, list[TextRow]] = {}
    scalars: dict[str, list[ScalarRow]] = {}
    images: dict[str, list[ImageRow]] = {}
    for agent_id, records in {"": session.main, **session.subagents}.items():
        prefix: str = f"agents/{agent_id}/" if agent_id else ""
        seen_message_ids: set[str] = set()
        calls: dict[str, ToolCall] = {}
        for event in records:
            if event.record.message is not None and isinstance(event.record.message.content, list):
                for content in event.record.message.content:
                    if content.type == "tool_use":
                        calls[content.id] = ToolCall(event.timestamp_ns, tool_path(content.name))
        timed: TimedRecord
        for timed in records:
            record: Record = timed.record
            if record.type == "system":
                texts.setdefault(f"{prefix}lifecycle/system", []).append(
                    TextRow(
                        timed.timestamp_ns,
                        record.content if record.content is not None else record.subtype,
                        (record.level or "INFO").upper(),
                        {"subtype": record.subtype, "file_index": timed.file_index, "extra_json": record.raw_json},
                    )
                )
            elif record.type == "attachment" and record.attachment is not None:
                description: str = (
                    record.attachment.text
                    or record.attachment.content
                    or record.attachment.command
                    or record.attachment.message
                    or record.attachment_json
                )
                texts.setdefault(f"{prefix}lifecycle/attachments", []).append(
                    TextRow(
                        timed.timestamp_ns,
                        f"{record.attachment.type}: {description}",
                        values={"subtype": record.attachment.type, "file_index": timed.file_index, "attachment_json": record.attachment_json},
                    )
                )
            elif record.type == "pr-link":
                texts.setdefault(f"{prefix}lifecycle/pr_links", []).append(
                    TextRow(
                        timed.timestamp_ns,
                        record.prUrl,
                        values={"pr_number": record.prNumber, "pr_repository": record.prRepository, "file_index": timed.file_index},
                    )
                )
            message: Message | None = record.message
            if message is None:
                continue
            if record.type == "assistant" and message.id and message.id not in seen_message_ids:
                seen_message_ids.add(message.id)
                usage: Usage = message.usage or Usage()
                cache: CacheCreation = usage.cache_creation or CacheCreation()
                details: OutputTokensDetails = usage.output_tokens_details or OutputTokensDetails()
                counters: dict[str, int] = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_input_tokens,
                    "cache_creation_tokens": usage.cache_creation_input_tokens,
                    "cache_creation_5m_tokens": cache.ephemeral_5m_input_tokens,
                    "cache_creation_1h_tokens": cache.ephemeral_1h_input_tokens,
                    "thinking_tokens": details.thinking_tokens,
                }
                for name, value in counters.items():
                    scalars.setdefault(f"{prefix}usage/{name}", []).append(ScalarRow(timed.timestamp_ns, float(value), timed.file_index))
            blocks: list[ContentBlock] = [ContentBlock(type="text", text=message.content)] if isinstance(message.content, str) else message.content
            for block in blocks:
                if block.type == "tool_use":
                    input_json: str = orjson.dumps(block.input).decode()
                    texts.setdefault(f"{prefix}tools/{tool_path(block.name)}", []).append(
                        TextRow(
                            timed.timestamp_ns,
                            f"▶ {block.name}  {input_json}",
                            values={"tool_use_id": block.id, "phase": "call", "file_index": timed.file_index, "input_json": input_json},
                            color=CALL_COLOR,
                        )
                    )
                if block.type == "tool_result":
                    call: ToolCall | None = calls.get(block.tool_use_id)
                    tool_name: str = call.name if call else "unknown"
                    elapsed_ms: float = (timed.timestamp_ns - call.timestamp_ns) / 1_000_000 if call else float("nan")
                    result_text: str = (
                        block.content if isinstance(block.content, str) else "".join(part.text for part in block.content or [] if part.type == "text")
                    )
                    elapsed_label: str = f"{elapsed_ms:.0f} ms" if elapsed_ms == elapsed_ms else "? ms"
                    texts.setdefault(f"{prefix}tools/{tool_name}", []).append(
                        TextRow(
                            timed.timestamp_ns,
                            f"◀ {tool_name} {elapsed_label}  {result_text}",
                            "ERROR" if block.is_error else "INFO",
                            {
                                "tool_use_id": block.tool_use_id,
                                "phase": "result",
                                "file_index": timed.file_index,
                                "result_text": result_text,
                                "is_error": block.is_error,
                                "elapsed_ms": elapsed_ms,
                                "agent_id": record.toolUseResult.agentId or "" if record.toolUseResult else "",
                            },
                        )
                    )
                    scalars.setdefault(f"{prefix}tools/elapsed_ms/{tool_name}", []).append(
                        ScalarRow(timed.timestamp_ns, elapsed_ms, timed.file_index)
                    )
                image_parts: list[ResultContent] = []
                if block.type == "tool_result" and isinstance(block.content, list):
                    image_parts = block.content
                elif block.type == "image" and record.type == "user":
                    image_parts = [ResultContent(type="image", source=block.source)]
                for part in image_parts:
                    source: ImageSource | None = part.source
                    if part.type == "image" and source is not None and source.type == "base64":
                        images.setdefault(f"{prefix}media/images", []).append(
                            ImageRow(
                                timed.timestamp_ns,
                                base64.b64decode(source.data),
                                source.media_type,
                                timed.file_index,
                                block.tool_use_id,
                                "tool_result" if block.type == "tool_result" else "user",
                            )
                        )
                if block.type not in {"text", "thinking"}:
                    continue
                name: str = "thinking" if block.type == "thinking" else ("compaction" if record.isCompactSummary else record.type)
                values: dict[str, Scalar] = {
                    "file_index": timed.file_index,
                    "uuid": record.uuid,
                    "parent_uuid": record.parentUuid or "",
                    "prompt_id": record.promptId or "",
                }
                if record.type == "assistant":
                    values.update(message_id=message.id, request_id=record.requestId or "", model=message.model)
                texts.setdefault(f"{prefix}conversation/{name}", []).append(
                    TextRow(
                        timed.timestamp_ns,
                        block.thinking if block.type == "thinking" else block.text,
                        "DEBUG" if block.type == "thinking" else "INFO",
                        values,
                        ROLE_COLORS[name],
                    )
                )
    recording: rr.RecordingStream = rr.RecordingStream("agent_traces", recording_id=session.session_id)
    # Explicit Arrow lists preserve sparse rows and bypass AnyValues' global
    # type inference, which otherwise depends on the first tool encountered.
    for entity, rows in texts.items():
        rows.sort(key=lambda row: row.timestamp_ns)
        recording.send_columns(
            entity,
            # SDK numeric timestamps are seconds. datetime64 keeps integer ns exact.
            indexes=[rr.TimeColumn("wall", timestamp=np.array([row.timestamp_ns for row in rows], dtype="datetime64[ns]"))],
            columns=[
                *rr.TextLog.columns(text=[row.text for row in rows], level=[row.level for row in rows], color=np.array([row.color for row in rows], dtype=np.uint32)),
                *rr.AnyValues.columns(
                    drop_untyped_nones=True,
                    **{
                        key: pa.array([[row.values[key]] if key in row.values else [] for row in rows])
                        for key in sorted({key for row in rows for key in row.values})
                    },
                ),
            ],
            strict=True,
        )
    for entity, scalar_rows in scalars.items():
        scalar_rows.sort(key=lambda row: row.timestamp_ns)
        recording.send_columns(
            entity,
            indexes=[rr.TimeColumn("wall", timestamp=np.array([row.timestamp_ns for row in scalar_rows], dtype="datetime64[ns]"))],
            columns=[
                *rr.Scalars.columns(scalars=[row.value for row in scalar_rows]),
                *rr.AnyValues.columns(file_index=[row.file_index for row in scalar_rows]),
            ],
            strict=True,
        )
    for entity, image_rows in images.items():
        image_rows.sort(key=lambda row: row.timestamp_ns)
        recording.send_columns(
            entity,
            indexes=[rr.TimeColumn("wall", timestamp=np.array([row.timestamp_ns for row in image_rows], dtype="datetime64[ns]"))],
            columns=[
                *rr.EncodedImage.columns(blob=[row.blob for row in image_rows], media_type=[row.media_type for row in image_rows]),
                *rr.AnyValues.columns(
                    file_index=[row.file_index for row in image_rows],
                    tool_use_id=[row.tool_use_id for row in image_rows],
                    source=[row.source for row in image_rows],
                ),
            ],
            strict=True,
        )
    recording.send_property(
        "session",
        rr.AnyValues(
            session_id=session.session_id,
            profile=session.profile,
            host=socket.gethostname(),
            cwd=session.cwd,
            git_branch=session.git_branch,
            cli_versions=",".join(sorted(session.cli_versions)),
            title=session.title,
            models=",".join(sorted(session.models)),
            n_subagents=len(session.subagents),
            n_tool_calls=sum(row.values.get("phase") == "call" for rows in texts.values() for row in rows),
            n_images=sum(len(rows) for rows in images.values()),
            n_inlined_outputs=session.n_inlined_outputs,
            total_cost_usd=session.total_cost_usd,
            source_path=str(session.source_path),
            source_sha256=session.source_sha256,
        ),
    )
    if session.skipped:
        recording.send_property("skipped", rr.AnyValues(drop_untyped_nones=True, **session.skipped))
    recording.send_recording_name(f"claude {session.session_id[:8]} {session.title or session.cwd}")
    out.parent.mkdir(parents=True, exist_ok=True)
    recording.save(out, default_blueprint=session_blueprint())
    recording.flush()
    recording.disconnect()
    return out
