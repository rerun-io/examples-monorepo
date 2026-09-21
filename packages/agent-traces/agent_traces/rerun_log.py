"""Write typed agent records as deterministic Rerun columns."""

import os
import socket
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import rerun as rr

from agent_traces.blueprint import session_blueprint
from agent_traces.events import (
    AssistantText,
    Image,
    Lifecycle,
    Payload,
    Prompt,
    Scalar,
    Session,
    Thinking,
    TimedRecord,
    ToolCall,
    ToolResult,
    UsageSample,
    neutral_records,
)
from agent_traces.turns import Turn, aggregate_turns

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


def write_session_rrd(session: Session, out: Path, *, host: str | None = None) -> Path:
    """Save one session, including its subagents, to an RRD file.

    Args:
        session: Typed records from the parser.
        out: Destination recording path.
        host: Machine the session ran on; defaults to this machine's hostname.

    Returns:
        The output path after the recording has been flushed and closed.
    """
    texts: dict[str, list[TextRow]] = {}
    scalars: dict[str, list[ScalarRow]] = {}
    images: dict[str, list[ImageRow]] = {}
    for agent_id, records in {"": session.main, **session.subagents}.items():
        prefix: str = f"agents/{agent_id}/" if agent_id else ""
        timed: TimedRecord
        for timed in neutral_records(records):
            payload: Payload = timed.payload
            values: dict[str, Scalar] = {"file_index": timed.file_index, **timed.values}
            if isinstance(payload, (Prompt, AssistantText, Thinking)):
                name: str = (
                    "thinking"
                    if isinstance(payload, Thinking)
                    else ("assistant" if isinstance(payload, AssistantText) else ("compaction" if payload.compaction else "user"))
                )
                texts.setdefault(f"{prefix}conversation/{name}", []).append(
                    TextRow(timed.timestamp_ns, payload.text, "DEBUG" if isinstance(payload, Thinking) else "INFO", values, ROLE_COLORS[name])
                )
            elif isinstance(payload, Lifecycle) and payload.name:
                texts.setdefault(f"{prefix}lifecycle/{payload.name}", []).append(TextRow(timed.timestamp_ns, payload.text, payload.level, values))
            elif isinstance(payload, UsageSample) and payload.emit:
                for name, value in payload.counters.items():
                    scalars.setdefault(f"{prefix}usage/{name}", []).append(ScalarRow(timed.timestamp_ns, float(value), timed.file_index))
            elif isinstance(payload, ToolCall):
                values.update(tool_use_id=payload.call_id, phase="call", input_json=payload.input_json, kind=payload.kind)
                texts.setdefault(f"{prefix}tools/{payload.name}", []).append(
                    TextRow(timed.timestamp_ns, f"▶ {payload.display_name or payload.name}  {payload.input_json}", values=values, color=CALL_COLOR)
                )
            elif isinstance(payload, ToolResult):
                elapsed_label: str = f"{payload.elapsed_ms:.0f} ms" if payload.elapsed_ms == payload.elapsed_ms else "? ms"
                values.update(
                    tool_use_id=payload.call_id,
                    phase="result",
                    result_text=payload.text,
                    tool_use_result_json=payload.raw_json,
                    is_error=payload.is_error,
                    elapsed_ms=payload.elapsed_ms,
                    agent_id=payload.agent_id,
                    kind=payload.kind,
                )
                texts.setdefault(f"{prefix}tools/{payload.name}", []).append(
                    TextRow(timed.timestamp_ns, f"◀ {payload.name} {elapsed_label}  {payload.text}", "ERROR" if payload.is_error else "INFO", values)
                )
                scalars.setdefault(f"{prefix}tools/elapsed_ms/{payload.name}", []).append(
                    ScalarRow(timed.timestamp_ns, payload.elapsed_ms, timed.file_index)
                )
            elif isinstance(payload, Image):
                images.setdefault(f"{prefix}media/images", []).append(
                    ImageRow(timed.timestamp_ns, payload.blob, payload.media_type, timed.file_index, payload.call_id, payload.source)
                )
    turns: list[Turn] = aggregate_turns(session.main)
    for turn in turns:
        texts.setdefault("turns", []).append(
            TextRow(
                turn.timestamp_ns,
                turn.prompt,
                values={
                    "turn_index": turn.turn_index,
                    "model": turn.model,
                    "effort": turn.effort,
                    "prompt_id": turn.prompt_id,
                    "file_index": turn.file_index,
                    "elapsed_ms": turn.elapsed_ms,
                    "n_tool_calls": turn.n_tool_calls,
                    "n_assistant_messages": turn.n_assistant_messages,
                    "n_images": turn.n_images,
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                    "cache_read_tokens": turn.cache_read_tokens,
                    "cache_creation_tokens": turn.cache_creation_tokens,
                    "thinking_tokens": turn.thinking_tokens,
                },
                color=ROLE_COLORS["user"],
            )
        )
        scalars.setdefault("turns/elapsed_ms", []).append(ScalarRow(turn.timestamp_ns, turn.elapsed_ms, turn.file_index))
        scalars.setdefault("turns/output_tokens", []).append(ScalarRow(turn.timestamp_ns, float(turn.output_tokens), turn.file_index))
        scalars.setdefault("turns/tool_calls", []).append(ScalarRow(turn.timestamp_ns, float(turn.n_tool_calls), turn.file_index))
    out.parent.mkdir(parents=True, exist_ok=True)
    recording: rr.RecordingStream = rr.RecordingStream("agent_traces", recording_id=session.session_id)
    with tempfile.NamedTemporaryFile(dir=out.parent, prefix=out.name + ".", suffix=".tmp", delete=False) as temporary:
        temp_path: Path = Path(temporary.name)
    try:
        try:
            recording.save(temp_path, default_blueprint=session_blueprint())
            # Explicit Arrow lists preserve sparse rows and bypass AnyValues' global
            # type inference, which otherwise depends on the first tool encountered.
            for entity, rows in texts.items():
                rows.sort(key=lambda row: row.timestamp_ns)
                recording.send_columns(
                    entity,
                    # SDK numeric timestamps are seconds. datetime64 keeps integer ns exact.
                    indexes=[rr.TimeColumn("wall", timestamp=np.array([row.timestamp_ns for row in rows], dtype="datetime64[ns]"))],
                    columns=[
                        *rr.TextLog.columns(
                            text=[row.text for row in rows],
                            level=[row.level for row in rows],
                            color=np.array([row.color for row in rows], dtype=np.uint32),
                        ),
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
                    drop_untyped_nones=True,
                    session_id=session.session_id,
                    profile=session.profile,
                    agent=session.agent,
                    host=host if host is not None else socket.gethostname(),
                    cwd=session.cwd,
                    git_branch=session.git_branch,
                    cli_versions=",".join(sorted(session.cli_versions)),
                    title=session.title,
                    models=",".join(sorted(session.models)),
                    n_turns=len(turns),
                    n_subagents=len(session.subagents),
                    n_tool_calls=sum(row.values.get("phase") == "call" for rows in texts.values() for row in rows),
                    n_images=sum(len(rows) for rows in images.values()),
                    n_inlined_outputs=session.n_inlined_outputs,
                    total_cost_usd=pa.array([session.total_cost_usd], type=pa.float64()),
                    **(
                        {
                            "provider": session.provider,
                            "originator": session.originator,
                            "thread_source": session.thread_source,
                            "forked_from": session.forked_from,
                            "parent_thread": session.parent_thread,
                            "total_input_tokens": session.total_input_tokens,
                            "total_output_tokens": session.total_output_tokens,
                        }
                        if session.agent == "codex"
                        else {}
                    ),
                    source_path=str(session.source_path),
                    source_sha256=session.source_sha256,
                ),
            )
            if session.skipped:
                recording.send_property("skipped", rr.AnyValues(drop_untyped_nones=True, **session.skipped))
            recording.send_recording_name(f"{session.agent} {session.session_id[:8]} {session.title or session.cwd}")
            recording.flush()
        finally:
            recording.disconnect()
        os.replace(temp_path, out)
    finally:
        temp_path.unlink(missing_ok=True)
    return out
