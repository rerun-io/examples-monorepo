"""Provider-neutral events at the parser, turn, and recording boundaries."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

Scalar: TypeAlias = str | int | float | bool
ToolKind: TypeAlias = Literal["shell", "file_read", "file_edit", "web_search", "mcp", "subagent", "plan", "image", "other"]


@dataclass(frozen=True, slots=True)
class Prompt:
    """One visible user text block."""

    text: str
    """Full text."""
    starts_turn: bool = True
    """Whether this block is part of a human prompt."""
    compaction: bool = False
    """Whether this text is a compaction summary."""


@dataclass(frozen=True, slots=True)
class AssistantText:
    """One visible assistant text block."""

    text: str
    """Full text."""


@dataclass(frozen=True, slots=True)
class Thinking:
    """Reasoning text or an encrypted-content placeholder."""

    text: str
    """Display text; encrypted content is never decoded."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One native tool invocation."""

    name: str
    """Native tool name, with MCP names mapped to mcp/server/tool."""
    call_id: str
    """Provider call identifier."""
    input_json: str
    """Uninterpreted arguments."""
    kind: ToolKind
    """Shared tool category."""
    display_name: str = ""
    """Original name for display when it differs from the entity name."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One completed tool response."""

    name: str
    """Native tool entity suffix."""
    call_id: str
    """Provider call identifier."""
    text: str
    """Full result text."""
    raw_json: str
    """Uninterpreted provider result metadata."""
    kind: ToolKind
    """Shared tool category."""
    elapsed_ms: float
    """Elapsed execution time, NaN when unknown."""
    is_error: bool = False
    """Whether execution failed."""
    agent_id: str = ""
    """Child agent identifier if present."""


@dataclass(frozen=True, slots=True)
class Image:
    """Encoded image bytes."""

    blob: bytes
    """Decoded file bytes."""
    media_type: str
    """Image MIME type."""
    call_id: str = ""
    """Originating tool call, if any."""
    source: Literal["tool_result", "user"] = "user"
    """Image origin."""


@dataclass(frozen=True, slots=True)
class Lifecycle:
    """A lifecycle row or a silent source-record marker."""

    name: str = ""
    """Entity suffix; empty markers preserve file-order timing without a row."""
    text: str = ""
    """Display text."""
    level: str = "INFO"
    """Severity."""


@dataclass(frozen=True, slots=True)
class UsageSample:
    """Per-response counters, with parser-controlled deduplication."""

    counters: dict[str, int]
    """Counters keyed by shared usage entity suffix."""
    response_id: str
    """Response identifier used for turn-local deduplication."""
    count_message: bool = True
    """Whether this sample represents an assistant message (Claude)."""
    emit: bool = True
    """Whether this response is new across the whole transcript."""


@dataclass(frozen=True, slots=True)
class TurnBoundary:
    """Explicit task start or completion."""

    phase: Literal["start", "complete"]
    """Boundary side."""
    duration_ms: float | None = None
    """Authoritative elapsed time on completion."""


Payload: TypeAlias = Prompt | AssistantText | Thinking | ToolCall | ToolResult | Image | Lifecycle | UsageSample | TurnBoundary


@dataclass(frozen=True, slots=True)
class TimedRecord:
    """A neutral payload with source position and flat provenance."""

    payload: Payload
    """Provider-independent event."""
    timestamp_ns: int
    """Unix nanoseconds."""
    file_index: int
    """Zero-based source line."""
    values: dict[str, Scalar] = field(default_factory=dict)
    """Flat recording metadata."""
    turn_id: str = ""
    """Explicit turn identifier, when supplied by the provider."""
    model: str = ""
    """Model in force."""
    effort: str = ""
    """Reasoning effort in force."""


@runtime_checkable
class EventGroup(Protocol):
    """Compatibility boundary for a source record producing several events."""

    @property
    def events(self) -> list[TimedRecord]:
        """Neutral events interpreted by the parser."""
        ...


def neutral_records(records: Sequence[TimedRecord | EventGroup]) -> list[TimedRecord]:
    """Flatten source groups without interpreting provider records."""
    return [event for record in records for event in ([record] if isinstance(record, TimedRecord) else record.events)]


@dataclass(frozen=True, slots=True)
class Session:
    """Shared recording metadata and neutral main and child events."""

    session_id: str
    """Provider session identifier."""
    profile: str
    """Home name without its leading dot."""
    source_path: Path
    """Main transcript path."""
    main: Sequence[TimedRecord | EventGroup]
    """Main events or parser-owned source groups."""
    subagents: Mapping[str, Sequence[TimedRecord | EventGroup]]
    """Child events keyed by thread identifier."""
    skipped: dict[str, int]
    """Omitted records by reason."""
    agent: Literal["claude", "codex"] = "claude"
    """Provider."""
    n_inlined_outputs: int = 0
    """Expanded offloaded outputs."""
    cwd: str = ""
    """Working directory."""
    git_branch: str = ""
    """Source branch."""
    cli_versions: set[str] = field(default_factory=set)
    """Versions represented in this recording."""
    models: set[str] = field(default_factory=set)
    """Models represented in this recording."""
    title: str = ""
    """Session title."""
    total_cost_usd: float | None = None
    """Reported cost, absent for Codex."""
    source_sha256: str = ""
    """Main transcript hash."""
    provider: str = ""
    """Model provider."""
    originator: str = ""
    """Client originator."""
    thread_source: str = ""
    """Thread source."""
    forked_from: str = ""
    """Source thread for a fork."""
    parent_thread: str = ""
    """Parent thread, including orphaned subagents."""
    total_input_tokens: int = 0
    """Final provider cumulative input counter."""
    total_output_tokens: int = 0
    """Final provider cumulative output counter."""
