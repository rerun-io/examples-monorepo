"""Provider-neutral events at the parser, turn, and recording boundaries."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

Scalar: TypeAlias = str | int | float | bool
ToolKind: TypeAlias = Literal["shell", "file_read", "file_edit", "web_search", "mcp", "subagent", "plan", "image", "other"]


@dataclass(frozen=True, slots=True)
class Prompt:
    """One visible user text block."""

    text: str
    """Full text."""
    human: bool = True
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
    """A visible lifecycle row."""

    name: str
    """Entity suffix."""
    text: str = ""
    """Display text."""
    level: str = "INFO"
    """Severity."""


@dataclass(frozen=True, slots=True)
class Usage:
    """Provider token counters; None means the provider does not report a field."""

    input_tokens: int | None = None
    """Input tokens."""
    output_tokens: int | None = None
    """Output tokens."""
    cache_read_tokens: int | None = None
    """Tokens read from cache."""
    cache_creation_tokens: int | None = None
    """Tokens written to cache."""
    thinking_tokens: int | None = None
    """Reasoning output tokens."""
    cache_creation_5m_tokens: int | None = None
    """Tokens written to the five-minute cache."""
    cache_creation_1h_tokens: int | None = None
    """Tokens written to the one-hour cache."""


@dataclass(frozen=True, slots=True)
class UsageSample:
    """Per-response counters, deduplicated by the parser."""

    usage: Usage
    """Reported counters."""


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
    prompt_id: str = ""
    """Prompt identity carried by a turn start."""
    message_id: str = ""
    """Assistant message identity, independent of token usage."""
    model: str = ""
    """Model in force."""
    effort: str = ""
    """Reasoning effort in force."""


@dataclass(frozen=True, slots=True)
class Session:
    """Shared recording metadata and neutral main and child events."""

    session_id: str
    """Provider session identifier."""
    profile: str
    """Home name without its leading dot."""
    source_path: Path
    """Main transcript path."""
    main: list[TimedRecord]
    """Main neutral events."""
    subagents: dict[str, list[TimedRecord]]
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
    """Fingerprint of transcript inputs and consumed extra inputs."""
    extra_inputs: dict[str, str] = field(default_factory=dict)
    """Resolved local image paths and hashes of the bytes consumed (or missing)."""
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
