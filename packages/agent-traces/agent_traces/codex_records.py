"""Partial typed Codex rollout records; unknown fields remain allowed."""

from dataclasses import dataclass
from typing import TypeAlias

import orjson
from serde import SerdeError, field, from_dict, serde


@serde
@dataclass(frozen=True, slots=True)
class Git:
    """Partial Git payload."""

    branch: str = ""
    """Source branch."""


@serde
@dataclass(frozen=True, slots=True)
class SubagentSource:
    """Partial SubagentSource payload."""

    subagent: object = field(default=None, serializer=lambda value: value, deserializer=lambda value: value)
    """Subagent origin metadata."""


def decode_source(value: object) -> str | SubagentSource:
    """Decode the untagged source string or subagent object."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return from_dict(SubagentSource, value)
    raise SerdeError("source must be a string or object")


@serde
@dataclass(frozen=True, slots=True)
class SessionMeta:
    """Partial SessionMeta payload."""

    id: str = ""
    """Thread identifier."""
    session_id: str = ""
    """Session identifier."""
    parent_thread_id: str | None = None
    """Parent thread identifier."""
    cli_version: str = ""
    """CLI version."""
    cwd: str = ""
    """Working directory."""
    originator: str = ""
    """Client origin."""
    source: str | SubagentSource = field(default="", deserializer=decode_source)
    """Thread launch source."""
    thread_source: str = ""
    """Thread source label."""
    model_provider: str | None = None
    """Model provider."""
    forked_from_id: str | None = None
    """Fork origin."""
    git: Git | None = None
    """Git metadata."""


    @property
    def thread_id(self) -> str:
        """Canonical thread identity across metadata versions."""
        return self.id or self.session_id


@serde
@dataclass(frozen=True, slots=True)
class Content:
    """Partial Content payload."""

    type: str = ""
    """Content tag."""
    text: str = ""
    """Visible text."""
    image_url: str = ""
    """Inline image data URL."""


@serde
@dataclass(frozen=True, slots=True)
class Duration:
    """Partial Duration payload."""

    secs: int = 0
    """Whole seconds."""
    nanos: int = 0
    """Fractional nanoseconds."""


@serde
@dataclass(frozen=True, slots=True)
class Error:
    """Partial Error payload."""

    message: str = ""
    """Tool failure text."""


@serde
@dataclass(frozen=True, slots=True)
class UnknownItem:
    """Partial UnknownItem payload."""

    type: str = ""
    """Unknown native item tag."""


@serde
@dataclass(frozen=True, slots=True)
class MessageItem:
    """Partial MessageItem payload."""

    type: str = ""
    """Native message role tag."""

    id: str = ""
    """Item identifier."""
    content: list[Content] = field(default_factory=list)
    """Visible content."""
    phase: str | None = None
    """Assistant delivery phase."""


@serde
@dataclass(frozen=True, slots=True)
class ReasoningItem:
    """Partial ReasoningItem payload."""

    id: str = ""
    """Reasoning identifier."""


@serde
@dataclass(frozen=True, slots=True)
class CommandExecution:
    """Partial CommandExecution payload."""

    id: str = ""
    """Execution identifier."""
    command: list[str] = field(default_factory=list)
    """Native argument vector."""
    cwd: str = ""
    """Execution directory."""
    status: str = ""
    """Completion status."""
    stdout: str = ""
    """Standard output."""
    stderr: str = ""
    """Standard error."""
    aggregated_output: str = ""
    """Combined output."""
    formatted_output: str = ""
    """Rendered output."""
    exit_code: int | None = None
    """Process exit code."""
    duration: Duration | None = None
    """Execution duration."""


@serde
@dataclass(frozen=True, slots=True)
class Change:
    """Partial Change payload."""

    type: str = ""
    """Change kind."""
    unified_diff: str = ""
    """Patch text."""
    move_path: str | None = None
    """Rename destination."""


@serde
@dataclass(frozen=True, slots=True)
class FileChange:
    """Partial FileChange payload."""

    id: str = ""
    """Call identifier."""
    changes: dict[str, Change] = field(default_factory=dict)
    """Changes by path."""
    status: str = ""
    """Completion status."""
    stdout: str = ""
    """Standard output."""
    stderr: str = ""
    """Standard error."""


@serde
@dataclass(frozen=True, slots=True)
class McpToolCall:
    """Partial McpToolCall payload."""

    id: str = ""
    """Call identifier."""
    server: str = ""
    """MCP server."""
    tool: str = ""
    """MCP tool."""
    arguments: object = field(default=None, serializer=lambda value: value, deserializer=lambda value: value)
    """Uninterpreted tool arguments."""
    status: str = ""
    """Completion status."""
    error: Error | None = None
    """Tool error."""
    duration: Duration | None = None
    """Execution duration."""


@serde
@dataclass(frozen=True, slots=True)
class OtherTool:
    """Partial OtherTool payload."""

    id: str = ""
    """Call identifier."""
    type: str = ""
    """Native tool category."""
    kind: str = ""
    """Extension or subagent action."""
    agent_thread_id: str = ""
    """Target agent thread."""
    agent_path: str = ""
    """Agent path."""
    path: str = ""
    """Image path."""
    durationMs: int | float | None = None
    """Extension duration."""


Item: TypeAlias = MessageItem | ReasoningItem | CommandExecution | FileChange | McpToolCall | OtherTool | UnknownItem
ITEM_TYPES: dict[str, type] = {
    "UserMessage": MessageItem,
    "AgentMessage": MessageItem,
    "Reasoning": ReasoningItem,
    "CommandExecution": CommandExecution,
    "FileChange": FileChange,
    "McpToolCall": McpToolCall,
    **{name: OtherTool for name in ("WebSearch", "Plan", "SubAgentActivity", "ImageView", "ContextCompaction", "Extension")},
}


def decode_item(value: object) -> Item | None:
    """Decode known tags strictly and preserve unknown tags by name."""
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise SerdeError("item requires a string type")
    tag: str = value["type"]
    cls: type | None = ITEM_TYPES.get(tag)
    return from_dict(cls, value) if cls is not None else UnknownItem(tag)


@serde
@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Partial TokenUsage payload."""

    input_tokens: int = 0
    """Input tokens."""
    output_tokens: int = 0
    """Output tokens."""
    cached_input_tokens: int = 0
    """Cache read tokens."""
    cache_write_input_tokens: int = 0
    """Cache creation tokens."""
    reasoning_output_tokens: int = 0
    """Reasoning output tokens."""


@serde
@dataclass(frozen=True, slots=True)
class TokenInfo:
    """Partial TokenInfo payload."""

    last_token_usage: TokenUsage | None = None
    """Last response counters."""


@serde
@dataclass(frozen=True, slots=True)
class ThreadSettings:
    """Partial ThreadSettings payload."""

    model: str | None = None
    """Model in force."""
    reasoning_effort: str | None = None
    """Reasoning effort in force."""


@serde
@dataclass(frozen=True, slots=True)
class Context:
    """Partial Context payload."""

    turn_id: str = ""
    """Turn identifier."""
    model: str = ""
    """Model in force."""


@serde
@dataclass(frozen=True, slots=True)
class ResponseMetadata:
    """Partial ResponseMetadata payload."""

    turn_id: str = ""
    """Owning turn."""


@serde
@dataclass(frozen=True, slots=True)
class ResponseItem:
    """Partial ResponseItem payload."""

    type: str = ""
    """Response item tag."""
    id: str | None = None
    """Item identifier."""
    call_id: str = ""
    """Call correlation identifier."""
    name: str = ""
    """Native tool name."""
    arguments: str = ""
    """Verbatim JSON arguments."""
    input: str = ""
    """Verbatim custom tool input."""
    output: object = field(default=None, serializer=lambda value: value, deserializer=lambda value: value)
    """Uninterpreted raw output."""
    encrypted_content: str | None = None
    """Opaque encrypted reasoning."""
    content: list[Content] | None = None
    """Message content used only for images."""
    internal_chat_message_metadata_passthrough: ResponseMetadata | None = None
    """Turn correlation metadata."""


@serde
@dataclass(frozen=True, slots=True)
class TokenRecord:
    """Partial TokenRecord payload."""

    turn_id: str = ""
    """Owning turn."""
    response_id: str = ""
    """Usage deduplication identifier."""
    usage: TokenUsage = field(default_factory=TokenUsage)
    """Per-response counters."""
    thread_token_usage: TokenUsage | None = None
    """Final cumulative thread counters."""


@serde
@dataclass(frozen=True, slots=True)
class Event:
    """Partial Event payload."""

    type: str = ""
    """Event tag."""
    turn_id: str = ""
    """Owning turn."""
    started_at_ms: int | None = None
    """Item start in Unix milliseconds."""
    completed_at_ms: int | None = None
    """Item completion in Unix milliseconds."""
    duration_ms: int | float | None = None
    """Task elapsed milliseconds."""
    item: Item | None = field(default=None, deserializer=decode_item)
    """Typed completed item."""
    info: TokenInfo | None = None
    """Legacy response usage."""
    thread_settings: ThreadSettings | None = None
    """Updated settings."""
    local_images: list[str] = field(default_factory=list)
    """Local image dependencies."""


@serde
@dataclass(frozen=True, slots=True)
class Envelope:
    """Line envelope decoded before dispatching its payload."""

    timestamp: str
    """ISO wall timestamp."""
    type: str
    """Envelope tag."""
    payload: dict[str, object] = field(serializer=lambda value: value, deserializer=lambda value: value)
    """Raw payload, consumed only at the decoder boundary."""


def arguments_equal(raw: str, arguments: object) -> bool:
    """Compare opaque tool arguments at the JSON boundary, retaining raw bytes."""
    try:
        decoded: object = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return False
    return decoded == arguments
