"""Partial typed Codex rollout records; unknown fields remain allowed."""

from dataclasses import dataclass
from typing import TypeAlias

from serde import SerdeError, field, from_dict, serde


@serde
@dataclass(frozen=True, slots=True)
class Git:
    """Partial Git payload."""

    branch: str = ""
    """Source branch."""


@serde
@dataclass(frozen=True, slots=True)
class Instructions:
    """Initial instructions carried by session metadata."""

    text: str = ""
    """Full instruction text."""


@serde
@dataclass(frozen=True, slots=True)
class SessionMeta:
    """Partial SessionMeta payload."""

    id: str = ""
    """Thread identifier."""
    session_id: str = ""
    """Session identifier."""
    agent_role: str = ""
    """Provider child role when supplied."""
    agent_nickname: str = ""
    """Provider child nickname when supplied."""
    parent_thread_id: str | None = None
    """Parent thread identifier."""
    cli_version: str = ""
    """CLI version."""
    cwd: str = ""
    """Working directory."""
    originator: str = ""
    """Client origin."""
    thread_source: str = ""
    """Thread source label."""
    model_provider: str | None = None
    """Model provider."""
    forked_from_id: str | None = None
    """Fork origin."""
    base_instructions: Instructions | None = None
    """Initial model instructions."""
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
    encrypted_content: str = ""
    """Opaque inter-agent payload, represented only by its byte size."""


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


@serde
@dataclass(frozen=True, slots=True)
class FileChange:
    """Partial FileChange payload."""

    id: str = ""
    """Call identifier."""
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
    status: str = ""
    """Completion status."""
    error: Error | None = None
    """Tool error."""


@serde
@dataclass(frozen=True, slots=True)
class ContextCompaction:
    """A native context boundary, not a tool invocation."""


@serde
@dataclass(frozen=True, slots=True)
class OtherTool:
    """Partial OtherTool payload."""

    id: str = ""
    """Call identifier."""
    type: str = ""
    """Native tool category."""


Item: TypeAlias = MessageItem | ReasoningItem | CommandExecution | FileChange | McpToolCall | OtherTool | ContextCompaction | UnknownItem


@dataclass(frozen=True, slots=True)
class ItemTag:
    """The decoder and tool presentation share this native tag definition."""

    record_type: type
    """Partial native schema."""
    entity_name: str = ""
    """Execution category, empty for non-tools."""
    input_fields: tuple[str, ...] = ()
    """Native fields retained as execution input."""
    output_fields: tuple[str, ...] = ()
    """Output fields already rendered as execution text."""


ITEM_TYPES: dict[str, ItemTag] = {
    "UserMessage": ItemTag(MessageItem),
    "AgentMessage": ItemTag(MessageItem),
    "Reasoning": ItemTag(ReasoningItem),
    "ContextCompaction": ItemTag(ContextCompaction),
    "CommandExecution": ItemTag(CommandExecution, "command", ("command", "cwd"), ("stdout", "stderr", "aggregated_output", "formatted_output")),
    "FileChange": ItemTag(FileChange, "file_change", ("changes",), ("stdout", "stderr")),
    "McpToolCall": ItemTag(McpToolCall, "mcp", ("server", "tool", "arguments")),
    "WebSearch": ItemTag(OtherTool, "web_search", ("query", "action")),
    "Plan": ItemTag(OtherTool, "update_plan"),
    "SubAgentActivity": ItemTag(OtherTool, "subagent"),
    "ImageView": ItemTag(OtherTool, "image_view", ("path",)),
    "Extension": ItemTag(OtherTool, "extension"),
}


def decode_item(value: object) -> Item | None:
    """Decode known tags strictly and preserve unknown tags by name."""
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise SerdeError("item requires a string type")
    tag: str = value["type"]
    spec: ItemTag | None = ITEM_TYPES.get(tag)
    if spec is None:
        return UnknownItem(tag)
    return from_dict(spec.record_type, value)


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

    total_token_usage: TokenUsage | None = None
    """Cumulative thread counters."""
    last_token_usage: TokenUsage | None = None
    """Last response counters."""


@serde
@dataclass(frozen=True, slots=True)
class ThreadSettings:
    """Partial ThreadSettings payload."""

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

    effort: str | None = None
    """Reasoning effort in force, when supplied."""


@serde
@dataclass(frozen=True, slots=True)
class ResponseMetadata:
    """Partial ResponseMetadata payload."""

    turn_id: str = ""
    """Owning turn."""


@serde
@dataclass(frozen=True, slots=True)
class ResponseItem:
    """Identity and provenance shared by response variants."""

    id: str | None = None
    """Item identifier."""
    content: list[Content] | None = None
    """Text or inline images supplied with the item."""
    internal_chat_message_metadata_passthrough: ResponseMetadata | None = None
    """Turn correlation metadata."""


@serde
@dataclass(frozen=True, slots=True)
class Message(ResponseItem):
    """A conversation message from a model or its context."""

    role: str = ""
    """Message author role."""


@serde
@dataclass(frozen=True, slots=True)
class Reasoning(ResponseItem):
    """Readable summary or opaque encrypted reasoning."""

    summary: list[Content] = field(default_factory=list)
    """Readable reasoning summary."""
    encrypted_content: str | None = None
    """Opaque encrypted reasoning."""


@serde
@dataclass(frozen=True, slots=True)
class FunctionCall(ResponseItem):
    """A named function invocation with verbatim arguments."""

    call_id: str = ""
    """Call correlation identifier."""
    name: str = ""
    """Native tool name."""
    arguments: str = ""
    """Verbatim JSON arguments."""


@serde
@dataclass(frozen=True, slots=True)
class CustomToolCall(ResponseItem):
    """A named custom tool invocation with verbatim input."""

    call_id: str = ""
    """Call correlation identifier."""
    name: str = ""
    """Native tool name."""
    input: str = ""
    """Verbatim custom tool input."""


@serde
@dataclass(frozen=True, slots=True)
class ToolCallOutput(ResponseItem):
    """A model tool result joined to its invocation."""

    call_id: str = ""
    """Call correlation identifier."""
    output: object = field(default=None, serializer=lambda value: value, deserializer=lambda value: value)
    """Uninterpreted raw output."""


@serde
@dataclass(frozen=True, slots=True)
class InterAgentMessage(ResponseItem):
    """Message passed between agents."""

    author: str = ""
    """Inter-agent author."""
    recipient: str = ""
    """Inter-agent recipient."""


@serde
@dataclass(frozen=True, slots=True)
class UnknownResponse(ResponseItem):
    """Unmodeled response subtype retained in skip accounting."""

    type: str = ""
    """Unknown response tag."""


ModelCall: TypeAlias = FunctionCall | CustomToolCall
ModelTool: TypeAlias = ModelCall | ToolCallOutput
RESPONSE_TYPES: dict[str, type[ResponseItem]] = {
    "message": Message, "reasoning": Reasoning, "function_call": FunctionCall,
    "custom_tool_call": CustomToolCall, "function_call_output": ToolCallOutput,
    "custom_tool_call_output": ToolCallOutput, "agent_message": InterAgentMessage,
}


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
    """Turn provenance and image dependencies shared by event variants."""

    turn_id: str = ""
    """Owning turn."""
    local_images: list[str] = field(default_factory=list)
    """Local image dependencies."""


@serde
@dataclass(frozen=True, slots=True)
class TaskStarted(Event):
    """Start of an explicit turn."""


@serde
@dataclass(frozen=True, slots=True)
class TaskComplete(Event):
    """Completion of an explicit turn."""

    duration_ms: int | float | None = None
    """Task elapsed milliseconds."""


@serde
@dataclass(frozen=True, slots=True)
class ItemCompleted(Event):
    """One completed native item."""

    completed_at_ms: int | None = None
    """Item completion in Unix milliseconds."""
    item: Item | None = field(default=None, deserializer=decode_item)
    """Typed completed item."""


@serde
@dataclass(frozen=True, slots=True)
class TokenCount(Event):
    """Legacy usage snapshot."""

    info: TokenInfo | None = None
    """Legacy response usage."""


@serde
@dataclass(frozen=True, slots=True)
class UserMessage(Event):
    """Fallback user conversation text."""

    message: str = ""
    """Visible message text."""


@serde
@dataclass(frozen=True, slots=True)
class AgentMessage(Event):
    """Fallback assistant conversation text."""

    message: str = ""
    """Visible message text."""


@serde
@dataclass(frozen=True, slots=True)
class ThreadSettingsApplied(Event):
    """Updated settings for subsequent records."""

    thread_settings: ThreadSettings | None = None
    """Updated settings."""


@serde
@dataclass(frozen=True, slots=True)
class UnknownEvent(Event):
    """Unmodeled event subtype retained in skip accounting."""

    type: str = ""
    """Unknown event tag."""


EVENT_TYPES: dict[str, type[Event]] = {
    "task_started": TaskStarted, "task_complete": TaskComplete, "item_completed": ItemCompleted,
    "token_count": TokenCount, "user_message": UserMessage, "agent_message": AgentMessage,
    "thread_settings_applied": ThreadSettingsApplied,
}


@serde
@dataclass(frozen=True, slots=True)
class Compacted:
    """Compaction summary and the replacement conversation."""

    message: str = ""
    """Summary text when supplied directly."""
    replacement_history: list[ResponseItem] | None = None
    """Typed replacement messages, including the summary."""


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


Decoded: TypeAlias = SessionMeta | Context | ResponseItem | TokenRecord | Event | Compacted
PAYLOAD_TYPES: dict[str, type] = {
    "session_meta": SessionMeta, "turn_context": Context,
    "token_usage_record": TokenRecord, "compacted": Compacted,
}


def decode_payload(tag: str, payload: dict[str, object]) -> Decoded | None:
    """Own envelope and variant dispatch, preserving unknown native tags."""
    cls: type | None = PAYLOAD_TYPES.get(tag)
    if tag == "event_msg":
        cls = EVENT_TYPES.get(str(payload.get("type", "")), UnknownEvent)
    elif tag == "response_item":
        cls = RESPONSE_TYPES.get(str(payload.get("type", "")), UnknownResponse)
    return from_dict(cls, payload) if cls is not None else None
