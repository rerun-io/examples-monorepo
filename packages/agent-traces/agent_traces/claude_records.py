"""Partial third-party Claude records; unknown fields are allowed."""

from dataclasses import dataclass
from typing import TypeAlias

from serde import SerdeError, from_dict, serde
from serde import field as serde_field


@serde
@dataclass(frozen=True, slots=True)
class ImageSource:
    """An inline image source."""

    type: str = ""
    """Source encoding, normally base64."""
    media_type: str = ""
    """Image MIME type."""
    data: str = ""
    """Base64 payload."""


@serde
@dataclass(frozen=True, slots=True)
class ResultContent:
    """Text or image nested inside a tool result."""

    type: str = ""
    """Nested content kind."""
    text: str = ""
    """Text seen by the model."""
    source: ImageSource | None = None
    """Encoded image when present."""


@serde
@dataclass(frozen=True, slots=True)
class TextBlock:
    """Visible message text."""

    text: str = ""
    """Text seen by the model."""


@serde
@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """Assistant reasoning."""

    thinking: str = ""
    """Reasoning text."""


@serde
@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    """One tool invocation."""

    id: str = ""
    """Tool call identifier."""
    name: str = ""
    """Tool name."""
    input: dict[str, object] = serde_field(default_factory=dict, serializer=lambda value: value, deserializer=lambda value: value)
    """Tool-specific JSON arguments, preserved without interpretation."""


@serde
@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """One tool response."""

    tool_use_id: str = ""
    """Identifier of the corresponding tool call."""
    content: str | list[ResultContent] | None = None
    """Tool result content."""
    is_error: bool = False
    """Whether this result reports a tool error."""


@serde
@dataclass(frozen=True, slots=True)
class ImageBlock:
    """Direct message image."""

    source: ImageSource | None = None
    """Encoded image when present."""


@serde
@dataclass(frozen=True, slots=True)
class UnknownBlock:
    """An unmodeled content kind."""

    type: str = ""
    """Original tag, or empty when absent."""


Block: TypeAlias = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock | ImageBlock | UnknownBlock
BLOCK_TYPES: dict[str, type] = {
    "text": TextBlock,
    "thinking": ThinkingBlock,
    "tool_use": ToolUseBlock,
    "tool_result": ToolResultBlock,
    "image": ImageBlock,
}


def decode_blocks(value: object) -> list[Block]:
    """Normalize text and decode each known tag without union fallback."""
    if isinstance(value, str):
        return [TextBlock(text=value)]
    if not isinstance(value, list):
        raise SerdeError("message content must be a string or list")
    blocks: list[Block] = []
    for item in value:
        if not isinstance(item, dict):
            raise SerdeError("content block must be an object")
        tag: object = item.get("type", "")
        if not isinstance(tag, str):
            raise SerdeError("content block type must be a string")
        cls: type | None = BLOCK_TYPES.get(tag)
        blocks.append(from_dict(cls, item) if cls is not None else UnknownBlock(type=tag))
    return blocks


@serde
@dataclass(frozen=True, slots=True)
class CacheCreation:
    """Cache creation token counts."""

    ephemeral_5m_input_tokens: int = 0
    """Tokens cached for five minutes."""
    ephemeral_1h_input_tokens: int = 0
    """Tokens cached for one hour."""


@serde
@dataclass(frozen=True, slots=True)
class OutputTokensDetails:
    """Output token breakdown."""

    thinking_tokens: int = 0
    """Reasoning token count."""


@serde
@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage repeated across split assistant records."""

    input_tokens: int = 0
    """Input tokens."""
    output_tokens: int = 0
    """Output tokens."""
    cache_read_input_tokens: int = 0
    """Tokens read from cache."""
    cache_creation_input_tokens: int = 0
    """Tokens used to create cache entries."""
    cache_creation: CacheCreation | None = None
    """Cache duration breakdown."""
    output_tokens_details: OutputTokensDetails | None = None
    """Output token breakdown."""


@serde
@dataclass(frozen=True, slots=True)
class Message:
    """Message envelope shared by user and assistant records."""

    usage: Usage | None = None
    """API token counters."""
    model: str = ""
    """Assistant model name."""
    id: str = ""
    """Assistant message identifier."""
    content: list[Block] = serde_field(default_factory=list, deserializer=decode_blocks)
    """Normalized typed content blocks."""


@serde
@dataclass(frozen=True, slots=True)
class Attachment:
    """Partial lifecycle attachment."""

    type: str = ""
    """Attachment subtype."""
    text: str = ""
    """Text detail when available."""
    content: object = serde_field(default=None, serializer=lambda value: value, deserializer=lambda value: value)
    """Content detail: a string for hook output, a list or object for reminders and file attachments; kept uninterpreted."""
    command: str | None = None
    """Queued command text."""
    message: str | None = None
    """Status message when available."""


@serde
@dataclass(frozen=True, slots=True)
class ToolUseResult:
    """Only the tool-specific metadata consumed by the converter."""

    agentId: str | None = None
    """Child agent named by the Agent tool."""
    persistedOutputPath: str | None = None
    """Path to an offloaded tool response."""


@serde
@dataclass(frozen=True, slots=True)
class Record:
    """One JSONL record with optional version-dependent fields."""

    type: str = ""
    """Claude record kind."""
    timestamp: str | None = None
    """ISO-8601 wall-clock timestamp."""
    attachment: Attachment | None = None
    """Lifecycle attachment."""
    message: Message | None = None
    """User or assistant message."""
    toolUseResult: ToolUseResult | None = None
    """Typed tool metadata when it is an object."""
    uuid: str = ""
    """Record identifier."""
    parentUuid: str | None = None
    """Parent record identifier."""
    promptId: str | None = None
    """Prompt identifier."""
    requestId: str | None = None
    """API request identifier."""
    isCompactSummary: bool = False
    """Whether the user text is a compaction summary."""
    subtype: str = ""
    """System event subtype."""
    level: str | None = None
    """System severity."""
    content: str | None = None
    """System event text."""
    prNumber: int = 0
    """Linked pull request number."""
    prUrl: str = ""
    """Linked pull request URL."""
    prRepository: str = ""
    """Linked repository name."""
    cwd: str = ""
    """Working directory."""
    gitBranch: str = ""
    """Source branch."""
    version: str = ""
    """Claude CLI version."""
    customTitle: str = ""
    """User-assigned session title."""
    aiTitle: str = ""
    """Generated session title."""
    effort: str = ""
    """Reasoning effort for this assistant record."""
    totalCostUSD: float | int | None = None
    """Reported total session cost."""
