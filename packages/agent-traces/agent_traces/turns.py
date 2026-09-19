"""Aggregate main-session records into typed turns in file order."""

from dataclasses import dataclass

from agent_traces.claude import TimedRecord, result_images
from agent_traces.claude_records import (
    Block,
    ImageBlock,
    ImageSource,
    Message,
    OutputTokensDetails,
    Record,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


@dataclass(slots=True)
class Turn:
    """Mutable totals for one human prompt and the records that follow it."""

    timestamp_ns: int
    """Prompt wall timestamp in nanoseconds."""
    prompt: str
    """Joined prompt text blocks."""
    prompt_id: str
    """Prompt identifier, empty when absent."""
    turn_index: int
    """Zero-based turn index in file order."""
    file_index: int
    """Prompt source line index."""
    last_timestamp_ns: int
    """Timestamp of the last record in file order."""
    n_tool_calls: int = 0
    """Tool invocations in this turn."""
    n_assistant_messages: int = 0
    """Distinct nonempty assistant message identifiers."""
    n_images: int = 0
    """Images emitted by the writer."""
    input_tokens: int = 0
    """Input tokens, counted once per message identifier."""
    output_tokens: int = 0
    """Output tokens, counted once per message identifier."""
    cache_read_tokens: int = 0
    """Tokens read from cache."""
    cache_creation_tokens: int = 0
    """Tokens used to create cache entries."""
    thinking_tokens: int = 0
    """Reasoning output tokens."""

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time from the prompt to the last record."""
        return (self.last_timestamp_ns - self.timestamp_ns) / 1_000_000


def aggregate_turns(records: list[TimedRecord]) -> list[Turn]:
    """Accumulate main-session records between human prompts without mutation.

    Args:
        records: Main transcript records in file order, including any preamble.

    Returns:
        One turn per non-compaction user prompt without tool results.
    """
    turns: list[Turn] = []
    seen: set[str] = set()
    for timed in records:
        record: Record = timed.record
        message: Message | None = record.message
        blocks: list[Block] = message.content if message is not None else []
        if record.type == "user" and not record.isCompactSummary and not any(isinstance(block, ToolResultBlock) for block in blocks):
            prompt_texts: list[str] = [block.text for block in blocks if isinstance(block, TextBlock)]
            if prompt_texts:
                turns.append(Turn(timed.timestamp_ns, "\n".join(prompt_texts), record.promptId or "", len(turns), timed.file_index, timed.timestamp_ns))
                seen = set()
        if not turns:
            continue
        turn: Turn = turns[-1]
        turn.last_timestamp_ns = timed.timestamp_ns
        for block in blocks:
            match block:
                case ToolUseBlock():
                    turn.n_tool_calls += 1
                case ImageBlock(source=ImageSource(type="base64")) if record.type == "user":
                    turn.n_images += 1
                case ToolResultBlock():
                    turn.n_images += sum(1 for source in result_images(block) if source.type == "base64")
        if record.type == "assistant" and message is not None and message.id and message.id not in seen:
            seen.add(message.id)
            usage: Usage = message.usage or Usage()
            details: OutputTokensDetails = usage.output_tokens_details or OutputTokensDetails()
            turn.n_assistant_messages += 1
            turn.input_tokens += usage.input_tokens
            turn.output_tokens += usage.output_tokens
            turn.cache_read_tokens += usage.cache_read_input_tokens
            turn.cache_creation_tokens += usage.cache_creation_input_tokens
            turn.thinking_tokens += details.thinking_tokens
    return turns
