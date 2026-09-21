"""Aggregate main-session records into typed turns in file order."""

from collections.abc import Sequence
from dataclasses import dataclass

from agent_traces.events import AssistantText, EventGroup, Image, Payload, Prompt, TimedRecord, ToolCall, TurnBoundary, UsageSample, neutral_records


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

    model: str = ""
    """First model in force in the turn."""
    effort: str = ""
    """First reasoning effort in force in the turn."""
    duration_ms: float | None = None
    """Explicit provider duration when available."""

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time from the prompt to the last record."""
        return self.duration_ms if self.duration_ms is not None else (self.last_timestamp_ns - self.timestamp_ns) / 1_000_000


def aggregate_turns(records: Sequence[TimedRecord | EventGroup]) -> list[Turn]:
    """Aggregate neutral events; explicit boundaries take precedence over prompts."""
    events: list[TimedRecord] = neutral_records(records)
    explicit: bool = any(isinstance(event.payload, TurnBoundary) for event in events)
    turns: list[Turn] = []
    by_id: dict[str, Turn] = {}
    seen: dict[int, set[str]] = {}
    seen_messages: dict[int, set[str]] = {}
    prompt_lines: dict[int, int] = {}
    for timed in events:
        payload: Payload = timed.payload
        if isinstance(payload, TurnBoundary) and payload.phase == "start":
            turn: Turn = Turn(timed.timestamp_ns, "", timed.turn_id, len(turns), timed.file_index, timed.timestamp_ns)
            turns.append(turn)
            by_id[timed.turn_id] = turn
        elif not explicit and isinstance(payload, Prompt) and payload.starts_turn:
            if not turns or prompt_lines.get(turns[-1].turn_index) != timed.file_index:
                turns.append(
                    Turn(timed.timestamp_ns, payload.text, str(timed.values.get("prompt_id", "")), len(turns), timed.file_index, timed.timestamp_ns)
                )
                prompt_lines[turns[-1].turn_index] = timed.file_index
            else:
                turns[-1].prompt += "\n" + payload.text
        if not turns:
            continue
        current: Turn | None = by_id.get(timed.turn_id) if explicit else turns[-1]
        if current is None:
            continue
        current.last_timestamp_ns = timed.timestamp_ns
        current.model = current.model or timed.model
        current.effort = current.effort or timed.effort
        if explicit and isinstance(payload, Prompt) and not payload.compaction:
            current.prompt += ("\n" if current.prompt else "") + payload.text
        if isinstance(payload, TurnBoundary) and payload.phase == "complete":
            current.duration_ms = payload.duration_ms if payload.duration_ms is not None else (timed.timestamp_ns - current.timestamp_ns) / 1_000_000
        elif isinstance(payload, ToolCall):
            current.n_tool_calls += 1
        elif isinstance(payload, Image):
            current.n_images += 1
        elif explicit and isinstance(payload, AssistantText):
            message_id: str = str(timed.values.get("message_id", timed.file_index))
            messages: set[str] = seen_messages.setdefault(current.turn_index, set())
            if message_id not in messages:
                messages.add(message_id)
                current.n_assistant_messages += 1
        elif isinstance(payload, UsageSample):
            identifiers: set[str] = seen.setdefault(current.turn_index, set())
            if payload.response_id in identifiers:
                continue
            identifiers.add(payload.response_id)
            current.n_assistant_messages += int(payload.count_message)
            current.input_tokens += payload.counters.get("input_tokens", 0)
            current.output_tokens += payload.counters.get("output_tokens", 0)
            current.cache_read_tokens += payload.counters.get("cache_read_tokens", 0)
            current.cache_creation_tokens += payload.counters.get("cache_creation_tokens", 0)
            current.thinking_tokens += payload.counters.get("thinking_tokens", 0)
    return turns
