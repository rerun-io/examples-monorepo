"""Aggregate main-session records into typed turns in file order."""

from dataclasses import dataclass, field, fields

from agent_traces.events import Image, Prompt, TimedRecord, ToolCall, TurnBoundary, Usage, UsageSample


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
    """Timestamp of the last emitted event in file order."""
    n_tool_calls: int = 0
    """Tool invocations in this turn."""
    n_assistant_messages: int = 0
    """Distinct nonempty assistant message identifiers."""
    n_images: int = 0
    """Images emitted by the writer."""
    usage: Usage = field(default_factory=Usage)
    """Sum of reported token counters."""

    model: str = ""
    """First model in force in the turn."""
    effort: str = ""
    """First reasoning effort in force in the turn."""
    duration_ms: float | None = None
    """Explicit provider duration when available."""

    @property
    def elapsed_ms(self) -> float:
        """Provider duration, or time from the prompt to its last emitted event."""
        return self.duration_ms if self.duration_ms is not None else (self.last_timestamp_ns - self.timestamp_ns) / 1_000_000


def aggregate_turns(records: list[TimedRecord]) -> list[Turn]:
    """Fold events into the turn named by each event's typed identity."""
    turns: list[Turn] = []
    by_id: dict[str, Turn] = {}
    seen_messages: dict[str, set[str]] = {}
    for timed in records:
        match timed.payload:
            case TurnBoundary(phase="start"):
                turn: Turn = Turn(timed.timestamp_ns, "", timed.prompt_id, len(turns), timed.file_index, timed.timestamp_ns)
                turns.append(turn)
                by_id[timed.turn_id] = turn
                seen_messages[timed.turn_id] = set()
        current: Turn | None = by_id.get(timed.turn_id)
        if current is None:
            continue
        current.last_timestamp_ns = timed.timestamp_ns
        current.model = current.model or timed.model
        current.effort = current.effort or timed.effort
        if timed.message_id:
            seen_messages[timed.turn_id].add(timed.message_id)
            current.n_assistant_messages = len(seen_messages[timed.turn_id])
        match timed.payload:
            case TurnBoundary(phase="complete", duration_ms=duration):
                current.duration_ms = duration if duration is not None else (timed.timestamp_ns - current.timestamp_ns) / 1_000_000
            case Prompt(human=True, text=text):
                current.prompt += ("\n" if current.prompt else "") + text
            case ToolCall():
                current.n_tool_calls += 1
            case Image():
                current.n_images += 1
            case UsageSample(usage=usage):
                totals: dict[str, int | None] = {}
                for counter in fields(Usage):
                    previous: int | None = getattr(current.usage, counter.name)
                    value: int | None = getattr(usage, counter.name)
                    totals[counter.name] = None if previous is None and value is None else (previous or 0) + (value or 0)
                current.usage = Usage(**totals)
    return turns
