"""Aggregate main-session records into typed turns in file order."""

from bisect import bisect_right
from dataclasses import dataclass, field, fields, replace

from agent_traces.events import AssistantText, Execution, Image, Prompt, Thinking, TimedRecord, ToolCall, ToolResult, TurnBoundary, Usage, UsageSample


@dataclass(frozen=True, slots=True)
class TurnTimeline:
    """Main prompt intervals used for temporal turn attribution."""

    starts: list[TimedRecord]
    """Start boundaries in stable wall order."""

    @classmethod
    def from_records(cls, records: list[TimedRecord]) -> "TurnTimeline":
        """Index emitted human-turn starts without mutating source records."""
        prompts: dict[str, TimedRecord] = {}
        for row in records:
            if isinstance(row.payload, Prompt) and row.payload.role == "human":
                prompts.setdefault(row.turn_id, row)
        return cls(sorted((replace(row, timestamp_ns=prompts[row.turn_id].timestamp_ns) if row.turn_id in prompts else row for row in records
                           if isinstance(row.payload, TurnBoundary) and row.payload.phase == "start"),
                          key=lambda row: row.timestamp_ns))

    def turn_id_at(self, timestamp_ns: int) -> str:
        """Return the turn containing wall time, or no turn before the first prompt."""
        index: int = bisect_right(self.starts, timestamp_ns, key=lambda row: row.timestamp_ns) - 1
        return self.starts[index].turn_id if index >= 0 else ""

    def assign(self, records: list[TimedRecord]) -> list[TimedRecord]:
        """Move activity into prompt intervals; preserve each boundary and human prompt's own identity."""
        return [row if isinstance(row.payload, TurnBoundary) or isinstance(row.payload, Prompt) and row.payload.role == "human"
                else replace(row, turn_id=self.turn_id_at(row.timestamp_ns)) for row in records]


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
    """Timestamp of the last assistant, thinking, tool or usage activity."""
    turn_id: str = ""
    """Provider turn identity."""
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
    @property
    def elapsed_ms(self) -> float:
        """Time from the prompt to its last qualifying activity."""
        return (self.last_timestamp_ns - self.timestamp_ns) / 1_000_000


def aggregate_turns(records: list[TimedRecord]) -> list[Turn]:
    """Fold events into the turn named by each event's typed identity."""
    turns: list[Turn] = []
    by_id: dict[str, Turn] = {}
    seen_messages: dict[str, set[str]] = {}
    timeline: TurnTimeline = TurnTimeline.from_records(records)
    starts: dict[str, TimedRecord] = {row.turn_id: row for row in timeline.starts}
    for boundary in records:
        if isinstance(boundary.payload, TurnBoundary) and boundary.payload.phase == "start":
            timed: TimedRecord = starts[boundary.turn_id]
            turn: Turn = Turn(timed.timestamp_ns, "", timed.prompt_id, len(turns), timed.file_index, timed.timestamp_ns, turn_id=timed.turn_id)
            turns.append(turn)
            by_id[timed.turn_id] = turn
            seen_messages[timed.turn_id] = set()
    for timed in timeline.assign(records):
        current: Turn | None = by_id.get(timed.turn_id)
        if current is None:
            continue
        if isinstance(timed.payload, (AssistantText, Thinking, ToolCall, ToolResult, UsageSample, Execution)):
            current.last_timestamp_ns = max(current.last_timestamp_ns, timed.timestamp_ns)
        current.model = current.model or timed.model
        current.effort = current.effort or timed.effort
        if timed.message_id and isinstance(timed.payload, (AssistantText, Thinking, ToolCall, UsageSample)):
            seen_messages[timed.turn_id].add(timed.message_id)
            current.n_assistant_messages = len(seen_messages[timed.turn_id])
        match timed.payload:
            case Prompt(role="human", text=text):
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
