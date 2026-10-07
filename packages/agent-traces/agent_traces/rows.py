"""Write typed agent records as deterministic Rerun columns."""

from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from typing import TypeVar, cast

import pyarrow as pa

from agent_traces.events import (
    AgentMetadata,
    AssistantText,
    Image,
    Lifecycle,
    Prompt,
    Scalar,
    Session,
    Thinking,
    TimedRecord,
    ToolCall,
    ToolResult,
    UsageSample,
)
from agent_traces.turns import Turn, TurnTimeline, aggregate_turns

TURN_TOKEN_KEYS: tuple[str, ...] = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens", "thinking_tokens")

FACT_TYPES: dict[str, pa.DataType] = {
    **dict.fromkeys(("agent_id", "turn_id", "call_id", "tool", "phase", "kind", "model", "effort", "input_json", "tool_use_result_json"), pa.string()),
    "is_error": pa.bool_(),
    "elapsed_ms": pa.float64(),
    **dict.fromkeys(("turn_index", "n_tool_calls", "n_assistant_messages", "n_images", *TURN_TOKEN_KEYS), pa.int64()),
}
"""Typed query facts, including turn totals; all other provenance stays in metadata_json."""

ROLE_COLORS: dict[str, int] = {"user": 0x8AB4F8FF, "assistant": 0xE8EAEDFF, "thinking": 0x9AA0A6FF, "compaction": 0xF5A623FF, "injected": 0xB5A1D8FF}
"""TextLog row colour (RGBA) per conversation entity, so roles read apart without the entity column."""
CALL_COLOR: int = 0xF9D67AFF
"""Tool call rows; result rows keep the neutral assistant colour."""


def darken(rgba: int) -> int:
    """Reduce each RGB channel to three quarters while retaining alpha."""
    return sum(((rgba >> shift & 255) * 3 // 4) << shift for shift in (8, 16, 24)) | (rgba & 255)


CHILD_COLORS: dict[int, int] = {color: darken(color) for color in {*ROLE_COLORS.values(), CALL_COLOR}}
"""One darker shade of each text role colour for child agents."""


@dataclass(frozen=True, slots=True)
class TextRow:
    """One text row and its flat metadata."""

    timestamp_ns: int
    """Wall timestamp in nanoseconds."""
    text: str
    """Full text without truncation."""
    level: str = "INFO"
    """Text severity."""
    values: dict[str, Scalar | None] = field(default_factory=dict)
    """Flat AnyValues components, including explicit null query facts."""
    color: int = ROLE_COLORS["assistant"]
    """RGBA row colour."""
    event: int = 0
    """Recording-wide event index; the builder first uses this as emission order."""


@dataclass(frozen=True, slots=True)
class ScalarRow:
    """One scalar measurement."""

    timestamp_ns: int
    """Wall timestamp in nanoseconds."""
    value: float
    """Measurement value."""
    values: dict[str, Scalar]
    """Typed provenance only; call_id also joins elapsed rows to tool results."""
    event: int = 0
    """Recording-wide event index; the builder first uses this as emission order."""


@dataclass(frozen=True, slots=True)
class ImageRow:
    """One encoded image with provenance."""

    timestamp_ns: int
    """Wall timestamp in nanoseconds."""
    blob: bytes
    """Decoded image file bytes."""
    media_type: str
    """Image MIME type."""
    values: dict[str, Scalar]
    """Typed provenance and provider metadata."""
    event: int = 0
    """Recording-wide event index; the builder first uses this as emission order."""


@dataclass(frozen=True, slots=True)
class AgentRow:
    """One recording-local child ordinal and its source identity."""

    n: int
    """Zero-based order by first timestamp, then ID; empty children sort last."""
    agent_id: str
    """Original provider child identity."""
    metadata: AgentMetadata
    """Optional provider labels."""


@dataclass(frozen=True, slots=True)
class Rows:
    """Pure temporal batches and recording metadata."""

    session_id: str
    """Recording identity."""
    name: str
    """Display name."""
    texts: dict[str, list[TextRow]]
    """TextLog rows by entity."""
    documents: dict[str, list[TextRow]]
    """Markdown documents by entity."""
    scalars: dict[str, list[ScalarRow]]
    """Finite scalar rows by entity."""
    series_names: dict[str, str]
    """Static legend labels for scalar entities."""
    images: dict[str, list[ImageRow]]
    """Encoded image rows by entity."""
    properties: dict[str, Scalar | None]
    """Provider and common recording facts."""
    skipped: dict[str, int]
    """Omitted input counts."""
    agents: list[AgentRow]
    """Child identity table for this recording."""


def scalar_series_name(entity: str) -> str:
    """Name each counter or tool and distinguish its main and pooled child series."""
    name: str = entity.removesuffix("/children").rsplit("/", 1)[-1]
    return f"{name} ({'children' if entity.endswith('/children') else 'main'})"


def provenance(timed: TimedRecord) -> dict[str, Scalar]:
    """Typed source identity shared by text, image and scalar rows."""
    return {"file_index": timed.file_index, "agent_id": timed.agent_id, **{
        key: identity for key in ("turn_id", "prompt_id", "message_id", "model", "effort") if (identity := getattr(timed, key))
    }}


def tool_call_row(timed: TimedRecord, call: ToolCall) -> TextRow:
    """Render a model invocation and its uninterpreted input."""
    return TextRow(timed.timestamp_ns, f"▶ {call.display_name or call.name}  {call.input_json}", color=CALL_COLOR,
                   values={**timed.extras, **provenance(timed), "call_id": call.call_id,
                           "tool": call.name, "phase": "call", "is_error": False, "input_json": call.input_json, "kind": call.kind,
                           "tool_use_result_json": None, "elapsed_ms": None})


def tool_result_row(timed: TimedRecord, result: ToolResult) -> TextRow:
    """Render full tool output once, with result metadata beside it."""
    elapsed: float | None = result.elapsed_ms
    label: str = f" {elapsed:.0f} ms" if elapsed is not None else ""
    return TextRow(timed.timestamp_ns, f"◀ {result.name}{label}  {result.text}", "ERROR" if result.is_error else "INFO",
                   {**timed.extras, **provenance(timed), "call_id": result.call_id,
                    "tool": result.name, "phase": "result", "tool_use_result_json": result.raw_json, "is_error": result.is_error,
                    "elapsed_ms": elapsed, "input_json": None, "child_agent_id": result.agent_id, "kind": result.kind})


def child_turn_ids(session: Session, timeline: TurnTimeline) -> dict[str, str]:
    """Resolve explicit spawn links, including children spawned by another child."""
    records: dict[str, list[TimedRecord]] = {"": session.main, **session.subagents}
    calls: dict[tuple[str, str], TimedRecord] = {(agent, row.payload.call_id): row for agent, batch in records.items()
                                               for row in batch if isinstance(row.payload, ToolCall)}
    links: dict[str, tuple[str, TimedRecord]] = {row.payload.agent_id: (agent, calls.get((agent, row.payload.call_id), row))
        for agent, batch in records.items() for row in batch if isinstance(row.payload, ToolResult) and row.payload.agent_id}

    def resolve(agent: str, visited: set[str]) -> str:
        """Follow parent links without looping on malformed source relationships."""
        if agent in visited or agent not in links:
            return ""
        parent, call = links[agent]
        return (resolve(parent, visited | {agent}) if parent else call.turn_id) or timeline.turn_id_at(call.timestamp_ns)

    return {agent: resolve(agent, set()) for agent in session.subagents if agent in links}


Row = TypeVar("Row", TextRow, ScalarRow, ImageRow)


def collect_rows(session: Session) -> Rows:
    """Collect provider-neutral rows without I/O or Rerun calls."""
    timeline: TurnTimeline = TurnTimeline.from_records(session.main)
    session = replace(session, main=timeline.assign(session.main))
    spawn_turns: dict[str, str] = child_turn_ids(session, timeline)
    texts: dict[str, list[TextRow]] = {}
    documents: dict[str, list[TextRow]] = {}
    scalars: dict[str, list[ScalarRow]] = {}
    images: dict[str, list[ImageRow]] = {}
    emission_order: int = 0
    def add(family: dict[str, list[Row]], entity: str, row: Row) -> None:  # noqa: UP047 - Runtime TypeVar annotations.
        """Stamp every emitted row once, including rows derived from completed turns."""
        nonlocal emission_order
        family.setdefault(entity, []).append(replace(row, event=emission_order))
        emission_order += 1

    child_ids: list[str] = sorted(session.subagents, key=lambda identity: (
        min((row.timestamp_ns for row in session.subagents[identity]), default=2**63 - 1), identity))
    agents: list[AgentRow] = [AgentRow(n, identity, session.agent_metadata.get(identity, AgentMetadata()))
                              for n, identity in enumerate(child_ids)]
    def add_conversation(timed: TimedRecord, text: str, role: str, agent_id: str) -> None:
        """Add a role-colored message and, for main messages, the current document."""
        row: TextRow = TextRow(timed.timestamp_ns, text, "DEBUG" if role == "thinking" else "INFO",
                               {**timed.extras, **provenance(timed)}, ROLE_COLORS[role])
        add(texts, f"conversation/{role}", row)
        if not agent_id and role != "thinking":
            stamp: str = datetime.fromtimestamp(timed.timestamp_ns / 1_000_000_000, UTC).isoformat()
            add(documents, "conversation/current", TextRow(timed.timestamp_ns, f"{role} | {stamp}\n\n{text}", values=row.values))

    n_tool_calls: int = 0
    for agent_id, records in {"": session.main, **session.subagents}.items():
        for timed in records:
            if agent_id:
                timed = replace(timed, agent_id=agent_id, turn_id=spawn_turns.get(agent_id) or timeline.turn_id_at(timed.timestamp_ns))
            match timed.payload:
                case Prompt(text=text, role=role):
                    add_conversation(timed, text, "user" if role == "human" else role, agent_id)
                case AssistantText(text=text):
                    add_conversation(timed, text, "assistant", agent_id)
                case Thinking(text=text):
                    add_conversation(timed, text, "thinking", agent_id)
                case Lifecycle() as lifecycle if lifecycle.name:
                    add(texts, f"lifecycle/{lifecycle.name}", TextRow(timed.timestamp_ns, lifecycle.text, lifecycle.level, {**timed.extras, **provenance(timed)}))
                case UsageSample(usage=usage):
                    for counter in fields(usage):
                        value: int | None = getattr(usage, counter.name)
                        if value is not None:
                            add(scalars, f"usage/{counter.name}", ScalarRow(timed.timestamp_ns, float(value), provenance(timed)))
                case ToolCall() as call:
                    add(texts, "tools", tool_call_row(timed, call))
                    n_tool_calls += 1
                case ToolResult() as result:
                    add(texts, "tools", tool_result_row(timed, result))
                    if result.elapsed_ms is not None:
                        add(scalars, f"elapsed/tools/{result.kind}", ScalarRow(timed.timestamp_ns, result.elapsed_ms, {**provenance(timed), "tool": result.name, "call_id": result.call_id}))
                case Image() as image:
                    add(images, "media/images", ImageRow(timed.timestamp_ns, image.blob, image.media_type,
                                 {**timed.extras, **provenance(timed), "call_id": image.call_id, "source": image.source}))
    turns: list[Turn] = aggregate_turns(session.main)
    for turn in turns:
        turn_provenance: dict[str, Scalar] = {"turn_id": turn.turn_id, "model": turn.model, "effort": turn.effort, "agent_id": "",
                      "prompt_id": turn.prompt_id, "file_index": turn.file_index}
        add(texts, "turns", TextRow(
                turn.timestamp_ns,
                turn.prompt,
                values={
                    **turn_provenance,
                    "turn_index": turn.turn_index,
                    "elapsed_ms": turn.elapsed_ms,
                    "n_tool_calls": turn.n_tool_calls,
                    "n_assistant_messages": turn.n_assistant_messages,
                    "n_images": turn.n_images,
                    **{key: getattr(turn.usage, key) or 0 for key in TURN_TOKEN_KEYS},
                },
                color=ROLE_COLORS["user"],
            ))
        add(scalars, "turns/elapsed_ms", ScalarRow(turn.timestamp_ns, turn.elapsed_ms, turn_provenance))
        add(scalars, "turns/output_tokens", ScalarRow(turn.timestamp_ns, float(turn.usage.output_tokens or 0), turn_provenance))
        add(scalars, "turns/tool_calls", ScalarRow(turn.timestamp_ns, float(turn.n_tool_calls), turn_provenance))
    ordinals: dict[str, int] = {agent.agent_id: agent.n for agent in agents}
    texts = {entity: [replace(row, text=f"[a{ordinals[str(row.values['agent_id'])]}] {row.text}", color=CHILD_COLORS[row.color])
                      if row.values.get("agent_id") else row for row in batch] for entity, batch in texts.items()}
    scalar_series: dict[str, list[ScalarRow]] = {}
    for entity, batch in scalars.items():
        for row in batch:
            target: str = f"{entity}/children" if row.values.get("agent_id") else entity
            scalar_series.setdefault(target, []).append(row)
    # Keep each row paired with its destination while sorting all temporal rows once.
    ordered = sorted(((row, cast(list[TextRow | ScalarRow | ImageRow], batch))
                      for family in (texts, documents, scalar_series, images) for batch in family.values() for row in batch),
                     key=lambda item: (item[0].timestamp_ns, item[0].event))
    for family in (texts, documents, scalar_series, images):
        for batch in family.values():
            batch.clear()
    for event, (row, destination) in enumerate(ordered):
        destination.append(replace(row, event=event))
    series_names: dict[str, str] = {entity: scalar_series_name(entity) for entity in scalar_series}
    properties: dict[str, Scalar | None] = {
        **session.properties, "session_id": session.session_id, "profile": session.profile, "agent": session.agent,
        "source_path": str(session.source_path), "source_sha256": session.source_sha256,
        "n_turns": len(turns), "n_subagents": len(session.subagents),
        "n_tool_calls": n_tool_calls,
        "n_images": sum(len(batch) for batch in images.values()),
    }
    name: str = f"{session.agent} {session.session_id[:8]} {session.properties.get('title') or session.properties.get('cwd', '')}"
    return Rows(session.session_id, name, texts, documents, scalar_series, series_names, images, properties, dict(session.skipped), agents)
