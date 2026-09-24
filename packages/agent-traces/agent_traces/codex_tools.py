"""Correlate native Codex completions without assigning ambiguous calls."""

import re
from dataclasses import dataclass

import orjson
from serde import SerdeError, serde
from serde.json import from_json

from agent_traces import codex_records as cr
from agent_traces import events as ev


@dataclass(frozen=True, slots=True)
class RawCall:
    """Uninterpreted response call with turn correlation."""

    response: cr.ResponseItem
    """Raw tool call."""
    turn_id: str
    """Owning turn."""
    timestamp_ns: int
    """When the call was issued (envelope timestamp)."""


@dataclass(frozen=True, slots=True)
class CompletedTool:
    """A completed native item awaiting optional raw response enrichment."""

    item: cr.CommandExecution | cr.FileChange | cr.McpToolCall | cr.OtherTool
    """Typed tool item."""
    file_index: int
    """Completion source line."""
    started_ns: int
    """Execution start."""
    completed_ns: int
    """Execution completion."""
    turn_id: str
    """Owning turn."""
    model: str
    """Model in force."""
    effort: str
    """Effort in force."""
    item_json: str
    """Native result metadata without the full output text."""
    item_input: str
    """Native command or arguments, excluding output."""


def matching_call(completed: CompletedTool, calls: list[RawCall], outputs: dict[str, tuple[str, int]]) -> RawCall | None:
    """Prefer unique exact keys, then unique family/argument/interval matches."""
    item: cr.CommandExecution | cr.FileChange | cr.McpToolCall | cr.OtherTool = completed.item
    same_turn: list[RawCall] = [call for call in calls if call.turn_id == completed.turn_id]
    exact: list[RawCall] = [call for call in same_turn if item.id and item.id in {call.response.call_id, call.response.id}]
    if exact:
        return exact[0] if len(exact) == 1 else None
    candidates: list[RawCall] = []
    for call in same_turn:
        response: cr.ResponseItem = call.response
        output: tuple[str, int] | None = outputs.get(response.call_id)
        if output is None or not call.timestamp_ns <= completed.started_ns <= completed.completed_ns <= output[1]:
            continue
        raw: str = response.arguments or response.input
        name: str = response.name.removeprefix("functions.")
        matches: bool = False
        if isinstance(item, cr.CommandExecution) and item.command and name in {"exec", "exec_command", "shell", "shell_command"}:
            matches = names_command(raw, item)
        elif isinstance(item, cr.McpToolCall):
            matches = name in {f"mcp__{item.server}__{item.tool}", f"mcp/{item.server}/{item.tool}"} and cr.arguments_equal(raw, item.arguments)
        elif isinstance(item, cr.FileChange):
            matches = name == "apply_patch" and bool(item.changes) and all(path in raw for path in item.changes)
        elif isinstance(item, cr.OtherTool) and item.tool_kind == "image":
            matches = name == "view_image" and bool(item.path) and orjson.dumps(item.path).decode() in raw
        if matches:
            candidates.append(call)
    return candidates[0] if len(candidates) == 1 else None


def names_command(raw: str, item: cr.CommandExecution) -> bool:
    """Recognize shell arguments or a literal command inside an exec script."""
    try:
        arguments: ShellArguments = from_json(ShellArguments, raw)
    except (SerdeError, orjson.JSONDecodeError):
        # Scripts may use double, single, or template quotes. Compare entire
        # literals, never a short command substring inside an unrelated word.
        for token in re.findall(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`", raw, re.DOTALL):
            if token[1:-1] == item.command[-1] or token == orjson.dumps(item.command[-1]).decode():
                return True
        return raw == item.command[-1]
    command: str | list[str] = arguments.cmd or arguments.command
    directory: str = arguments.workdir or arguments.cwd
    return command in (item.command[-1], item.command) and not (directory and item.cwd and directory != item.cwd)


@serde
@dataclass(frozen=True, slots=True)
class ShellArguments:
    """Identifying fields in the older function-call shell layout."""

    cmd: str = ""
    """Shell command text."""
    command: str | list[str] = ""
    """Alternative command text or argument vector."""
    workdir: str = ""
    """Explicit execution directory."""
    cwd: str = ""
    """Alternative execution directory."""


def tool_events(completed: CompletedTool, calls: list[RawCall], outputs: dict[str, tuple[str, int]]) -> list[ev.TimedRecord]:
    """Emit native call and result events, enriched with matched raw data."""
    item: cr.CommandExecution | cr.FileChange | cr.McpToolCall | cr.OtherTool = completed.item
    matched: RawCall | None = matching_call(completed, calls, outputs)
    response: cr.ResponseItem | None = matched.response if matched is not None else None
    name: str
    kind: ev.ToolKind
    text: str = ""
    is_error: bool = False
    agent_id: str = ""
    if isinstance(item, cr.CommandExecution):
        spec: cr.ItemTag = cr.ITEM_TYPES["CommandExecution"]
        name, kind = spec.entity_name, spec.tool_kind
        text = item.aggregated_output or item.stdout + item.stderr or item.formatted_output
        is_error = item.status in {"failed", "declined", "cancelled"} or item.exit_code not in {None, 0}
    elif isinstance(item, cr.FileChange):
        spec = cr.ITEM_TYPES["FileChange"]
        name, kind = spec.entity_name, spec.tool_kind
        text = item.stdout + item.stderr
        is_error = item.status in {"failed", "declined", "cancelled"}
    elif isinstance(item, cr.McpToolCall):
        spec = cr.ITEM_TYPES["McpToolCall"]
        name, kind = f"{spec.entity_name}/{item.server}/{item.tool}", spec.tool_kind
        text = item.error.message if item.error else ""
        is_error = item.error is not None or item.status in {"failed", "declined", "cancelled"}
    else:
        name, kind = item.kind or item.entity_name, item.tool_kind
        agent_id = item.agent_thread_id
    input_json: str = (response.arguments or response.input) if response is not None else completed.item_input
    call_id: str = response.call_id if response is not None else item.id
    output: tuple[str, int] | None = outputs.get(call_id)
    raw_output: str = output[0] if output is not None and not isinstance(item, (cr.CommandExecution, cr.FileChange)) else completed.item_json
    paired: bool = matched is not None and output is not None
    started_ns: int = matched.timestamp_ns if paired and matched is not None else completed.started_ns
    completed_ns: int = output[1] if paired and output is not None else completed.completed_ns
    # Startup durations and returned wall_time_seconds often measure microseconds
    # even for real shell commands. Script/envelope spans include orchestration
    # and may contain several commands. None establishes per-command elapsed.
    elapsed: float = float("nan")
    return [
        ev.TimedRecord(
            ev.ToolCall(name, call_id, input_json, kind),
            started_ns,
            completed.file_index,
            turn_id=completed.turn_id,
            model=completed.model,
            effort=completed.effort,
        ),
        ev.TimedRecord(
            ev.ToolResult(name, call_id, text, raw_output, kind, elapsed, is_error, agent_id),
            completed_ns,
            completed.file_index,
            turn_id=completed.turn_id,
            model=completed.model,
            effort=completed.effort,
        ),
    ]

