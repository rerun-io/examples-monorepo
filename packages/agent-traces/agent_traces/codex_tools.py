"""Correlate native Codex completions without assigning ambiguous calls."""

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

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
            matches = (
                any(arguments.cmd == item.command[-1] and (not arguments.has_workdir or same_directory(arguments.workdir, item.cwd))
                    for arguments in script_commands(raw))
                if name == "exec" else names_command(raw, item)
            )
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
    """Recognize JSON shell arguments for native shell calls."""
    try:
        arguments: ShellArguments = from_json(ShellArguments, raw)
    except (SerdeError, orjson.JSONDecodeError):
        return False
    command: str | list[str] = arguments.cmd or arguments.command
    directory: str = arguments.workdir or arguments.cwd
    return command in (item.command[-1], item.command) and not (directory and item.cwd and not same_directory(directory, item.cwd))


def same_directory(requested: str, native_cwd: str) -> bool:
    """Compare a call's plain path with the item's cwd, which Codex records as a file:// URI."""
    native: str = unquote(urlparse(native_cwd).path) if native_cwd.startswith("file://") else native_cwd
    return bool(requested) and requested.rstrip("/") == native.rstrip("/")


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
    has_workdir: bool = False
    """A script explicitly supplied workdir, including an empty string."""


def script_commands(script: str) -> list[ShellArguments]:
    """Extract flat literal arguments from tools.exec_command calls, never quoted code.

    This deliberately supports only a small JavaScript grammar. Expressions,
    spreads, duplicate keys, interpolations and regex literals stay unmatched.
    """
    tokens: list[str] = [
        token for token in re.findall(
            r'''//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\[\s\S]|[^`\\])*`|[\w$]+|[^\s]''',
            script,
        ) if not token.startswith(("//", "/*"))
    ]
    # A slash outside strings/comments may start a regex; don't scan its contents.
    if any(token in {"/", '"', "'", "`"} for token in tokens):
        return []
    commands: list[ShellArguments] = []
    for start in range(len(tokens) - 5):
        if tokens[start:start + 5] != ["tools", ".", "exec_command", "(", "{"] or (start and tokens[start - 1] == "."):
            continue
        index: int = start + 5
        fields: dict[str, str | None] = {}
        while index + 2 < len(tokens):
            key: str = tokens[index]
            if key.startswith(('"', "'")):
                key = script_literal(key) or ""
            if not re.fullmatch(r"[A-Za-z_$][\w$]*", key) or key in fields or tokens[index + 1] != ":":
                break
            value: str = tokens[index + 2]
            fields[key] = script_literal(value)
            if fields[key] is None and not re.fullmatch(r"(?:\d+|true|false|null)", value):
                break
            index += 3
            if tokens[index:index + 2] == [",", "}"]:
                index += 1
            if tokens[index:index + 2] == ["}", ")"]:
                command: str | None = fields.get("cmd")
                directory: str | None = fields.get("workdir")
                if command is not None and ("workdir" not in fields or directory is not None):
                    commands.append(ShellArguments(cmd=command, workdir=directory or "", has_workdir="workdir" in fields))
                break
            if index >= len(tokens) or tokens[index] != ",":
                break
            index += 1
    return commands


def script_literal(token: str) -> str | None:
    """Decode supported JavaScript string escapes without evaluating any code."""
    if len(token) < 2 or token[0] not in {'"', "'", "`"} or token[-1] != token[0]:
        return None
    body: str = token[1:-1]
    if token[0] == "`" and "${" in body:
        return None
    escapes: dict[str, str] = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
    parts: list[str] = []
    index: int = 0
    while index < len(body):
        char: str = body[index]
        index += 1
        if char == "\\":
            if index == len(body):
                return None
            char = body[index]
            index += 1
            if char in {"x", "u"}:
                width: int = 2 if char == "x" else 4
                digits: str = body[index:index + width]
                if len(digits) != width or not re.fullmatch(r"[0-9a-fA-F]+", digits):
                    return None
                char = chr(int(digits, 16))
                index += width
            elif char in escapes:
                if char == "0" and index < len(body) and body[index].isdigit():
                    return None
                char = escapes[char]
            elif char not in {'"', "'", "`", "\\", "/"}:
                return None
        parts.append(char)
    return "".join(parts)


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
