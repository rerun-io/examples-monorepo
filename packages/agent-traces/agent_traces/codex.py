"""Stream Codex rollouts into provider-neutral recording events."""

import base64
import binascii
import mimetypes
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias

import orjson
from serde import SerdeError, from_dict

from agent_traces import claude
from agent_traces import codex_records as cr
from agent_traces import events as ev
from agent_traces.sources import Discovery, SessionSource, provider_for
from agent_traces.timestamps import parse_timestamp_ns

Decoded: TypeAlias = cr.SessionMeta | cr.Context | cr.ResponseItem | cr.TokenRecord | cr.Event
PAYLOAD_TYPES: dict[str, type] = {
    "session_meta": cr.SessionMeta,
    "turn_context": cr.Context,
    "response_item": cr.ResponseItem,
    "token_usage_record": cr.TokenRecord,
    "event_msg": cr.Event,
}


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Typed payload and line provenance."""

    payload: Decoded | None
    """Decoded known payload."""
    tag: str
    """Envelope type."""
    timestamp_ns: int
    """Envelope Unix nanoseconds."""
    file_index: int
    """Zero-based source line."""
    item_json: str = ""
    """Uninterpreted native item details for tool-specific fields."""


def iter_rollout(path: Path) -> Iterator[SourceRecord]:
    """Decode one line at a time; diagnostics never include source content."""
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            try:
                raw: object = orjson.loads(line)
                if not isinstance(raw, dict):
                    raise SerdeError("expected an object")
                envelope: cr.Envelope = from_dict(cr.Envelope, raw)
                cls: type | None = PAYLOAD_TYPES.get(envelope.type)
                payload: Decoded | None = from_dict(cls, envelope.payload) if cls is not None else None
                timestamp: int = parse_timestamp_ns(envelope.timestamp)
            except (orjson.JSONDecodeError, SerdeError, ValueError) as error:
                raise ValueError(f"line={index + 1} invalid rollout structure ({type(error).__name__})") from None
            item_json: str = orjson.dumps(envelope.payload["item"]).decode() if envelope.type == "event_msg" and "item" in envelope.payload else ""
            yield SourceRecord(payload, envelope.type, timestamp, index, item_json)


class SkipRollout(ValueError):
    """A rollout excluded by the supported-history policy."""


def rollout_meta(path: Path) -> cr.SessionMeta:
    """Read only the first line, without touching legacy event schemas."""
    with path.open("rb") as stream:
        try:
            raw: object = orjson.loads(stream.readline())
        except orjson.JSONDecodeError:
            raise ValueError("line=1 invalid rollout structure (JSONDecodeError)") from None
    if isinstance(raw, dict) and "type" not in raw and "id" in raw and "timestamp" in raw:
        raise SkipRollout("no-item_completed")
    first: SourceRecord | None = next(iter_rollout(path), None)
    if first is None or not isinstance(first.payload, cr.SessionMeta):
        raise ValueError("missing session_meta")
    return first.payload


def check_version(meta: cr.SessionMeta) -> None:
    """Apply the CLI version floor before decoding any events."""
    version: re.Match[str] | None = re.match(r"^(\d+)\.(\d+)(?:\.|$)", meta.cli_version)
    if version is None:
        raise ValueError("invalid cli_version")
    if (int(version[1]), int(version[2])) < (0, 150):
        raise SkipRollout(f"codex-cli-{meta.cli_version}")


def parse_file(path: Path, images: dict[str, Path]) -> ev.Session:
    """Interpret authoritative completion events without replaying response text."""
    check_version(rollout_meta(path))
    completed_items: int = 0
    has_usage_records: bool = False
    seen_responses: set[str] = set()
    seen_legacy: set[tuple[str, ev.Usage]] = set()
    legacy_indices: set[int] = set()
    total: cr.TokenUsage = cr.TokenUsage()
    raw_calls: list[RawCall] = []
    outputs: dict[str, tuple[str, int]] = {}
    pending_tools: list[CompletedTool] = []
    events: list[ev.TimedRecord] = []
    skipped: dict[str, int] = {}
    meta: cr.SessionMeta | None = None
    model: str = ""
    effort: str = ""
    turn_id: str = ""
    models: set[str] = set()
    turn_models: dict[str, str] = {}
    reasoning: dict[str, list[int]] = defaultdict(list)
    reasoning_items: dict[str, list[int]] = defaultdict(list)
    for source in iter_rollout(path):
        payload: Decoded | None = source.payload
        match payload:
            case cr.SessionMeta():
                meta = payload
            case cr.Context():
                turn_id = payload.turn_id or turn_id
                model = payload.model or model
                turn_models[turn_id] = model
                if model:
                    models.add(model)
            case cr.ResponseItem(type="reasoning"):
                owner: str = (
                    payload.internal_chat_message_metadata_passthrough.turn_id if payload.internal_chat_message_metadata_passthrough else turn_id
                )
                reasoning[owner].append(len(payload.encrypted_content or ""))
            case cr.ResponseItem():
                for part in payload.content or []:
                    if part.type == "input_image" and part.image_url.startswith("data:image/"):
                        parts: tuple[str, str, str] = part.image_url.partition(",")
                        header: str = parts[0]
                        separator: str = parts[1]
                        encoded: str = parts[2]
                        if separator and header.endswith(";base64"):
                            try:
                                blob: bytes = base64.b64decode(encoded, validate=True)
                            except binascii.Error:
                                raise ValueError(f"line={source.file_index + 1} invalid image encoding") from None
                            events.append(ev.TimedRecord(ev.Image(blob, header[5:-7]), source.timestamp_ns, source.file_index, turn_id=turn_id))
                if payload.type in {"function_call", "custom_tool_call"}:
                    raw_calls.append(RawCall(payload, turn_id, source.timestamp_ns))
                elif payload.type in {"function_call_output", "custom_tool_call_output"}:
                    outputs[payload.call_id] = (orjson.dumps(payload.output).decode(), source.timestamp_ns)
            case cr.TokenRecord():
                has_usage_records = True
                total = payload.thread_token_usage or total
                if payload.response_id not in seen_responses:
                    seen_responses.add(payload.response_id)
                    events.append(
                        ev.TimedRecord(
                            ev.UsageSample(usage_counters(payload.usage)),
                            source.timestamp_ns,
                            source.file_index,
                            turn_id=payload.turn_id or turn_id,
                            model=model,
                            effort=effort,
                        )
                    )
            case cr.Event():
                turn_id = payload.turn_id or turn_id
                for local in payload.local_images:
                    image_path: Path | None = images.get(local)
                    if image_path is not None:
                        events.append(
                            ev.TimedRecord(
                                ev.Image(image_path.read_bytes(), mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"),
                                source.timestamp_ns,
                                source.file_index,
                                turn_id=turn_id,
                            )
                        )
                if payload.thread_settings is not None:
                    effort = payload.thread_settings.reasoning_effort or ""
                event: ev.Payload | None = None
                values: dict[str, ev.Scalar] = {}
                message_id: str = ""
                timestamp: int = source.timestamp_ns
                if payload.type == "token_count" and payload.info is not None and payload.info.last_token_usage is not None:
                    counters: ev.Usage = usage_counters(payload.info.last_token_usage)
                    key: tuple[str, ev.Usage] = (turn_id, counters)
                    if key not in seen_legacy:
                        seen_legacy.add(key)
                        legacy_indices.add(len(events))
                        event = ev.UsageSample(counters)
                elif payload.type == "task_started":
                    # `started_at` is Unix seconds; the envelope timestamp is the same instant at millisecond precision.
                    event = ev.TurnBoundary("start")
                elif payload.type == "task_complete":
                    event = ev.TurnBoundary("complete", float(payload.duration_ms) if payload.duration_ms is not None else None)
                elif payload.type == "item_completed":
                    completed_items += 1
                    timestamp = payload.completed_at_ms * 1_000_000 if payload.completed_at_ms is not None else timestamp
                    item: cr.Item | None = payload.item
                    if isinstance(item, cr.MessageItem):
                        text: str = "\n".join(part.text for part in item.content if part.text)
                        event = ev.Prompt(text) if item.type == "UserMessage" else ev.AssistantText(text)
                        message_id = item.id if item.type == "AgentMessage" else ""
                        values = {"message_id": item.id, "phase": item.phase or ""}
                    elif isinstance(item, (cr.CommandExecution, cr.FileChange, cr.McpToolCall, cr.OtherTool)):
                        if isinstance(item, cr.OtherTool) and item.type == "ContextCompaction":
                            event = ev.Prompt("Context compacted", False, True)
                        else:
                            pending_tools.append(
                                CompletedTool(
                                    item,
                                    source.file_index,
                                    payload.started_at_ms * 1_000_000 if payload.started_at_ms is not None else timestamp,
                                    timestamp,
                                    turn_id,
                                    model,
                                    effort,
                                    source.item_json,
                                )
                            )
                    elif isinstance(item, cr.UnknownItem):
                        skipped[item.type] = skipped.get(item.type, 0) + 1
                    elif isinstance(item, cr.ReasoningItem):
                        reasoning_items[turn_id].append(len(events))
                        event = ev.Thinking("<encrypted reasoning, 0 bytes>")
                if event is not None:
                    events.append(
                        ev.TimedRecord(
                            event, timestamp, source.file_index, values,
                            turn_id=turn_id, prompt_id=turn_id, message_id=message_id, model=model, effort=effort,
                        )
                    )
                elif payload.type != "item_completed":
                    skipped[payload.type] = skipped.get(payload.type, 0) + 1
            case _:
                skipped[source.tag] = skipped.get(source.tag, 0) + 1
    if not completed_items:
        raise SkipRollout("no-item_completed")
    if meta is None:
        raise ValueError("missing session_meta")

    for owner, indices in reasoning_items.items():
        for index, size in zip(indices, reasoning.get(owner, []), strict=False):
            events[index] = replace(events[index], payload=ev.Thinking(f"<encrypted reasoning, {size} bytes>"))
    if has_usage_records:
        events = [event for index, event in enumerate(events) if index not in legacy_indices]
    used: set[int] = set()
    for completed in pending_tools:
        events.extend(tool_events(completed, raw_calls, outputs, used))
    events = [replace(event, model=turn_models.get(event.turn_id, event.model)) for event in events]
    events.sort(key=lambda event: event.file_index)
    home: Path = rollout_home(path)
    return ev.Session(
        meta.thread_id,
        home.name.lstrip("."),
        path.resolve(),
        events,
        {},
        skipped,
        agent="codex",
        cwd=meta.cwd,
        git_branch=meta.git.branch if meta.git else "",
        cli_versions={meta.cli_version},
        models=models,
        provider=meta.model_provider or "",
        originator=meta.originator,
        thread_source=meta.thread_source,
        forked_from=meta.forked_from_id or "",
        parent_thread=meta.parent_thread_id or "",
        total_input_tokens=total.input_tokens,
        total_output_tokens=total.output_tokens,
    )


def usage_counters(usage: cr.TokenUsage) -> ev.Usage:
    """Translate per-response counters to the shared usage vocabulary."""
    return ev.Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cached_input_tokens,
        cache_creation_tokens=usage.cache_write_input_tokens,
        thinking_tokens=usage.reasoning_output_tokens,
    )


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
    """Original native item JSON, including tool-specific result fields."""


def matching_call(completed: CompletedTool, calls: list[RawCall], used: set[int]) -> RawCall | None:
    """Join exact IDs or tool-specific arguments, never unrelated call order."""
    item: cr.CommandExecution | cr.FileChange | cr.McpToolCall | cr.OtherTool = completed.item
    for index, call in enumerate(calls):
        if index in used or (call.turn_id and completed.turn_id and call.turn_id != completed.turn_id):
            continue
        response: cr.ResponseItem = call.response
        raw: str = response.arguments or response.input
        matches: bool = bool(item.id and item.id == response.call_id)
        if isinstance(item, cr.CommandExecution) and item.command:
            command: str = item.command[-1]
            matches |= bool(command and (orjson.dumps(command).decode() in raw or command == raw))
        elif isinstance(item, cr.McpToolCall):
            matches |= response.name in {f"mcp__{item.server}__{item.tool}", f"mcp/{item.server}/{item.tool}"} and cr.arguments_equal(
                raw, item.arguments
            )
        elif isinstance(item, cr.FileChange):
            matches |= bool(item.changes) and all(path in raw for path in item.changes) and "apply_patch" in response.name
        elif isinstance(item, cr.OtherTool) and item.type == "ImageView":
            matches |= bool(item.path) and orjson.dumps(item.path).decode() in raw
        if matches:
            used.add(index)
            return call
    return None


def tool_events(completed: CompletedTool, calls: list[RawCall], outputs: dict[str, tuple[str, int]], used: set[int]) -> list[ev.TimedRecord]:
    """Emit native call and result events, enriched with matched raw data."""
    item: cr.CommandExecution | cr.FileChange | cr.McpToolCall | cr.OtherTool = completed.item
    matched: RawCall | None = matching_call(completed, calls, used)
    response: cr.ResponseItem | None = matched.response if matched is not None else None
    name: str
    kind: ev.ToolKind
    text: str = ""
    is_error: bool = False
    agent_id: str = ""
    if isinstance(item, cr.CommandExecution):
        name, kind = "exec", "shell"
        text = item.aggregated_output or item.stdout + item.stderr or item.formatted_output
        is_error = item.status in {"failed", "declined", "cancelled"} or item.exit_code not in {None, 0}
    elif isinstance(item, cr.FileChange):
        name, kind = "apply_patch", "file_edit"
        text = item.stdout + item.stderr
        is_error = item.status in {"failed", "declined", "cancelled"}
    elif isinstance(item, cr.McpToolCall):
        name, kind = f"mcp/{item.server}/{item.tool}", "mcp"
        text = item.error.message if item.error else ""
        is_error = item.error is not None or item.status in {"failed", "declined", "cancelled"}
    else:
        names: dict[str, tuple[str, ev.ToolKind]] = {
            "WebSearch": ("web_search", "web_search"),
            "Plan": ("update_plan", "plan"),
            "SubAgentActivity": (item.kind or "subagent", "subagent"),
            "ImageView": ("view_image", "image"),
            "Extension": (item.kind or "extension", "other"),
        }
        name, kind = names[item.type]
        agent_id = item.agent_thread_id
    input_json: str = (response.arguments or response.input) if response is not None else completed.item_json
    call_id: str = response.call_id if response is not None else item.id
    output: tuple[str, int] | None = outputs.get(call_id)
    raw_output: str = output[0] if output is not None else completed.item_json
    # The item's own started/completed stamps record when Codex logged it, ~1 ms apart. The raw call → output
    # envelope span is the command's real wall time; fall back to the item stamps when the pair is missing.
    paired: bool = matched is not None and output is not None
    started_ns: int = matched.timestamp_ns if paired and matched is not None else completed.started_ns
    completed_ns: int = output[1] if paired and output is not None else completed.completed_ns
    elapsed: float = (completed_ns - started_ns) / 1_000_000 if paired else float("nan")  # unknown, never the ~1 ms logging span
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


def rollout_home(path: Path) -> Path:
    """Find the owning home from either active or archived layout."""
    return next((parent.parent for parent in path.parents if parent.name in {"sessions", "archived_sessions"}), path.parent)


def discover_rollouts(home: Path, result: Discovery) -> dict[Path, cr.SessionMeta]:
    """Index metadata once, retaining excluded parents for child ownership."""
    index: dict[Path, cr.SessionMeta] = {}
    for directory in ("sessions", "archived_sessions"):
        for candidate in sorted((home / directory).rglob("*.jsonl")):
            if candidate.name.startswith("._"):
                continue
            path: Path = candidate.resolve()
            try:
                if provider_for(path) is claude:
                    # Untagged historical Codex metadata retains its exclusion policy.
                    try:
                        rollout_meta(path)
                    except SkipRollout:
                        raise
                    except ValueError:
                        pass
                    result.sessions.append(claude.session_source(path))
                    continue
                index[path] = rollout_meta(path)
                check_version(index[path])
            except SkipRollout as error:
                result.skipped[path] = str(error)
            except (ValueError, SerdeError, OSError) as error:
                result.failed[path] = str(error)
    return index


def discover(home: Path) -> Discovery:
    """Inventory recording owners, keeping excluded descendants out of orphan recordings."""
    result: Discovery = Discovery()
    index: dict[Path, cr.SessionMeta] = discover_rollouts(home, result)
    thread_ids: set[str] = {meta.thread_id for meta in index.values()}
    for path, meta in index.items():
        if meta.parent_thread_id in thread_ids:
            continue
        descendants: list[Path] = rollout_tree(path, index)[1:]
        folded: tuple[Path, ...] = tuple(child for child in descendants if child not in result.skipped and child not in result.failed)
        if path not in result.skipped and path not in result.failed:
            try:
                result.sessions.append(rollout_source(path, index, folded))
                continue
            except SkipRollout as error:
                result.skipped[path] = str(error)
            except (ValueError, SerdeError, OSError) as error:
                result.failed[path] = str(error)
        reason: str = "parent-failed" if path in result.failed else "parent-skipped"
        result.skipped.update(dict.fromkeys(folded, reason))
    return result


def rollout_tree(path: Path, index: dict[Path, cr.SessionMeta]) -> list[Path]:
    """Find recursive children by parent ID, guarding against cycles."""
    paths: list[Path] = [path]
    seen: set[str] = set()
    for current in paths:
        meta: cr.SessionMeta = index[current]
        thread_id: str = meta.thread_id
        if thread_id in seen:
            continue
        seen.add(thread_id)
        for candidate, child in index.items():
            if child.parent_thread_id == thread_id and candidate not in paths:
                paths.append(candidate)
    return paths


def session_source(path: Path) -> SessionSource:
    """Build a single rollout inventory using the provider-owned home index."""
    main: Path = path.expanduser().resolve()
    result: Discovery = Discovery()
    index: dict[Path, cr.SessionMeta] = discover_rollouts(rollout_home(main), result)
    if main not in index:
        index[main] = rollout_meta(main)
    return rollout_source(main, index, ())


def rollout_source(path: Path, index: dict[Path, cr.SessionMeta], folded: tuple[Path, ...]) -> SessionSource:
    """Inventory the rollout tree and the local images its parser can read."""
    check_version(index[path])
    paths: list[Path] = rollout_tree(path, index)
    transcripts: dict[str, Path] = {"": path, **{index[child].thread_id: child for child in paths[1:]}}
    assets: dict[Path, dict[str, Path]] = {}
    for source in paths:
        assets[source] = {}
        meta: cr.SessionMeta = index[source]
        try:
            check_version(meta)
        except SkipRollout:
            continue
        for record in iter_rollout(source):
            if isinstance(record.payload, cr.Event):
                for local in record.payload.local_images:
                    image: Path = Path(local)
                    if not image.is_absolute():
                        image = Path(meta.cwd) / image
                    if image.is_file():
                        assets[source][local] = image.resolve()
    inputs: tuple[Path, ...] = (*paths, *sorted({image for images in assets.values() for image in images.values()}))
    return SessionSource(index[path].thread_id, path, inputs, lambda: parse_rollout_inventory(path, transcripts, assets), folded)


def parse_rollout(path: Path) -> ev.Session:
    """Inventory and parse one rollout with all available descendants."""
    return session_source(path).parse()


def parse_rollout_inventory(path: Path, transcripts: dict[str, Path], assets: dict[Path, dict[str, Path]]) -> ev.Session:
    """Parse inventoried transcripts without discovering files again."""
    parent: ev.Session = parse_file(path, assets[path])
    children: dict[str, list[ev.TimedRecord]] = {}
    for agent_id, child_path in transcripts.items():
        if not agent_id:
            continue
        try:
            child: ev.Session = parse_file(child_path, assets[child_path])
        except SkipRollout as error:
            reason: str = str(error)
            parent.skipped[reason] = parent.skipped.get(reason, 0) + 1
            continue
        children[child.session_id] = child.main
        parent.models.update(child.models)
        parent.cli_versions.update(child.cli_versions)
        for reason, count in child.skipped.items():
            parent.skipped[reason] = parent.skipped.get(reason, 0) + count
    return replace(parent, subagents=children)
