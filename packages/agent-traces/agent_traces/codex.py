"""Stream Codex rollouts into provider-neutral recording events."""

import base64
import binascii
import mimetypes
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeAlias

import orjson
from serde import SerdeError, from_dict

from agent_traces import claude
from agent_traces import codex_records as cr
from agent_traces import events as ev
from agent_traces.codex_tools import CompletedTool, RawCall, tool_events
from agent_traces.sources import Discovery, SessionSource
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
    """Native result metadata without duplicated text output."""
    item_input: str = ""
    """Only native input fields, never command output."""


def iter_rollout(path: Path) -> Iterator[SourceRecord]:
    """Decode one line at a time; diagnostics never include source content."""
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            try:
                raw: object = orjson.loads(line)
                if not isinstance(raw, dict):
                    raise SerdeError("expected an object")
                if index == 0 and "type" not in raw and "id" in raw and "timestamp" in raw:
                    raise SkipRollout("no-item_completed")
                if index == 0 and raw.get("type") != "session_meta":
                    raise NotCodex("missing session_meta")
                envelope: cr.Envelope = from_dict(cr.Envelope, raw)
                cls: type | None = PAYLOAD_TYPES.get(envelope.type)
                payload: Decoded | None = from_dict(cls, envelope.payload) if cls is not None else None
                timestamp: int = parse_timestamp_ns(envelope.timestamp)
            except (SkipRollout, NotCodex):
                raise
            except (orjson.JSONDecodeError, SerdeError, ValueError) as error:
                raise ValueError(f"line={index + 1} invalid rollout structure ({type(error).__name__})") from None
            native: object = envelope.payload.get("item") if envelope.type == "event_msg" else None
            item_json: str = ""
            item_input: str = ""
            if isinstance(native, dict):
                item_input = orjson.dumps({key: value for key, value in native.items() if key in {
                    "command", "cwd", "changes", "arguments", "path", "query", "action", "server", "tool", "kind",
                }}).decode()
                duplicated_output: set[str] = (
                    {"stdout", "stderr", "aggregated_output", "formatted_output"}
                    if isinstance(payload, cr.Event) and isinstance(payload.item, (cr.CommandExecution, cr.FileChange)) else set()
                )
                item_json = orjson.dumps({key: value for key, value in native.items() if key not in duplicated_output}).decode()
            yield SourceRecord(payload, envelope.type, timestamp, index, item_json, item_input)



class SkipRollout(ValueError):
    """A rollout excluded by the supported-history policy."""


class NotCodex(ValueError):
    """A transcript with a non-Codex first-line tag."""


def check_version(meta: cr.SessionMeta) -> None:
    """Apply the CLI version floor before decoding any events."""
    version: re.Match[str] | None = re.match(r"^(\d+)\.(\d+)(?:\.|$)", meta.cli_version)
    if version is None:
        raise ValueError("invalid cli_version")
    if (int(version[1]), int(version[2])) < (0, 150):
        raise SkipRollout(f"codex-cli-{meta.cli_version}")


@dataclass(frozen=True, slots=True)
class ContextualRecord:
    """Source record with the context in force at arrival."""

    source: SourceRecord
    """Typed source and provenance."""
    turn_id: str
    """Owning turn."""
    model: str
    """Model known at arrival."""
    effort: str
    """Reasoning effort at arrival."""


@dataclass(slots=True)
class RolloutFacts:
    """Collected rollout facts; emission never repairs event-list indices."""

    path: Path
    """Canonical source path."""
    meta: cr.SessionMeta
    """First-line metadata."""
    records: list[ContextualRecord] = field(default_factory=list)
    """Records in source arrival order."""
    turn_models: dict[str, str] = field(default_factory=dict)
    """Final model for each turn."""
    models: set[str] = field(default_factory=set)
    """All observed models."""
    reasoning_sizes: dict[str, list[int]] = field(default_factory=dict)
    """Encrypted response sizes in per-turn arrival order."""
    raw_calls: list[RawCall] = field(default_factory=list)
    """Raw calls, reusable by several native completions."""
    outputs: dict[str, tuple[str, int]] = field(default_factory=dict)
    """Output JSON and envelope timestamp by raw call ID."""
    images: dict[str, Path] = field(default_factory=dict)
    """Existing local image paths, shared with the source inventory."""
    total: cr.TokenUsage = field(default_factory=cr.TokenUsage)
    """Last thread usage total."""
    has_usage_records: bool = False
    """Whether authoritative response usage supersedes legacy counts."""
    completed_items: int = 0
    """Native completions establish supported history."""
    skipped: dict[str, int] = field(default_factory=dict)
    """Unmodeled payloads only."""
    failure: str = ""
    """Decode failure after valid metadata, retained for parent ownership."""
    exclusion: str = ""
    """Version-floor exclusion, retaining metadata for child ownership."""


def collect(path: Path) -> RolloutFacts:
    """Read the header once, check its version once, then collect typed facts."""
    records: Iterator[SourceRecord] = iter_rollout(path)
    first: SourceRecord | None = next(records, None)
    if first is None or not isinstance(first.payload, cr.SessionMeta):
        raise ValueError("missing session_meta")
    facts: RolloutFacts = RolloutFacts(path.resolve(), first.payload)
    try:
        check_version(facts.meta)
    except SkipRollout as error:
        facts.exclusion = str(error)
        return facts
    turn_id: str = ""
    model: str = ""
    effort: str = ""
    try:
        for source in records:
            payload: Decoded | None = source.payload
            if isinstance(payload, (cr.Context, cr.Event)):
                turn_id = payload.turn_id or turn_id
            if isinstance(payload, cr.Context):
                model = payload.model or model
                facts.turn_models[turn_id] = model
                if model:
                    facts.models.add(model)
            elif isinstance(payload, cr.Event):
                if payload.thread_settings is not None:
                    effort = payload.thread_settings.reasoning_effort or ""
                facts.completed_items += payload.type == "item_completed"
                for local in payload.local_images:
                    image: Path = Path(local)
                    if not image.is_absolute():
                        image = Path(facts.meta.cwd) / image
                    if image.is_file():
                        facts.images[local] = image.resolve()
            elif isinstance(payload, cr.ResponseItem):
                if payload.type == "reasoning":
                    owner: str = payload.internal_chat_message_metadata_passthrough.turn_id if payload.internal_chat_message_metadata_passthrough else turn_id
                    facts.reasoning_sizes.setdefault(owner, []).append(len(payload.encrypted_content or ""))
                elif payload.type in {"function_call", "custom_tool_call"}:
                    facts.raw_calls.append(RawCall(payload, turn_id, source.timestamp_ns))
                elif payload.type in {"function_call_output", "custom_tool_call_output"}:
                    facts.outputs[payload.call_id] = (orjson.dumps(payload.output).decode(), source.timestamp_ns)
            elif isinstance(payload, cr.TokenRecord):
                facts.has_usage_records = True
                facts.total = payload.thread_token_usage or facts.total
            facts.records.append(ContextualRecord(source, turn_id, model, effort))
    except (ValueError, SerdeError, OSError) as error:
        facts.failure = str(error)
    # Some older layouts omit context until the first completion. A sole
    # native turn identifies those leading calls; multiple turns stay unknown.
    turns: set[str] = {entry.turn_id for entry in facts.records if entry.turn_id}
    if len(turns) == 1:
        owner: str = next(iter(turns))
        facts.raw_calls = [replace(call, turn_id=call.turn_id or owner) for call in facts.raw_calls]
    return facts


def emit(facts: RolloutFacts) -> list[ev.TimedRecord]:
    """Emit once from complete facts, with usage and reasoning decisions settled."""
    if facts.failure:
        raise ValueError(facts.failure)
    if facts.exclusion:
        raise SkipRollout(facts.exclusion)
    if not facts.completed_items:
        raise SkipRollout("no-item_completed")
    facts.skipped.clear()
    events: list[ev.TimedRecord] = []
    seen_responses: set[str] = set()
    seen_legacy: set[tuple[str, ev.Usage]] = set()
    reasoning: dict[str, Iterator[int]] = {owner: iter(sizes) for owner, sizes in facts.reasoning_sizes.items()}
    for entry in facts.records:
        source: SourceRecord = entry.source
        turn_id: str = entry.turn_id
        model: str = facts.turn_models.get(turn_id, entry.model)
        effort: str = entry.effort
        payload: Decoded | None = source.payload
        match payload:
            case cr.SessionMeta() | cr.Context() | cr.ResponseItem(type="reasoning"):
                continue
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
            case cr.TokenRecord():
                if payload.response_id not in seen_responses:
                    seen_responses.add(payload.response_id)
                    events.append(
                        ev.TimedRecord(
                            ev.UsageSample(usage_counters(payload.usage)),
                            source.timestamp_ns,
                            source.file_index,
                            turn_id=payload.turn_id or turn_id,
                            model=facts.turn_models.get(payload.turn_id or turn_id, model),
                            effort=effort,
                        )
                    )
            case cr.Event():
                for local in payload.local_images:
                    image_path: Path | None = facts.images.get(local)
                    if image_path is not None:
                        events.append(
                            ev.TimedRecord(
                                ev.Image(image_path.read_bytes(), mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"),
                                source.timestamp_ns,
                                source.file_index,
                                turn_id=turn_id,
                            )
                        )
                event: ev.Payload | None = None
                values: dict[str, ev.Scalar] = {}
                message_id: str = ""
                timestamp: int = source.timestamp_ns
                if payload.type == "token_count" and payload.info is not None and payload.info.last_token_usage is not None:
                    counters: ev.Usage = usage_counters(payload.info.last_token_usage)
                    key: tuple[str, ev.Usage] = (turn_id, counters)
                    if not facts.has_usage_records and key not in seen_legacy:
                        seen_legacy.add(key)
                        event = ev.UsageSample(counters)
                elif payload.type == "task_started":
                    # `started_at` is Unix seconds; the envelope timestamp is the same instant at millisecond precision.
                    event = ev.TurnBoundary("start")
                elif payload.type == "task_complete":
                    event = ev.TurnBoundary("complete", float(payload.duration_ms) if payload.duration_ms is not None else None)
                elif payload.type == "item_completed":
                    timestamp = payload.completed_at_ms * 1_000_000 if payload.completed_at_ms is not None else timestamp
                    item: cr.Item | None = payload.item
                    if isinstance(item, cr.MessageItem):
                        text: str = "\n".join(part.text for part in item.content if part.text)
                        event = ev.Prompt(text) if item.type == "UserMessage" else ev.AssistantText(text)
                        message_id = item.id if item.type == "AgentMessage" else ""
                        values = {"message_id": item.id, "phase": item.phase or ""}
                    elif isinstance(item, (cr.CommandExecution, cr.FileChange, cr.McpToolCall, cr.OtherTool)):
                        completed: CompletedTool = CompletedTool(
                            item, source.file_index,
                            payload.started_at_ms * 1_000_000 if payload.started_at_ms is not None else timestamp,
                            timestamp, turn_id, model, effort, source.item_json, source.item_input,
                        )
                        events.extend(tool_events(completed, facts.raw_calls, facts.outputs))
                    elif isinstance(item, cr.ContextCompaction):
                        event = ev.Prompt("Context compacted", False, True)
                    elif isinstance(item, cr.UnknownItem):
                        facts.skipped[item.type] = facts.skipped.get(item.type, 0) + 1
                    elif isinstance(item, cr.ReasoningItem):
                        size: int = next(reasoning.get(turn_id, iter(())), 0)
                        event = ev.Thinking(f"<encrypted reasoning, {size} bytes>")
                if event is not None:
                    events.append(
                        ev.TimedRecord(
                            event, timestamp, source.file_index, values,
                            turn_id=turn_id, prompt_id=turn_id, message_id=message_id, model=model, effort=effort,
                        )
                    )
                elif payload.type not in {"item_completed", "token_count", "user_message", "thread_settings_applied"}:
                    facts.skipped[payload.type] = facts.skipped.get(payload.type, 0) + 1
            case _:
                facts.skipped[source.tag] = facts.skipped.get(source.tag, 0) + 1
    return events


def parse_file(path: Path, images: dict[str, Path]) -> ev.Session:
    """Parse a standalone transcript with explicitly inventoried images."""
    facts: RolloutFacts = collect(path)
    facts.images = images
    return session_from_facts(facts)


def session_from_facts(facts: RolloutFacts) -> ev.Session:
    """Build a session without re-reading or re-checking its rollout."""
    events: list[ev.TimedRecord] = emit(facts)
    # Replayed histories may carry later session metadata. Preserve the existing
    # emitted identity; discovery still uses the first-line owner metadata.
    meta: cr.SessionMeta = next(
        (entry.source.payload for entry in reversed(facts.records) if isinstance(entry.source.payload, cr.SessionMeta)), facts.meta
    )
    home: Path = rollout_home(facts.path)
    return ev.Session(
        meta.thread_id,
        home.name.lstrip("."),
        facts.path,
        events,
        {},
        dict(facts.skipped),
        agent="codex",
        cwd=meta.cwd,
        git_branch=meta.git.branch if meta.git else "",
        cli_versions={meta.cli_version},
        models=set(facts.models),
        provider=meta.model_provider or "",
        originator=meta.originator,
        thread_source=meta.thread_source,
        forked_from=meta.forked_from_id or "",
        parent_thread=meta.parent_thread_id or "",
        total_input_tokens=facts.total.input_tokens,
        total_output_tokens=facts.total.output_tokens,
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



def rollout_home(path: Path) -> Path:
    """Find the owning home from either active or archived layout."""
    return next((parent.parent for parent in path.parents if parent.name in {"sessions", "archived_sessions"}), path.parent)


def discover_rollouts(home: Path, result: Discovery) -> dict[Path, RolloutFacts]:
    """Index metadata once, retaining excluded parents for child ownership."""
    index: dict[Path, RolloutFacts] = {}
    for directory in ("sessions", "archived_sessions"):
        for candidate in sorted((home / directory).rglob("*.jsonl")):
            if candidate.name.startswith("._"):
                continue
            path: Path = candidate.resolve()
            try:
                index[path] = collect(path)
                if index[path].failure:
                    result.failed[path] = index[path].failure
                if index[path].exclusion:
                    result.skipped[path] = index[path].exclusion
            except NotCodex:
                result.sessions.append(claude.session_source(path))
            except SkipRollout as error:
                result.skipped[path] = str(error)
            except (ValueError, SerdeError, OSError) as error:
                result.failed[path] = str(error)
    return index


def discover(home: Path) -> Discovery:
    """Inventory recording owners, keeping excluded descendants out of orphan recordings."""
    result: Discovery = Discovery()
    index: dict[Path, RolloutFacts] = discover_rollouts(home, result)
    thread_ids: set[str] = {facts.meta.thread_id for facts in index.values()}
    for path, facts in index.items():
        if facts.meta.parent_thread_id in thread_ids:
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


def rollout_tree(path: Path, index: dict[Path, RolloutFacts]) -> list[Path]:
    """Find recursive children by parent ID, guarding against cycles."""
    paths: list[Path] = [path]
    seen: set[str] = set()
    for current in paths:
        meta: cr.SessionMeta = index[current].meta
        thread_id: str = meta.thread_id
        if thread_id in seen:
            continue
        seen.add(thread_id)
        for candidate, child in index.items():
            if child.meta.parent_thread_id == thread_id and candidate not in paths:
                paths.append(candidate)
    return paths


def session_source(path: Path) -> SessionSource:
    """Build a single rollout inventory using the provider-owned home index."""
    main: Path = path.expanduser().resolve()
    result: Discovery = Discovery()
    index: dict[Path, RolloutFacts] = discover_rollouts(rollout_home(main), result)
    if main not in index:
        index[main] = collect(main)
    return rollout_source(main, index, ())


def rollout_source(path: Path, index: dict[Path, RolloutFacts], folded: tuple[Path, ...]) -> SessionSource:
    """Inventory the rollout tree and the local images its parser can read."""
    if index[path].exclusion:
        raise SkipRollout(index[path].exclusion)
    paths: list[Path] = rollout_tree(path, index)
    inventory: dict[Path, RolloutFacts] = {source: index[source] for source in paths}
    inputs: tuple[Path, ...] = (*paths, *sorted({image for facts in inventory.values() for image in facts.images.values()}))
    return SessionSource(index[path].meta.thread_id, path, inputs, lambda: parse_rollout_inventory(path, inventory), folded)


def parse_rollout(path: Path) -> ev.Session:
    """Inventory and parse one rollout with all available descendants."""
    return session_source(path).parse()


def parse_rollout_inventory(path: Path, inventory: dict[Path, RolloutFacts]) -> ev.Session:
    """Parse inventoried transcripts without discovering files again."""
    parent: ev.Session = session_from_facts(inventory[path])
    children: dict[str, list[ev.TimedRecord]] = {}
    for child_path, facts in inventory.items():
        if child_path == path:
            continue
        try:
            child: ev.Session = session_from_facts(facts)
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
