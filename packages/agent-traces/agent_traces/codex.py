"""Stream Codex rollouts into provider-neutral recording events."""

import base64
import binascii
import hashlib
import mimetypes
import re
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Literal

import orjson
from serde import SerdeError, from_dict

from agent_traces import codex_records as cr
from agent_traces import events as ev
from agent_traces.codex_tools import execution, model_tool
from agent_traces.errors import SkipSession
from agent_traces.sources import MISSING_INPUT, PARENT_FAILED, PARENT_SKIPPED, Discovery, SessionSource, iter_jsonl
from agent_traces.timestamps import parse_timestamp_ns


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Typed payload and line provenance."""

    payload: cr.Decoded | None
    """Decoded known payload."""
    tag: str
    """Envelope type, or native output tag for model-tool replay identity."""
    timestamp_ns: int
    """Envelope Unix nanoseconds."""
    file_index: int
    """Zero-based source line."""
    item_json: str = ""
    """Native result metadata without duplicated text output."""
    item_input: str = ""
    """Only native input fields, never command output."""
    entity_name: str = ""
    """Native execution entity category."""
    call_id: str = ""
    """Explicit native relationship to a model call."""
    output_images: list[ev.Image] = field(default_factory=list)
    """Tool output images decoded alongside their redacted JSON text."""


@dataclass(frozen=True, slots=True)
class EnvelopeRecord:
    """A decoded envelope and its source position."""

    envelope: cr.Envelope
    """Untyped payload remains at the decoding boundary."""
    file_index: int
    """Zero-based source line."""


def decode_envelope(raw: dict[str, object], file_index: int) -> EnvelopeRecord:
    """Decode the common envelope without applying header policy."""
    return EnvelopeRecord(from_dict(cr.Envelope, raw), file_index)


def decode_record(source: EnvelopeRecord) -> SourceRecord:
    """Decode known payloads and preserve native details at one boundary."""
    envelope: cr.Envelope = source.envelope
    output_images: list[ev.Image] = []
    if envelope.type == "response_item" and envelope.payload.get("type") in {"function_call_output", "custom_tool_call_output"}:
        output: object = envelope.payload.get("output")
        if isinstance(output, list):
            redacted: list[object] = []
            for part in output:
                image: ev.Image | None = inline_image(from_dict(cr.Content, part)) if isinstance(part, dict) else None
                if image is not None:
                    output_images.append(image)
                    part = {**part, "image_url": ev.redacted_image(len(image.blob))}
                redacted.append(part)
            envelope = replace(envelope, payload={**envelope.payload, "output": redacted})
    try:
        payload: cr.Decoded | None = cr.decode_payload(envelope.type, envelope.payload)
        timestamp: int = parse_timestamp_ns(envelope.timestamp)
    except (SerdeError, ValueError) as error:
        raise ValueError(f"line={source.file_index + 1} invalid rollout structure ({type(error).__name__})") from None
    native: object = envelope.payload.get("item") if envelope.type == "event_msg" else None
    item_json: str = ""
    item_input: str = ""
    entity_name: str = ""
    call_id: str = ""
    if isinstance(native, dict):
        spec: cr.ItemTag | None = cr.ITEM_TYPES.get(str(native.get("type", "")))
        if spec is not None and spec.entity_name:
            entity_name = spec.entity_name
            identity: object = native.get("call_id", "")
            if not isinstance(identity, str):
                raise ValueError(f"line={source.file_index + 1} native call_id must be a string")
            call_id = identity
            item_input = orjson.dumps({key: value for key, value in native.items() if key in spec.input_fields}).decode()
            item_json = orjson.dumps({key: value for key, value in native.items() if key not in spec.output_fields}).decode()
    return SourceRecord(payload, str(envelope.payload.get("type", "")) if isinstance(payload, cr.ToolCallOutput) else envelope.type, timestamp, source.file_index, item_json, item_input, entity_name, call_id, output_images)


def check_version(meta: cr.SessionMeta) -> None:
    """Apply the CLI version floor before decoding any events."""
    version: re.Match[str] | None = re.match(r"^(\d+)\.(\d+)(?:\.|$)", meta.cli_version)
    if version is None:
        raise ValueError("invalid cli_version")
    if (int(version[1]), int(version[2])) < (0, 150):
        raise SkipSession(f"codex-cli-{meta.cli_version}")


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
    reasoning_texts: dict[str, list[str]] = field(default_factory=dict)
    """Reasoning text or opaque-size placeholders in per-turn arrival order."""
    raw_calls: dict[str, cr.ModelCall] = field(default_factory=dict)
    """Model calls indexed only by explicit call identity."""
    extra_inputs: dict[str, str] = field(default_factory=dict)
    """Hashes captured from the first read of each local image."""
    total: cr.TokenUsage | None = None
    """Last thread usage total."""
    has_usage_records: bool = False
    """Whether authoritative response usage supersedes legacy counts."""
    local_images: dict[int, list[ev.Image]] = field(default_factory=dict)
    """Image bytes consumed during collection by source line."""
    has_compacted: bool = False
    """Whether a summary supersedes native compaction markers."""
    skipped: Counter[str] = field(default_factory=Counter)
    """Unmodeled payloads only."""


def collect(path: Path, meta: cr.SessionMeta) -> RolloutFacts:
    """Collect body facts using the already decoded and accepted header."""
    facts: RolloutFacts = RolloutFacts(path.resolve(), meta)
    records: Iterator[EnvelopeRecord] = iter_jsonl(path, decode_envelope)
    legacy_total: cr.TokenUsage | None = None
    turn_id: str = ""
    model: str = ""
    effort: str = ""
    for envelope in records:
        source: SourceRecord = (SourceRecord(meta, "session_meta", parse_timestamp_ns(envelope.envelope.timestamp), 0)
                                if envelope.file_index == 0 else decode_record(envelope))
        payload: cr.Decoded | None = source.payload
        if isinstance(payload, (cr.Context, cr.Event)):
            turn_id = payload.turn_id or turn_id
        owner: str = payload.internal_chat_message_metadata_passthrough.turn_id if isinstance(payload, cr.ResponseItem) and payload.internal_chat_message_metadata_passthrough else turn_id
        if isinstance(payload, cr.Event):
            for local in payload.local_images:
                image: Path = Path(local)
                if not image.is_absolute():
                    image = Path(facts.meta.cwd) / image
                try:
                    blob: bytes = image.read_bytes()
                except FileNotFoundError:
                    facts.extra_inputs.setdefault(str(image), MISSING_INPUT)
                    continue
                # Match sources.input_digest without reading the image bytes twice.
                facts.extra_inputs.setdefault(str(image), hashlib.sha256(blob).hexdigest())
                facts.local_images.setdefault(source.file_index, []).append(ev.Image(blob, mimetypes.guess_type(image.name)[0] or "application/octet-stream"))
        match payload:
            case cr.Context():
                model = payload.model or model
                effort = payload.effort if payload.effort is not None else effort
                facts.turn_models[turn_id] = model
                if model:
                    facts.models.add(model)
            case cr.TokenCount(info=info) if info is not None:
                legacy_total = info.total_token_usage or legacy_total
            case cr.ThreadSettingsApplied(thread_settings=settings) if settings is not None:
                effort = settings.reasoning_effort or effort
            case cr.Reasoning():
                facts.reasoning_texts.setdefault(owner, []).append(reasoning_text(payload))
            case cr.FunctionCall() | cr.CustomToolCall():
                facts.raw_calls.setdefault(payload.call_id, payload)
            case cr.Compacted():
                facts.has_compacted = True
            case cr.TokenRecord():
                facts.has_usage_records = True
                facts.total = payload.thread_token_usage or facts.total
        facts.records.append(ContextualRecord(source, owner, model, effort))
    facts.total = facts.total or legacy_total
    return facts


@dataclass(frozen=True, slots=True)
class Emission:
    """Pure emitted events and unmodeled-record counts."""

    events: list[ev.TimedRecord]
    """Neutral events."""
    skipped: Counter[str]
    """All omitted records by full tag."""


@dataclass(frozen=True, slots=True)
class Conversation:
    """A candidate from one of the repeated conversation streams."""

    role: str
    """Message kind."""
    text: str
    """Full joined text."""
    message_id: str
    """Stable message identity when present."""
    rank: int
    """Native items precede response messages, then event messages."""


def content_text(parts: list[cr.Content] | None, *, encrypted_kind: Literal["payload", "reasoning"] = "payload") -> str:
    """Join the visible text in one native or response content list."""
    return "\n".join(
        ev.redacted(len(part.encrypted_content.encode()), kind=encrypted_kind) if part.type == "encrypted_content" else part.text
        for part in parts or [] if part.text or part.type == "encrypted_content")


def reasoning_text(item: cr.Reasoning) -> str:
    """Prefer readable reasoning, otherwise report only its encrypted size."""
    return content_text(item.summary, encrypted_kind="reasoning") or ev.redacted(len((item.encrypted_content or "").encode()))


def conversation(source: SourceRecord) -> Conversation | None:
    """Read one text candidate without choosing among duplicate sources."""
    payload: cr.Decoded | None = source.payload
    candidate: Conversation
    match payload:
        case cr.ItemCompleted(item=cr.MessageItem() as item):
            candidate = Conversation("user" if item.type == "UserMessage" else "assistant", content_text(item.content), item.id, 0)
        case cr.Message():
            candidate = Conversation(payload.role, content_text(payload.content), payload.id or "", 1)
        case cr.UserMessage(message=text) if text:
            candidate = Conversation("user", text, "", 2)
        case cr.AgentMessage(message=text) if text:
            candidate = Conversation("assistant", text, "", 2)
        case _:
            return None
    if candidate.role == "user" and candidate.text.lstrip().startswith("<environment_context>"):
        candidate = replace(candidate, role="environment")
    return candidate


def select_conversation(records: list[ContextualRecord]) -> list[Conversation | None]:
    """Match format copies one-to-one by turn/text or same-time/text, preferring native items."""
    candidates: list[Conversation | None] = [conversation(entry.source) for entry in records]
    selected: list[Conversation | None] = [None] * len(records)
    available: dict[tuple[int, str, str, str, str], deque[int]] = {}
    consumed: set[tuple[int, int]] = set()
    identities: set[tuple[int, str, str, str]] = set()
    for rank in range(3):
        for index, (entry, candidate) in enumerate(zip(records, candidates, strict=True)):
            if candidate is None or candidate.rank != rank:
                continue
            identity: tuple[int, str, str, str] = (rank, entry.turn_id, candidate.role, candidate.message_id)
            if candidate.message_id and identity in identities:
                continue
            identities.add(identity)
            keys: tuple[tuple[str, str, str, str], ...] = (
                ("time", str(entry.source.timestamp_ns), candidate.role, candidate.text),
                ("turn", entry.turn_id, candidate.role, candidate.text),
            )
            owner: int = index
            for key in keys:
                queue: deque[int] = available.get((rank, *key), deque())
                while queue and (rank, queue[0]) in consumed:
                    queue.popleft()
                if queue:
                    owner = queue.popleft()
                    consumed.add((rank, owner))
                    break
            if owner == index:
                selected[index] = candidate
            # Copies can expose an inherited turn ID; make that alias available too.
            for lower_rank in range(rank + 1, 3):
                for key in keys:
                    available.setdefault((lower_rank, *key), deque()).append(owner)
    return selected


INJECTED_TAGS: tuple[str, ...] = (
    "<recommended_plugins>", "# AGENTS.md instructions", "<codex_internal_context", "<turn_aborted>", "<skill>", "<chat-history-summary>",
)
"""Harness messages that do not express a human request."""


def prompt_role(text: str) -> Literal["human", "injected"]:
    """Classify user-authored text without treating task markup as instructions."""
    return "injected" if text.lstrip().startswith(INJECTED_TAGS) else "human"


def inline_image(part: cr.Content) -> ev.Image | None:
    """Decode one inline data image, leaving non-image parts to their own handlers."""
    if part.type != "input_image" or not part.image_url.startswith("data:image/"):
        return None
    parts: tuple[str, str, str] = part.image_url.partition(",")
    header: str = parts[0]
    if not parts[1] or not header.endswith(";base64"):
        return None
    try:
        blob: bytes = base64.b64decode(parts[2], validate=True)
    except binascii.Error:
        raise ValueError("invalid image encoding") from None
    return ev.Image(blob, header[5:-7])


def emit(facts: RolloutFacts) -> Emission:
    """Emit once from complete facts, with usage and reasoning decisions settled."""
    skipped: Counter[str] = facts.skipped.copy()
    events: list[ev.TimedRecord] = []
    seen_responses: set[str] = set()
    seen_instructions: set[str] = set()
    seen_executions: set[str] = set()
    seen_tools: set[tuple[str, str]] = set()
    seen_legacy: set[tuple[str, ev.Usage]] = set()
    reasoning: dict[str, Iterator[str]] = {owner: iter(texts) for owner, texts in facts.reasoning_texts.items()}
    native_reasoning: Counter[str] = Counter(entry.turn_id for entry in facts.records
        if isinstance(entry.source.payload, cr.ItemCompleted)
        and isinstance(entry.source.payload.item, cr.ReasoningItem))
    response_reasoning: Counter[str] = Counter()
    candidates: list[Conversation | None] = select_conversation(facts.records)
    human_turns: set[str] = {entry.turn_id for entry, candidate in zip(facts.records, candidates, strict=True)
                             if candidate is not None and candidate.role == "user" and prompt_role(candidate.text) == "human"}
    for entry, candidate in zip(facts.records, candidates, strict=True):
        source: SourceRecord = entry.source
        turn_id: str = entry.turn_id
        model: str = facts.turn_models.get(turn_id, entry.model)
        effort: str = entry.effort
        payload: cr.Decoded | None = source.payload

        timed: partial[ev.TimedRecord] = partial(ev.TimedRecord, timestamp_ns=source.timestamp_ns, file_index=source.file_index,
                                                turn_id=turn_id, model=model, effort=effort)

        if candidate is not None:
            message: ev.Payload
            if candidate.role == "user":
                message = ev.Prompt(candidate.text, prompt_role(candidate.text))
            elif candidate.role == "assistant":
                message = ev.AssistantText(candidate.text)
            else:
                message = ev.ContextText(candidate.role or "message", candidate.text)
            events.append(timed(message, message_id=candidate.message_id if candidate.role == "assistant" else ""))
        for local_image in facts.local_images.get(source.file_index, []):
            events.append(timed(local_image))
        event: ev.Payload | None = None
        timestamp: int = source.timestamp_ns
        match payload:
            case cr.SessionMeta():
                if payload.base_instructions is not None and payload.base_instructions.text and payload.base_instructions.text not in seen_instructions:
                    seen_instructions.add(payload.base_instructions.text)
                    events.append(timed(ev.ContextText("base_instructions", payload.base_instructions.text), turn_id="", model="", effort=""))
            case cr.Compacted():
                text_compaction: str = payload.message or "\n\n".join(
                    part.text for item in payload.replacement_history or [] for part in item.content or [] if part.text)
                event = ev.Prompt(text_compaction or "Context compacted", "compaction")
            case cr.Reasoning():
                response_reasoning[turn_id] += 1
                if response_reasoning[turn_id] > native_reasoning[turn_id]:
                    event = ev.Thinking(reasoning_text(payload))
            case cr.FunctionCall() | cr.CustomToolCall() | cr.ToolCallOutput():
                output_identity: str = payload.call_id if isinstance(payload, cr.ModelCall) else payload.id or ""
                key_call: tuple[str, str] = (type(payload).__name__ if isinstance(payload, cr.ModelCall) else source.tag, output_identity)
                if not output_identity or key_call not in seen_tools:
                    seen_tools.add(key_call)
                    event = model_tool(payload, facts.raw_calls)
                    for image in source.output_images:
                        events.append(timed(replace(image, call_id=payload.call_id, source="tool_result")))
            case cr.InterAgentMessage():
                events.append(timed(ev.InterAgent(content_text(payload.content)), extras={"author": payload.author, "recipient": payload.recipient}, message_id=payload.id or ""))
            case cr.UnknownResponse():
                skipped[f"response_item/{payload.type}"] += 1
            case cr.TokenRecord():
                if payload.response_id not in seen_responses:
                    seen_responses.add(payload.response_id)
                    events.append(timed(ev.UsageSample(usage_counters(payload.usage)), turn_id=payload.turn_id or turn_id,
                                        model=facts.turn_models.get(payload.turn_id or turn_id, model)))
            case cr.TokenCount(info=info):
                if info is not None and info.last_token_usage is not None:
                    counters: ev.Usage = usage_counters(info.last_token_usage)
                    key: tuple[str, ev.Usage] = (turn_id, counters)
                    if not facts.has_usage_records and key not in seen_legacy:
                        seen_legacy.add(key)
                        event = ev.UsageSample(counters)
            case cr.TaskStarted():
                # `started_at` is Unix seconds; the envelope has the same instant at millisecond precision.
                event = ev.TurnBoundary("start") if turn_id in human_turns else None
            case cr.TaskComplete():
                event = ev.TurnBoundary("complete", float(payload.duration_ms) if payload.duration_ms is not None else None)
            case cr.ItemCompleted(item=cr.CommandExecution() | cr.FileChange() | cr.McpToolCall() | cr.OtherTool() as item):
                if not item.id or item.id not in seen_executions:
                    seen_executions.add(item.id)
                    event = execution(item, source.item_json, source.item_input, source.entity_name, source.call_id)
            case cr.ItemCompleted(item=cr.ContextCompaction()):
                if not facts.has_compacted:
                    event = ev.Prompt("Context compacted", "compaction")
            case cr.ItemCompleted(item=cr.UnknownItem() as item):
                skipped[f"event_msg/item_completed/{item.type}"] += 1
            case cr.ItemCompleted(item=cr.ReasoningItem()):
                event = ev.Thinking(next(reasoning.get(turn_id, iter(())), ev.redacted(None)))
            case cr.UnknownEvent():
                skipped[f"event_msg/{payload.type}"] += 1
            case cr.Context() | cr.Message() | cr.UserMessage() | cr.AgentMessage() | cr.ThreadSettingsApplied() | cr.ItemCompleted():
                pass  # Already consumed by collection or conversation selection.
            case _:
                skipped[source.tag] += 1
        if event is not None:
            if isinstance(payload, cr.ItemCompleted) and payload.completed_at_ms is not None:
                timestamp = payload.completed_at_ms * 1_000_000
            events.append(timed(event, timestamp_ns=timestamp, prompt_id=turn_id if isinstance(event, ev.TurnBoundary) and event.phase == "start" else ""))
        if isinstance(payload, cr.ResponseItem) and not isinstance(payload, cr.Reasoning):
            for part in payload.content or []:
                try:
                    image: ev.Image | None = inline_image(part)
                except ValueError as error:
                    raise ValueError(f"line={source.file_index + 1} {error}") from None
                if image is not None:
                    events.append(timed(image))
    return Emission(events, skipped)


def usage_counters(usage: cr.TokenUsage) -> ev.Usage:
    """Translate per-response counters to the shared usage vocabulary."""
    return ev.Usage(
        input_tokens=max(usage.input_tokens - usage.cached_input_tokens, 0),
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cached_input_tokens,
        cache_creation_tokens=usage.cache_write_input_tokens,
        thinking_tokens=usage.reasoning_output_tokens,
    )


def rollout_home(path: Path) -> Path:
    """Find the owning home from either active or archived layout."""
    return next((parent.parent for parent in path.parents if parent.name in {"sessions", "archived_sessions"}), path.parent)


def rollout_header(path: Path) -> cr.SessionMeta:
    """Decode a header and identify legacy layouts without reading its body."""
    with path.open("rb") as stream:
        line: bytes = stream.readline()
    try:
        raw: object = orjson.loads(line)
        if isinstance(raw, dict) and "type" not in raw and "id" in raw and "timestamp" in raw:
            raise SkipSession("legacy-rollout")
        envelope: cr.Envelope = from_dict(cr.Envelope, raw)
        if envelope.type != "session_meta":
            raise ValueError("missing session_meta")
        meta: cr.SessionMeta = from_dict(cr.SessionMeta, envelope.payload)
        if not meta.thread_id:
            raise ValueError("missing thread identity")
        return meta
    except (SerdeError, orjson.JSONDecodeError) as error:
        raise ValueError(f"line=1 invalid rollout header ({type(error).__name__})") from None


def discover_rollouts(home: Path, result: Discovery) -> dict[Path, cr.SessionMeta]:
    """Index headers only, retaining excluded parents for child ownership."""
    index: dict[Path, cr.SessionMeta] = {}
    for directory in ("sessions", "archived_sessions"):
        for candidate in sorted((home / directory).rglob("*.jsonl")):
            if candidate.name.startswith("._"):
                continue
            path: Path = candidate.resolve()
            try:
                index[path] = rollout_header(path)
                check_version(index[path])
            except SkipSession as error:
                result.skipped[path] = str(error)
            except (ValueError, SerdeError, OSError) as error:
                result.failed[path] = str(error)
    return index


def child_index(index: dict[Path, cr.SessionMeta]) -> dict[str, list[Path]]:
    """Index direct descendants once per home inventory."""
    children: dict[str, list[Path]] = {}
    for path, meta in index.items():
        if meta.parent_thread_id:
            children.setdefault(meta.parent_thread_id, []).append(path)
    return children


def is_home(home: Path) -> bool:
    """Whether this directory has the provider's transcript layout."""
    return any((home / name).is_dir() for name in ("sessions", "archived_sessions"))


def discover(home: Path) -> Discovery:
    """Inventory recording owners without decoding or retaining rollout bodies."""
    result: Discovery = Discovery()
    index: dict[Path, cr.SessionMeta] = discover_rollouts(home, result)
    children: dict[str, list[Path]] = child_index(index)
    thread_ids: set[str] = {meta.thread_id for meta in index.values()}
    for path, meta in index.items():
        if meta.parent_thread_id in thread_ids:
            continue
        descendants: list[Path] = rollout_tree(path, index, children)[1:]
        folded: tuple[Path, ...] = tuple(child for child in descendants if child not in result.skipped and child not in result.failed)
        if path not in result.skipped and path not in result.failed:
            result.sessions.append(rollout_source(path, index, children, result, folded))
        else:
            reason: str = PARENT_FAILED if path in result.failed else PARENT_SKIPPED
            result.skipped.update(dict.fromkeys(folded, reason))
    return result


def rollout_tree(path: Path, index: dict[Path, cr.SessionMeta], children: dict[str, list[Path]]) -> list[Path]:
    """Walk indexed children with constant-time checks for already visited paths."""
    paths: list[Path] = [path]
    seen: set[Path] = {path}
    for current in paths:
        for child in children.get(index[current].thread_id, []):
            if child not in seen:
                seen.add(child)
                paths.append(child)
    return paths


def session_source(path: Path) -> SessionSource:
    """Build a single rollout inventory using the provider-owned header index."""
    main: Path = path.expanduser().resolve()
    result: Discovery = Discovery()
    index: dict[Path, cr.SessionMeta] = discover_rollouts(rollout_home(main), result)
    if main in result.skipped:
        raise SkipSession(result.skipped[main])
    if main in result.failed:
        raise ValueError(result.failed[main])
    if main not in index:
        index[main] = rollout_header(main)
        check_version(index[main])
    return rollout_source(main, index, child_index(index), result)


def rollout_source(path: Path, index: dict[Path, cr.SessionMeta], children: dict[str, list[Path]], result: Discovery,
                   folded: tuple[Path, ...] = ()) -> SessionSource:
    """Capture transcript paths and header policy results for one recording."""
    inputs: tuple[Path, ...] = tuple(rollout_tree(path, index, children))
    skipped: dict[Path, str] = result.skipped
    failed: dict[Path, str] = result.failed
    return SessionSource(index[path].thread_id, path, inputs, lambda: parse_rollout_inventory(inputs, index, skipped, failed),
                         project=index[path].cwd, folded=folded)


def parse_rollout_inventory(inputs: tuple[Path, ...], index: dict[Path, cr.SessionMeta], skipped: dict[Path, str],
                            failed: dict[Path, str]) -> ev.Session:
    """Collect, emit and release each rollout before building the merged session."""
    meta: cr.SessionMeta = index[inputs[0]]
    total: cr.TokenUsage | None = None
    main: list[ev.TimedRecord] = []
    children: dict[str, list[ev.TimedRecord]] = {}
    metadata: dict[str, ev.AgentMetadata] = {}
    omitted: Counter[str] = Counter()
    extra_inputs: dict[str, str] = {}
    models: set[str] = set()
    versions: set[str] = set()
    for path in inputs:
        if path != inputs[0]:
            if path in skipped:
                omitted[skipped[path]] += 1
                continue
            if path in failed:
                raise ValueError(failed[path])
        facts: RolloutFacts = collect(path, index[path])
        emitted: Emission = emit(facts)
        if path == inputs[0]:
            main = emitted.events
            total = facts.total
        else:
            children[facts.meta.thread_id] = emitted.events
            metadata[facts.meta.thread_id] = ev.AgentMetadata(facts.meta.agent_role, facts.meta.agent_nickname)
        for image, digest in facts.extra_inputs.items():
            extra_inputs.setdefault(image, digest)
        models.update(facts.models)
        versions.add(facts.meta.cli_version)
        omitted.update(emitted.skipped)
        del facts
    return ev.Session(
        meta.thread_id, rollout_home(inputs[0]).name.lstrip("."), inputs[0].resolve(), main, children, omitted,
        agent="codex", extra_inputs=extra_inputs, agent_metadata=metadata,
        properties={
            "cwd": meta.cwd, "git_branch": meta.git.branch if meta.git else "", "provider": meta.model_provider or "", "originator": meta.originator,
            "thread_source": meta.thread_source, "forked_from": meta.forked_from_id or "", "parent_thread": meta.parent_thread_id or "",
            "total_input_tokens": usage_counters(total).input_tokens if total else None,
            "total_cache_read_tokens": total.cached_input_tokens if total else None, "total_output_tokens": total.output_tokens if total else None, "total_cost_usd": None,
            "models": ",".join(sorted(models)), "cli_versions": ",".join(sorted(versions)),
        },
    )
