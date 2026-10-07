"""Stream Claude JSONL records into one typed session."""

import base64
import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias, cast

import orjson
from serde import SerdeError, from_dict
from serde.json import from_json

from agent_traces import events as ev
from agent_traces.claude_records import (
    AgentLabels,
    Block,
    CacheCreation,
    ImageBlock,
    ImageSource,
    OutputTokensDetails,
    Record,
    ResultContent,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from agent_traces.claude_workflows import workflow_records
from agent_traces.sources import Discovery, SessionSource, iter_jsonl
from agent_traces.timestamps import parse_timestamp_ns


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    """Typed record and uninterpreted source metadata."""

    file_index: int
    """Zero-based source line index."""
    record: Record
    """Decoded Claude record."""
    tool_metadata: object
    """Opaque tool metadata, redacted only when its image is emitted."""
    raw_json: str
    """Whole system record or just the attachment JSON."""
    timestamp_ns: int | None
    """Parsed wall timestamp, absent on untimed metadata."""
    replay_key: str
    """UUID or content digest scoped to timestamp (source position when untimed)."""


def decode_record(raw: dict[str, object], file_index: int) -> _SourceRecord:
    """Decode one Claude record and preserve opaque metadata at the boundary."""
    digest: str = hashlib.sha256(orjson.dumps(raw, option=orjson.OPT_SORT_KEYS)).hexdigest() if not raw.get("uuid") else ""
    raw_json: str = (orjson.dumps(raw.get("attachment")).decode() if raw.get("type") == "attachment" else
                     orjson.dumps(raw).decode() if raw.get("type") == "system" else "")
    tool_metadata: object = raw.get("toolUseResult")
    if not isinstance(tool_metadata, dict):
        raw = {**raw, "toolUseResult": None}
    record: Record = from_dict(Record, raw)
    if record.message is not None and record.message.model == "<synthetic>":
        record = replace(record, message=replace(record.message, model=""))
    timestamp_ns: int | None = parse_timestamp_ns(record.timestamp) if record.timestamp is not None else None
    return _SourceRecord(file_index, record, tool_metadata, raw_json, timestamp_ns, record.uuid or f"{timestamp_ns if timestamp_ns is not None else file_index}:{digest}")


def result_text(block: ToolResultBlock) -> str:
    """Return the text the model saw in a tool response."""
    if isinstance(block.content, str):
        return block.content
    return "\n".join(part.text if part.type == "text" else part.tool_name
                     for part in block.content or [] if part.type in {"text", "tool_reference"})


def result_images(block: ToolResultBlock) -> list[ImageSource]:
    """Return image sources from a tool response."""
    if not isinstance(block.content, list):
        return []
    return [part.source for part in block.content if part.type == "image" and part.source is not None]


def output_candidate(reference: str, tool_results_dir: Path, outputs: frozenset[Path], *, persisted: bool) -> Path | None:
    """Resolve an output reference within this session; reject prose and escaping paths."""
    candidate: Path = Path(reference)
    if ".." in candidate.parts:
        return None
    if candidate.is_absolute():
        # Copied homes retain the old prefix; this session's suffix identifies the output.
        parts: tuple[str, ...] = candidate.parts
        for index in range(len(parts) - 2):
            if parts[index:index + 2] == (tool_results_dir.parent.name, "tool-results"):
                candidate = tool_results_dir.joinpath(*parts[index + 2:])
                break
    elif candidate.parts[:1] == ("tool-results",):
        candidate = tool_results_dir.parent / candidate
    elif persisted or tool_results_dir / candidate in outputs:
        candidate = tool_results_dir / candidate
    else:
        return None
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    return candidate if candidate.is_relative_to(tool_results_dir) else None


def inline_offloaded_output(block: ToolResultBlock, persisted_path: str | None, tool_results_dir: Path, outputs: frozenset[Path], counts: Counter[str]) -> ToolResultBlock:
    """Expand a local tool response only within its session's output directory."""
    reference: str | None = persisted_path
    if not reference:
        marker: re.Match[str] | None = re.search(r"(?:[Oo]utput (?:has been )?saved to|[Ss]aved to(?: file)?):?\s*([^\n]+)", result_text(block))
        reference = marker.group(1).strip(" `").removesuffix(".") if marker else None
    if not reference:
        return block
    candidate: Path | None = output_candidate(reference, tool_results_dir, outputs, persisted=bool(persisted_path))
    if candidate is None or candidate not in outputs:
        if persisted_path or candidate is not None:
            counts["offloaded-output-missing"] += 1
        return block
    full_text: str = candidate.read_bytes().decode("utf-8", "replace")
    replacement: str | list[ResultContent] = (
        [ResultContent(type="text", text=full_text), *(part for part in block.content if part.type != "text")]
        if isinstance(block.content, list)
        else full_text
    )
    return replace(block, content=replacement)


KEPT_ATTACHMENTS: frozenset[str] = frozenset({"queued_command", "command_permissions", "hook_success", "edited_text_file", "auto_mode"})

INJECTED_TAGS: tuple[str, ...] = (
    "<task-notification>", "<local-command-stdout>", "<local-command-stderr>",
    "<local-command-caveat>", "<chat-history-summary>", "[Request interrupted", "<paseo-system>", "[Workflow harness",
)
"""Harness prefixes that do not represent human prompts."""


def session_source(path: Path) -> SessionSource:
    """Resolve one transcript and inventory its children and offloaded files."""
    main: Path = path.expanduser().resolve()
    directory: Path = main.with_suffix("")
    transcripts: dict[str, list[Path]] = {"": [main]}
    children_dir: Path = directory / "subagents"
    children: list[Path] = sorted(child for child in children_dir.rglob("agent-*.jsonl") if child.is_file())
    for child in children:
        key: str = child.stem.removeprefix("agent-")
        transcripts.setdefault(key, []).append(child.resolve())
    metadata_paths: dict[str, list[Path]] = {
        identity: [meta for path in files if (meta := path.with_suffix(".meta.json")).is_file()]
        for identity, files in transcripts.items() if identity
    }
    tool_results_dir: Path = (directory / "tool-results").resolve()
    outputs: tuple[Path, ...] = tuple(asset.resolve() for asset in sorted(tool_results_dir.rglob("*")) if asset.is_file())
    workflows: tuple[Path, ...] = tuple(sorted(
        path.resolve() for pattern in ("workflows/wf_*.json", "workflows/scripts/*.js") for path in directory.glob(pattern) if path.is_file()))
    inputs: tuple[Path, ...] = (*(path for files in transcripts.values() for path in files), *outputs, *workflows, *(path for files in metadata_paths.values() for path in files))
    return SessionSource(main.stem, main, inputs, lambda: parse_session_inventory(main, transcripts, tool_results_dir, frozenset(outputs), workflows, metadata_paths), project=main.parent.name)


def is_home(home: Path) -> bool:
    """Whether this directory has the provider's transcript layout."""
    return (home / "projects").is_dir()


def discover(home: Path) -> Discovery:
    """Discover main transcripts, reporting inventory failures by path."""
    if not is_home(home):
        raise ValueError(f"{home}: Claude home requires a projects directory")
    result: Discovery = Discovery()
    for path in sorted((home / "projects").glob("*/*.jsonl")):
        if path.name.startswith(("agent-", "._")):
            continue
        try:
            result.sessions.append(session_source(path))
        except (ValueError, SerdeError, OSError) as error:
            result.failed[path] = str(error)
    return result


def unique_records(paths: list[Path], skipped: Counter[str]) -> Iterator[_SourceRecord]:
    """Keep the first UUID occurrence across all files of one agent."""
    seen: set[str] = set()
    for path in paths:
        for source in iter_jsonl(path, decode_record, skipped=skipped):
            if source.replay_key in seen:
                skipped["replayed-record"] += 1
            else:
                seen.add(source.replay_key)
                yield source


def prepare_record(source: _SourceRecord, tool_results_dir: Path, outputs: frozenset[Path], counts: Counter[str]) -> tuple[_SourceRecord, int] | None:
    """Select timed payloads and expand inventoried tool results."""
    record: Record = source.record
    kind: str = record.attachment.type if record.type == "attachment" and record.attachment is not None else record.type
    keep: bool = kind in KEPT_ATTACHMENTS if record.type == "attachment" else record.type in {"user", "assistant", "system", "pr-link"}
    timestamp_ns: int | None = source.timestamp_ns
    if timestamp_ns is None or not keep:
        counts[record.type if source.timestamp_ns is None else kind] += 1
        return None
    n_inlined: int = 0
    if record.message is not None:
        blocks: list[Block] | None = None
        persisted: str | None = record.toolUseResult.persistedOutputPath if record.toolUseResult else None
        for index, block in enumerate(record.message.content):
            if isinstance(block, ToolResultBlock):
                expanded: ToolResultBlock = inline_offloaded_output(block, persisted, tool_results_dir, outputs, counts)
                if expanded is not block:
                    if blocks is None:
                        blocks = list(record.message.content)
                    blocks[index] = expanded
                    n_inlined += 1
        if blocks is not None:
            source = replace(source, record=replace(record, message=replace(record.message, content=blocks)))
    return source, n_inlined


def parse_session_inventory(source_path: Path, paths: dict[str, list[Path]], tool_results_dir: Path, outputs: frozenset[Path], workflows: tuple[Path, ...], metadata_paths: dict[str, list[Path]]) -> ev.Session:
    """Fold selected transcripts, metadata and skip counts into a session."""
    skipped: Counter[str] = Counter()
    properties: dict[str, ev.Scalar | None] = {"cwd": "", "git_branch": "", "title": "", "total_cost_usd": None}
    cli_versions: set[str] = set()
    models: set[str] = set()
    transcripts: dict[str, list[ev.TimedRecord]] = {}
    n_inlined: int = 0
    for agent_id, files in paths.items():
        rows: list[_SourceRecord] = []
        for source in unique_records(files, skipped):
            record: Record = source.record
            if record.version:
                cli_versions.add(record.version)
            if record.type == "assistant" and record.message is not None and record.message.model:
                models.add(record.message.model)
            if not agent_id:
                properties["cwd"] = record.cwd or properties["cwd"]
                properties["git_branch"] = record.gitBranch or properties["git_branch"]
                if record.type in {"custom-title", "ai-title"}:
                    properties["title"] = record.customTitle or record.aiTitle or record.content or ""
                if record.type == "cost-state":
                    properties["total_cost_usd"] = float(record.totalCostUSD) if record.totalCostUSD is not None else None
            prepared: tuple[_SourceRecord, int] | None = prepare_record(source, tool_results_dir, outputs, skipped)
            if prepared is not None:
                rows.append(prepared[0])
                n_inlined += prepared[1]
        if agent_id:
            rows.sort(key=lambda row: cast(int, row.timestamp_ns))
        transcripts[agent_id] = interpret_records(rows)
    metadata: dict[str, ev.AgentMetadata] = {}
    for identity, files in metadata_paths.items():
        for path in files:
            try:
                labels: AgentLabels = from_json(AgentLabels, path.read_bytes())
            except (SerdeError, ValueError) as error:
                raise ValueError(f"{path}: invalid agent metadata ({type(error).__name__})") from None
            metadata[identity] = ev.AgentMetadata(labels.agentType, labels.description)
    transcripts[""].extend(workflow_records(workflows, min((row.timestamp_ns for records in transcripts.values() for row in records), default=None)))
    properties.update(n_inlined_outputs=n_inlined, cli_versions=",".join(sorted(cli_versions)), models=",".join(sorted(models)))
    return ev.Session(source_path.stem, source_path.parents[2].name.lstrip("."), source_path, transcripts.pop(""), transcripts, skipped, properties=properties, agent_metadata=metadata)


def tool_kind(name: str) -> ev.ToolKind:
    """Map Claude native names to the shared tool vocabulary."""
    if name.startswith("mcp__"):
        return "mcp"
    kinds: dict[str, ev.ToolKind] = {
        "Bash": "shell",
        "Read": "file_read",
        "Edit": "file_edit",
        "Write": "file_edit",
        "WebFetch": "web_search",
        "WebSearch": "web_search",
        "Agent": "subagent",
        "Workflow": "subagent",
    }
    return kinds.get(name, "other")


RecordPayloads: TypeAlias = list[tuple[ev.Payload, dict[str, ev.Scalar]]]


def prompt_role(record: Record, text: str) -> Literal["human", "injected", "compaction"]:
    """Use explicit origin first, then classify each text block by its prefix."""
    if record.isMeta or record.promptSource == "system" or record.turnOrigin in {"task_notification", "peer", "scheduled", "system"}:
        return "injected"
    if record.turnOrigin == "human" or record.promptSource == "typed":
        return "human"
    if record.isCompactSummary:
        return "compaction"
    return "injected" if text.lstrip().startswith(INJECTED_TAGS) else "human"


def system_payloads(source: _SourceRecord) -> RecordPayloads:
    """Keep system text and the uninterpreted source envelope."""
    record: Record = source.record
    return [(ev.Lifecycle("system", record.content if record.content is not None else record.subtype, (record.level or "INFO").upper()),
             {"subtype": record.subtype, "extra_json": source.raw_json})]


def attachment_payloads(source: _SourceRecord) -> RecordPayloads:
    """Keep selected attachment text and metadata."""
    record: Record = source.record
    if record.attachment is None:
        return []
    attachment_json: str = source.raw_json
    description: str = (record.attachment.text or (record.attachment.content if isinstance(record.attachment.content, str) else "")
                        or record.attachment.command or record.attachment.message or attachment_json)
    return [(ev.Lifecycle("attachments", f"{record.attachment.type}: {description}"),
             {"subtype": record.attachment.type, "attachment_json": attachment_json})]


def pr_link_payloads(source: _SourceRecord) -> RecordPayloads:
    """Keep a linked pull request."""
    record: Record = source.record
    return [(ev.Lifecycle("pr_links", record.prUrl), {"pr_number": record.prNumber, "pr_repository": record.prRepository})]


def message_payloads(source: _SourceRecord, timestamp_ns: int, calls: dict[str, tuple[int, str]]) -> RecordPayloads:
    """Turn message blocks into payloads without mutating parser state."""
    record: Record = source.record
    if record.message is None:
        return []
    payloads: RecordPayloads = []
    for block in record.message.content:
        images: list[ev.Image] = []
        match block:
            case ToolUseBlock():
                payloads.append((ev.ToolCall(ev.tool_path(block.name), block.id, orjson.dumps(block.input).decode(), tool_kind(block.name), block.name), {}))
            case ToolResultBlock():
                call: tuple[int, str] | None = calls.get(block.tool_use_id)
                name: str = call[1] if call else "unknown"
                elapsed: float | None = (timestamp_ns - call[0]) / 1_000_000 if call else None
                images = [ev.Image(base64.b64decode(image.data), image.media_type, block.tool_use_id, "tool_result")
                          for image in result_images(block) if image.type == "base64"]
                metadata: object = source.tool_metadata
                if images and isinstance(metadata, dict) and metadata.get("type") == "image":
                    file: object = metadata.get("file")
                    if isinstance(file, dict) and isinstance(file.get("base64"), str):
                        encoded: str = file["base64"]
                        if len(encoded) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", encoded):
                            n_bytes: int = len(encoded) * 3 // 4 - encoded[-2:].count("=")
                            metadata = {**metadata, "file": {**file, "base64": ev.redacted_image(n_bytes)}}
                metadata_json: str = orjson.dumps(metadata).decode() if metadata is not None else ""
                payloads.append((ev.ToolResult(ev.tool_path(name), block.tool_use_id, result_text(block), metadata_json,
                                              tool_kind(name), elapsed, block.is_error,
                                              (record.toolUseResult.agentId or "") if record.toolUseResult else ""), {}))
            case ImageBlock(source=image) if record.type == "user" and image is not None and image.type == "base64":
                images = [ev.Image(base64.b64decode(image.data), image.media_type)]
            case TextBlock(text=text) | ThinkingBlock(thinking=text):
                extras: dict[str, ev.Scalar] = {"uuid": record.uuid, "parent_uuid": record.parentUuid or ""}
                if record.type == "assistant":
                    extras["request_id"] = record.requestId or ""
                payload: ev.Payload = (ev.Thinking(text or ev.redacted(len(block.signature.encode()))) if isinstance(block, ThinkingBlock) else
                                       ev.Prompt(text, prompt_role(record, text)) if record.type == "user" else ev.AssistantText(text))
                payloads.append((payload, extras))
        payloads.extend((image, {}) for image in images)
    return payloads


def user_payloads(source: _SourceRecord, timestamp_ns: int, calls: dict[str, tuple[int, str]]) -> RecordPayloads:
    """Any human text block opens exactly one turn."""
    payloads: RecordPayloads = message_payloads(source, timestamp_ns, calls)
    if any(isinstance(payload, ev.Prompt) and payload.role == "human" for payload, _ in payloads):
        payloads.insert(0, (ev.TurnBoundary("start"), {}))
    return payloads


def assistant_payloads(source: _SourceRecord, timestamp_ns: int, calls: dict[str, tuple[int, str]], last_message: dict[str, _SourceRecord]) -> RecordPayloads:
    """Emit finalized usage at the last split record and keep every content block."""
    payloads: RecordPayloads = message_payloads(source, timestamp_ns, calls)
    record: Record = source.record
    if record.isApiErrorMessage:
        return [(ev.Lifecycle("api_errors", payload.text, "ERROR"), extras) if isinstance(payload, ev.AssistantText) else (payload, extras)
                for payload, extras in payloads]
    if record.message is not None and record.message.id and last_message[record.message.id] is source:
        usage: Usage = record.message.usage or Usage()
        cache: CacheCreation = usage.cache_creation or CacheCreation()
        details: OutputTokensDetails = usage.output_tokens_details or OutputTokensDetails()
        counters: ev.Usage = ev.Usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens, cache_creation_tokens=usage.cache_creation_input_tokens,
            cache_creation_5m_tokens=cache.ephemeral_5m_input_tokens, cache_creation_1h_tokens=cache.ephemeral_1h_input_tokens,
            thinking_tokens=details.thinking_tokens)
        payloads.insert(0, (ev.UsageSample(counters), {}))
    return payloads


RECORD_PAYLOADS: dict[str, Callable[[_SourceRecord], RecordPayloads]] = {
    "system": system_payloads, "attachment": attachment_payloads, "pr-link": pr_link_payloads,
}


def interpret_records(records: list[_SourceRecord]) -> list[ev.TimedRecord]:
    """Fold pure record payloads into typed provenance in transcript order."""
    events: list[ev.TimedRecord] = []
    turn_id: str = ""
    calls: dict[str, tuple[int, str]] = {}
    last_message: dict[str, _SourceRecord] = {}
    for source in records:
        stamp: int = cast(int, source.timestamp_ns)
        if source.record.message is not None:
            if source.record.type == "assistant" and source.record.message.id:
                last_message[source.record.message.id] = source
            for block in source.record.message.content:
                if isinstance(block, ToolUseBlock):
                    calls[block.id] = (stamp, block.name)
    for record_index, source in enumerate(records):
        stamp = cast(int, source.timestamp_ns)
        record: Record = source.record
        payloads: RecordPayloads
        if record.type == "user":
            payloads = user_payloads(source, stamp, calls)
        elif record.type == "assistant":
            payloads = assistant_payloads(source, stamp, calls, last_message)
        else:
            payloads = RECORD_PAYLOADS[record.type](source)
        for payload, extras in payloads:
            if isinstance(payload, ev.TurnBoundary) and payload.phase == "start":
                turn_id = record.uuid or f"record-{record_index}"
            events.append(ev.TimedRecord(payload, stamp, source.file_index, extras, turn_id=turn_id,
                prompt_id=record.promptId or "", message_id=record.message.id if record.type == "assistant" and record.message else "",
                model=record.message.model if record.type == "assistant" and record.message else "",
                effort=record.effort if record.type == "assistant" else ""))
    return events
