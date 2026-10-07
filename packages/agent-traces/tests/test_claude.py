"""Tests at the streaming parser and session boundary."""

from collections import Counter
from pathlib import Path

import pyarrow as pa
import pytest

from agent_traces.claude import decode_record as decode_claude_record
from agent_traces.claude import session_source as claude_session_source
from agent_traces.claude_records import Record, TextBlock, UnknownBlock
from agent_traces.events import (
    AssistantText,
    Image,
    Lifecycle,
    Prompt,
    Session,
    Thinking,
    TimedRecord,
    ToolCall,
    ToolResult,
    TurnBoundary,
    UsageSample,
)
from agent_traces.rerun_log import PROPERTY_TYPES, write_session_rrd
from agent_traces.sources import fingerprint, iter_jsonl
from agent_traces.timestamps import parse_timestamp_ns
from agent_traces.turns import aggregate_turns
from tests.conftest import SessionBuilder, metadata_values, parse_session, read_entities


def test_streams_typed_records_and_preserves_line_indices(session_builder: SessionBuilder) -> None:
    """Unknown fields are allowed and file order remains available."""
    session_builder.add("user", message={"role": "user", "content": "hello"}, future_field=True)
    session_builder.add("assistant", message={"id": "m1", "content": [{"type": "text", "text": "hi"}]})
    records = iter_jsonl(session_builder.path, decode_claude_record, skipped=Counter())
    assert iter(records) is records
    first_line = next(records)
    first: Record = first_line.record
    assert first.message is not None
    assert first.message.content == [TextBlock(text="hello")]
    session: Session = parse_session(session_builder.path)
    assert session.session_id == "session-123"
    assert session.profile == "claude"
    assert [row.file_index for row in session.main] == [0, 0, 1, 1]
    assert session.main[0].timestamp_ns == 1_789_761_600_000_000_000


def test_skips_noise_and_folds_subagents_without_losing_tool_results(session_builder: SessionBuilder) -> None:
    """Count noise by kind, retain typed results, and preserve child ids."""
    session_builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "done"}]})
    session_builder.add("attachment", attachment={"type": "total_tokens_reminder"})
    session_builder.add("queue-operation")
    session_builder.add("future-kind")
    session_builder.add("system", timestamp=None)
    session_builder.add("user", path=session_builder.path.with_suffix("") / "subagents/agent-child.jsonl", message={"content": "child"})
    session: Session = parse_session(session_builder.path)
    assert session.skipped == {"total_tokens_reminder": 1, "queue-operation": 1, "future-kind": 1, "system": 1}
    assert len(session.main) == 1
    assert isinstance(session.main[0].payload, ToolResult)
    assert session.main[0].payload.call_id == "t1"
    assert session.main[0].payload.text == "done"
    assert list(session.subagents) == ["child"]
    assert session.subagents["child"][0].file_index == 0


def test_bad_lines_report_source_and_line_without_reading_ahead(session_builder: SessionBuilder) -> None:
    """A valid first line is yielded before a damaged second line is reported; a wrongly shaped object still raises."""

    session_builder.add("user", message={"content": "valid"})
    with session_builder.path.open("ab") as stream:
        stream.write(b"{broken\n")
    records = iter_jsonl(session_builder.path, decode_claude_record, skipped=Counter())
    first_line = next(records)
    assert first_line.record.type == "user"
    with pytest.warns(UserWarning, match=r"session-123.jsonl:2: damaged-line"), pytest.raises(StopIteration):
        next(records)
    session_builder.path.write_bytes(b'{"type": "user", "message": {"content": 123}}\n')
    with pytest.raises(ValueError, match=r"session-123.jsonl:1"):
        list(iter_jsonl(session_builder.path, decode_claude_record, skipped=Counter()))


def test_inlines_only_outputs_inside_session_tool_results(session_builder: SessionBuilder) -> None:
    """Both CLI output references and persisted paths resolve within the session."""
    results: Path = session_builder.path.with_suffix("") / "tool-results"
    results.mkdir(parents=True)
    output: Path = results / "result.txt"
    output.write_text("full tool output")
    session_builder.add(
        "user",
        message={"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Output saved to: missing-preview.txt"}]},
        toolUseResult={"agentId": "child", "persistedOutputPath": str(output), "future": 42},
    )
    session_builder.add(
        "user", message={"content": [{"type": "tool_result", "tool_use_id": "t2", "content": f"Output too large. Full output saved to: {output}"}]}
    )
    session_builder.add(
        "user",
        message={"content": [{"type": "tool_result", "tool_use_id": "t3", "content": "missing"}]},
        toolUseResult={"persistedOutputPath": str(results / "missing.txt")},
    )
    outside: Path = results.parent / "outside.txt"
    outside.write_text("must not inline")
    session_builder.add(
        "user", message={"content": [{"type": "tool_result", "content": "outside"}]}, toolUseResult={"persistedOutputPath": str(outside)}
    )
    session: Session = parse_session(session_builder.path)
    results_emitted: list[ToolResult] = [row.payload for row in session.main if isinstance(row.payload, ToolResult)]
    assert [result.text for result in results_emitted] == ["full tool output", "full tool output", "missing", "outside"]
    assert session.properties["n_inlined_outputs"] == 2
    assert results_emitted[0].agent_id == "child"
    assert '"future":42' in results_emitted[0].raw_json


def test_inlines_list_result_text_without_removing_images(png_bytes: bytes, png_block: dict[str, object], session_builder: SessionBuilder) -> None:
    """Persisted text replaces previews while image content stays attached."""
    results: Path = session_builder.path.with_suffix("") / "tool-results"
    results.mkdir(parents=True)
    output: Path = results / "result.txt"
    output.write_text("complete output")
    session_builder.add(
        "user",
        message={
            "content": [
                {
                    "type": "tool_result",
                    "content": [
                        {"type": "text", "text": f"Full output saved to: {output}"},
                        png_block,
                    ],
                }
            ]
        },
    )
    session: Session = parse_session(session_builder.path)
    assert isinstance(session.main[0].payload, ToolResult)
    assert session.main[0].payload.text == "complete output"
    assert isinstance(session.main[1].payload, Image)
    assert session.main[1].payload.blob == png_bytes
    assert session.main[1].payload.media_type == "image/png"
    assert session.main[1].payload.source == "tool_result"
    assert session.properties["n_inlined_outputs"] == 1


def test_invalid_record_shape_and_timestamp_name_the_source(session_builder: SessionBuilder) -> None:
    """Malformed record envelopes and timestamps have actionable diagnostics."""

    session_builder.path.parent.mkdir(parents=True)
    session_builder.path.write_text("[]\n")
    with pytest.raises(ValueError, match=r"session-123.jsonl:1"):
        parse_session(session_builder.path)
    session_builder.path.unlink()
    session_builder.add("user", timestamp="not-a-timestamp", message={"content": "hello"})
    with pytest.raises(ValueError, match=r"session-123.jsonl:1"):
        parse_session(session_builder.path)


def test_wall_timestamp_preserves_nanoseconds_and_timezone(session_builder: SessionBuilder) -> None:
    """ISO timestamps retain their full precision rather than rounding floats."""
    session_builder.add("user", timestamp="2026-09-18T15:00:00.123456789-05:00", message={"content": "precise"})
    session: Session = parse_session(session_builder.path)
    assert session.main[0].timestamp_ns == 1_789_761_600_123_456_789


def test_content_boundary_normalizes_and_rejects_malformed_known_blocks(session_builder: SessionBuilder) -> None:
    """Known tags are strict; unknown tags retain only their identity."""


    session_builder.add("user", message={"content": "hello"})
    session_builder.add("assistant", message={"content": [{"type": "future", "text": 42}, {}]})
    session: Session = parse_session(session_builder.path)
    assert [row.payload.text for row in session.main if isinstance(row.payload, Prompt)] == ["hello"]
    records = [line.record for line in iter_jsonl(session_builder.path, decode_claude_record, skipped=Counter())]
    assert len(records) == 2
    assert records[0].message is not None
    assert records[1].message is not None
    assert records[0].message.content == [TextBlock(text="hello")]
    assert records[1].message.content == [UnknownBlock(type="future"), UnknownBlock()]
    session_builder.add("assistant", message={"content": [{"type": "text", "text": 42}]})
    with pytest.raises(ValueError, match=r"session-123.jsonl:3"):
        parse_session(session_builder.path)


def test_source_metadata_cannot_be_spoofed_and_invalid_utf8_is_replaced(session_builder: SessionBuilder) -> None:
    """Raw metadata stays outside the source schema; output decoding is explicit."""

    results: Path = session_builder.path.with_suffix("") / "tool-results"
    results.mkdir(parents=True)
    (results / "result.txt").write_bytes(b"full\xffoutput")
    session_builder.add(
        "user",
        raw_json="spoof",
        tool_use_result_json="spoof",
        toolUseResult={"persistedOutputPath": "result.txt", "unknown": {"nested": [None]}},
        message={"content": [{"type": "tool_result", "content": "preview"}]},
    )
    session: Session = parse_session(session_builder.path)
    result = session.main[0].payload
    assert isinstance(result, ToolResult)
    assert result.raw_json == '{"persistedOutputPath":"result.txt","unknown":{"nested":[null]}}'
    assert result.text == "full\ufffdoutput"


def test_timestamp_grammar_and_integer_precision(session_builder: SessionBuilder) -> None:
    """Accept only explicit zoned timestamps with up to nine fraction digits."""


    equivalent: list[str] = ["1970-01-01T00:00:00Z", "1970-01-01T05:30:00+05:30", "1969-12-31T19:00:00-05:00"]
    for stamp in equivalent:
        assert parse_timestamp_ns(stamp) == 0
    fractions: list[tuple[str, int]] = [
        ("1", 100000000),
        ("12", 120000000),
        ("123", 123000000),
        ("1234", 123400000),
        ("12345", 123450000),
        ("123456", 123456000),
        ("1234567", 123456700),
        ("12345678", 123456780),
        ("123456789", 123456789),
    ]
    for fraction, expected in fractions:
        assert parse_timestamp_ns(f"1970-01-01T00:00:00.{fraction}Z") == expected
    assert parse_timestamp_ns("1970-01-01T00:00:00,123456789Z") == 123456789
    assert parse_timestamp_ns("1969-12-31T23:59:59.999999999Z") == -1
    rejected: list[str] = [
        "1970-01-01 00:00:00Z",
        "1970-01-01T00:00:00",
        "1970-01-01T00:00:00.1234567890Z",
        "1970-01-01T00:00:00+00:00.5",
        "1970-01-01T00:00:00+00:00:01",
        "1970-01-01T00:00:00.Z",
        "1970-01-01T00:00:00+24:00",
        "1970-01-01T00:00:00+00:60",
        "1970-02-30T00:00:00Z",
    ]
    for stamp in rejected:
        with pytest.raises(ValueError):
            parse_timestamp_ns(stamp)
    session_builder.add("user", timestamp=rejected[0], message={"content": "bad"})
    with pytest.raises(ValueError, match=r"session-123.jsonl:1"):
        parse_session(session_builder.path)
def test_structured_attachment_content_is_counted_not_rejected(session_builder: SessionBuilder) -> None:
    """Reminder and file attachments carry list or object content; they must parse and be counted as skipped."""
    session_builder.add("attachment", attachment={"type": "task_reminder", "content": [{"id": "1", "status": "open"}]})
    session_builder.add("attachment", attachment={"type": "file", "content": {"filePath": "/x", "content": "y"}})
    session_builder.add("attachment", attachment={"type": "hook_success", "command": "echo", "content": "hook said hi"})
    session: Session = parse_session(session_builder.path)
    assert session.skipped == {"task_reminder": 1, "file": 1}
    assert [timed.extras["subtype"] for timed in session.main if isinstance(timed.payload, Lifecycle)] == ["hook_success"]


def test_inlines_offloaded_output_with_invalid_utf8_bytes(session_builder: SessionBuilder) -> None:
    """Tool stdout saved by the CLI can contain bytes that are not UTF-8; they are replaced, not fatal."""
    session_dir: Path = session_builder.path.with_suffix("")
    (session_dir / "tool-results").mkdir(parents=True)
    (session_dir / "tool-results" / "out.txt").write_bytes(b"ok \xeb bad")
    session_builder.add("assistant", message={"id": "m1", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]})
    session_builder.add(
        "user",
        message={"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Output saved to out.txt"}]},
        toolUseResult={"persistedOutputPath": str(session_dir / "tool-results" / "out.txt")},
    )
    session: Session = parse_session(session_builder.path)
    assert session.properties["n_inlined_outputs"] == 1
    result = session.main[-1].payload
    assert isinstance(result, ToolResult) and result.text == "ok \ufffd bad"


def test_session_sources_include_sorted_recursive_outputs(session_builder: SessionBuilder) -> None:
    """Discovery includes ignored inputs but parses only child transcripts."""
    session_builder.add("user", message={"content": "main"})
    root: Path = session_builder.path.with_suffix("")
    for name in ["z", "a"]:
        session_builder.add("user", path=root / f"subagents/agent-{name}.jsonl", message={"content": name})
    (root / "tool-results/pdf-id").mkdir(parents=True)
    (root / "tool-results/z.txt").write_text("output")
    (root / "tool-results/pdf-id/page.jpg").write_bytes(b"image")
    (root / "tool-results/agent-ignored.jsonl").write_text("not a transcript")
    (root / "subagents/agent-directory.jsonl").mkdir()
    assert list(claude_session_source(session_builder.path).inputs) == [
        session_builder.path,
        root / "subagents/agent-a.jsonl",
        root / "subagents/agent-z.jsonl",
        root / "tool-results/agent-ignored.jsonl",
        root / "tool-results/pdf-id/page.jpg",
        root / "tool-results/z.txt",
    ]
    assert list(parse_session(session_builder.path).subagents) == ["a", "z"]


def test_marker_prose_that_is_not_a_path_is_left_alone(session_builder: SessionBuilder) -> None:
    """A tool result whose text says "saved to" followed by a paragraph must not be treated as a file reference."""
    prose: str = "Output saved to " + "x" * 5000
    session_builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": "1", "content": prose}]})
    session: Session = parse_session(session_builder.path)
    result = session.main[0].payload
    assert isinstance(result, ToolResult) and result.text == prose
    assert session.properties["n_inlined_outputs"] == 0


def test_flat_events_carry_turn_and_assistant_identity(png_block: dict[str, object], session_builder: SessionBuilder) -> None:
    """A boundary precedes an image-first prompt; tool-only messages retain identity."""
    session_builder.add("assistant", message={"id": "before", "content": "preamble"})
    session_builder.add("user", uuid="prompt-uuid", promptId="prompt-id", message={"content": [
        png_block,
        {"type": "text", "text": "first"}, {"type": "text", "text": "second"},
    ]})
    session_builder.add("assistant", message={"id": "m1", "content": [
        {"type": "thinking", "thinking": "reason"}, {"type": "text", "text": "reply"},
        {"type": "tool_use", "id": "call", "name": "Bash"},
    ]})
    session_builder.add("assistant", message={"id": "m1", "content": [{"type": "tool_use", "id": "next", "name": "Read"}]})
    session: Session = parse_session(session_builder.path)
    assert all(isinstance(event, TimedRecord) for event in session.main)
    assert all(event.turn_id == "" for event in session.main if event.file_index == 0)
    prompt_events: list[TimedRecord] = [event for event in session.main if event.file_index == 1]
    assert [type(event.payload) for event in prompt_events] == [TurnBoundary, Image, Prompt, Prompt]
    assert prompt_events[0].payload == TurnBoundary("start")
    assert prompt_events[0].prompt_id == "prompt-id"
    assert all(event.turn_id == "prompt-uuid" for event in session.main if event.file_index >= 1)
    assistant_events: list[TimedRecord] = [event for event in session.main if event.file_index >= 2]
    assert [type(event.payload) for event in assistant_events] == [Thinking, AssistantText, ToolCall, UsageSample, ToolCall]
    assert all(event.message_id == "m1" for event in assistant_events)


def test_parser_reads_only_inventoried_files(session_builder: SessionBuilder) -> None:
    """Children and offloaded outputs added after inventory wait for the next conversion."""
    output = session_builder.path.with_suffix("") / "tool-results/new.txt"
    session_builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": "t", "content": "preview"}]}, toolUseResult={"persistedOutputPath": str(output)})
    source = claude_session_source(session_builder.path)
    output.parent.mkdir(parents=True)
    output.write_text("full output")
    session_builder.add("user", path=session_builder.path.with_suffix("") / "subagents/agent-late.jsonl", message={"content": "late"})
    session = source.parse()
    assert not session.subagents
    assert [event.payload.text for event in session.main if isinstance(event.payload, ToolResult)] == ["preview"]
    updated = claude_session_source(session_builder.path).parse()
    assert set(updated.subagents) == {"late"}
    assert [event.payload.text for event in updated.main if isinstance(event.payload, ToolResult)] == ["full output"]


def test_damaged_lines_are_skipped_counted_and_warned(session_builder: SessionBuilder) -> None:
    """A line that is not valid JSON, in the main file or a subagent file, costs that line only."""


    session_builder.add("user", message={"content": "before"})
    with session_builder.path.open("ab") as stream:
        stream.write(b'{"type": "user", "mess\x00\x00\n')
    session_builder.add("user", message={"content": "after"})
    child: Path = session_builder.path.with_suffix("") / "subagents/agent-child.jsonl"
    session_builder.add("user", path=child, message={"content": "child"})
    with child.open("ab") as stream:
        stream.write(b'{"type": "user", "message": {"con\n')
    with pytest.warns(UserWarning) as caught:
        session = parse_session(session_builder.path)
    assert sorted(str(warning.message).split(": ")[0] for warning in caught) == sorted([f"{session_builder.path}:2", f"{child}:2"])
    assert session.skipped["damaged-line"] == 2
    assert [e.payload.text for e in session.main if isinstance(e.payload, Prompt)] == ["before", "after"]
    assert [e.payload.text for e in session.subagents["child"] if isinstance(e.payload, Prompt)] == ["child"]


def test_replay_keeps_first_record_and_does_not_repeat_turns(session_builder: SessionBuilder) -> None:
    """Re-appending a transcript block changes only the replay count."""
    session_builder.add("user", message={"content": "request"})
    session_builder.add("assistant", message={"id": "answer", "content": "done", "usage": {"output_tokens": 7}})
    before = parse_session(session_builder.path)
    original = session_builder.path.read_bytes()
    with session_builder.path.open("ab") as stream:
        stream.write(original)
    after = parse_session(session_builder.path)
    assert after.main == before.main
    assert aggregate_turns(after.main) == aggregate_turns(before.main)
    assert after.skipped == {"replayed-record": 2}


def test_nested_children_are_parsed_and_fingerprinted(session_builder: SessionBuilder) -> None:
    """Workflow children and colliding names are all recording inputs."""
    session_builder.add("user", message={"content": "parent"})
    directory = session_builder.path.with_suffix("") / "subagents"
    child = directory / "workflows/flow/agent-child.jsonl"
    session_builder.add("user", path=child, message={"content": "nested"})
    source = claude_session_source(session_builder.path)
    assert source.parse().subagents["child"][-1].payload == Prompt("nested")
    assert child.resolve() in source.inputs
    before = fingerprint(source.inputs)
    session_builder.add("assistant", path=child, message={"content": "changed"})
    assert fingerprint(source.inputs) != before
    session_builder.add("user", path=directory / "agent-child.jsonl", message={"content": "direct"})
    assert set(claude_session_source(session_builder.path).parse().subagents) == {"child"}


def test_split_message_usage_is_finalized_at_last_record(session_builder: SessionBuilder) -> None:
    """Provisional output counters are replaced, without dropping either text block."""
    session_builder.add("assistant", message={"id": "split", "content": "first", "usage": {"output_tokens": 1}})
    session_builder.add("assistant", message={"id": "split", "content": "last", "usage": {"output_tokens": 9}})
    session = parse_session(session_builder.path)
    samples = [row for row in session.main if isinstance(row.payload, UsageSample)]
    assert len(samples) == 1
    assert isinstance(samples[0].payload, UsageSample)
    assert samples[0].payload.usage.output_tokens == 9
    assert samples[0].file_index == 1
    assert [row.payload.text for row in session.main if isinstance(row.payload, AssistantText)] == ["first", "last"]


def test_injected_messages_stay_visible_without_starting_turns(session_builder: SessionBuilder) -> None:
    """Harness text is context; slash commands are human prompts."""
    session_builder.add("user", message={"content": "request"})
    for text in ("<task-notification>done", "<paseo-system>continue", "<local-command-stdout>output", "[Request interrupted"):
        session_builder.add("user", message={"content": text})
    session_builder.add("user", isMeta=True, message={"content": "instruction"})
    session_builder.add("user", message={"content": "<command-name>/help</command-name>"})
    session = parse_session(session_builder.path)
    assert len(aggregate_turns(session.main)) == 2
    assert [row.payload.role for row in session.main if isinstance(row.payload, Prompt)] == ["human", *(["injected"] * 5), "human"]


def test_child_files_merge_by_agent_id(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Both files enter the fingerprint and one time-ordered child, including split usage."""
    session_builder.add("user", message={"content": "parent"})
    directory = session_builder.path.with_suffix("") / "subagents"
    workflow = directory / "workflows/flow/agent-child.jsonl"
    direct = directory / "agent-child.jsonl"
    session_builder.add("assistant", path=workflow, message={"id": "split", "content": "first", "usage": {"output_tokens": 1}})
    session_builder.add("user", path=workflow, message={"content": "continue"})
    session_builder.add("assistant", path=direct, message={"id": "split", "content": "last", "usage": {"output_tokens": 9}})
    session_builder.add("assistant", path=workflow, message={"content": "end"})
    source = claude_session_source(session_builder.path)
    assert {workflow, direct} <= set(source.inputs)
    session = source.parse()
    assert set(session.subagents) == {"child"}
    rows = session.subagents["child"]
    assert [row.payload.text for row in rows if isinstance(row.payload, (Prompt, AssistantText))] == ["first", "continue", "last", "end"]
    assert [row.timestamp_ns for row in rows] == sorted(row.timestamp_ns for row in rows)
    assert [row.payload.usage.output_tokens for row in rows if isinstance(row.payload, UsageSample)] == [9]
    entities = read_entities(write_session_rrd(session, tmp_path / "merged.rrd").path)
    assert not any("/agents/" in path for path in entities)
    for child in (workflow, direct):
        before = fingerprint(source.inputs)
        original = child.read_bytes()
        child.write_bytes(original + b"\n")
        assert fingerprint(source.inputs) != before
        child.write_bytes(original)
    with direct.open("ab") as stream:
        stream.write(workflow.read_bytes())
    replayed = source.parse()
    assert [(row.timestamp_ns, row.payload) for row in replayed.subagents["child"]] == [(row.timestamp_ns, row.payload) for row in rows]
    assert replayed.skipped["replayed-record"] == 3


@pytest.mark.parametrize("with_result", [False, True])
def test_mixed_prompt_blocks_start_one_human_turn(session_builder: SessionBuilder, with_result: bool) -> None:
    """The injected summary is retained but excluded from the human turn prompt."""
    session_builder.add("user", message={"content": [
        *([{"type": "tool_result", "content": "tool output"}] if with_result else []),
        {"type": "text", "text": "<chat-history-summary>earlier</chat-history-summary>"},
        {"type": "text", "text": "keep going"},
    ]})
    session = parse_session(session_builder.path)
    assert [row.payload.role for row in session.main if isinstance(row.payload, Prompt)] == ["injected", "human"]
    turns = aggregate_turns(session.main)
    assert len(turns) == 1
    assert turns[0].prompt == "keep going"


@pytest.mark.parametrize(("origin", "text", "role"), [
    ({"turnOrigin": "task_notification"}, "task finished", "injected"),
    ({"turnOrigin": "peer"}, "peer message", "injected"),
    ({"turnOrigin": "scheduled"}, "scheduled message", "injected"),
    ({"turnOrigin": "system"}, "system message", "injected"),
    ({"promptSource": "system", "turnOrigin": "human"}, "system wins", "injected"),
    ({"isMeta": True, "promptSource": "typed"}, "meta wins", "injected"),
    ({"promptSource": "typed"}, "<task-notification>quoted tag", "human"),
    ({"turnOrigin": "human"}, "<chat-history-summary>quoted tag", "human"),
    ({"promptSource": "sdk", "turnOrigin": "sdk"}, "sdk request", "human"),
    ({"promptSource": "sdk"}, "<paseo-system>injected", "injected"),
])
def test_explicit_prompt_origins(session_builder: SessionBuilder, origin: dict[str, object], text: str, role: str) -> None:
    """Explicit origins take precedence over tags; SDK records still use text prefixes."""
    session_builder.add("user", path=None, **origin, message={"content": [{"type": "text", "text": text}]})
    session = parse_session(session_builder.path)
    assert [row.payload.role for row in session.main if isinstance(row.payload, Prompt)] == [role]
    assert len(aggregate_turns(session.main)) == int(role == "human")


@pytest.mark.parametrize("name,kind", [("Bash", "shell"), ("Read", "file_read"), ("Edit", "file_edit"), ("Write", "file_edit"), ("WebFetch", "web_search"), ("WebSearch", "web_search"), ("mcp__server__tool", "mcp"), ("Agent", "subagent"), ("Workflow", "subagent"), ("Unknown", "other")])
def test_claude_tool_kind_and_turn_model_effort(session_builder: SessionBuilder, tmp_path: Path, name: str, kind: str) -> None:
    """Provider metadata survives the shared recording boundary."""
    session_builder.add("user", message={"content": "inspect"})
    session_builder.add("assistant", effort="high", message={"id": "m", "model": "claude-test", "content": [{"type": "tool_use", "id": "c", "name": name, "input": {}}]})
    session_builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": "c", "content": "ok"}]})
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "kinds.rrd").path)
    entity: str = "mcp/server/tool" if kind == "mcp" else name
    assert entities["/tools"]["kind"].to_pylist() == [[kind], [kind]]
    assert entities["/tools"]["tool"].to_pylist() == [[entity], [entity]]
    assert entities["/turns"]["model"].to_pylist() == [["claude-test"]]
    assert entities["/turns"]["effort"].to_pylist() == [["high"]]


def test_signed_thinking_is_visible_but_never_current(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Opaque signatures have size placeholders, while current holds messages."""
    session_builder.add("user", message={"content": "hello"})
    session_builder.add("assistant", message={"content": [{"type": "thinking", "thinking": "", "signature": "opaque"}]})
    entities = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "thinking.rrd").path)
    assert entities["/conversation/thinking"]["TextLog:text"].to_pylist() == [["<encrypted reasoning, 6 bytes>"]]
    assert entities["/conversation/current"].num_rows == 1


def test_uuidless_same_time_pr_link_replay_is_deduplicated(session_builder: SessionBuilder) -> None:
    """Repeated UUID-less links at one timestamp are replays, later links are events."""
    for timestamp in ["2026-09-18T20:00:00Z", "2026-09-18T20:00:00Z", "2026-09-18T20:00:01Z"]:
        session_builder.add("pr-link", uuid="", timestamp=timestamp, prNumber=7, prUrl="https://example.test/pr/7")
    session = parse_session(session_builder.path)
    assert len(session.main) == 2
    assert session.skipped["replayed-record"] == 1


def test_workflow_harness_is_injected(session_builder: SessionBuilder) -> None:
    """Workflow child instructions do not create a human turn."""
    session_builder.add("user", message={"content": "[Workflow harness — computed task] instructions"})
    session = parse_session(session_builder.path)
    assert [row.payload.role for row in session.main if isinstance(row.payload, Prompt)] == ["injected"]
    assert not any(isinstance(row.payload, TurnBoundary) for row in session.main)


def test_provider_properties_have_declared_types(session_builder: SessionBuilder) -> None:
    """Every property emitted by this provider belongs to the shared catalog schema."""
    session_builder.add("user", message={"content": "hello"})
    session: Session = parse_session(session_builder.path)
    assert session.properties.keys() <= PROPERTY_TYPES.keys()


@pytest.mark.parametrize("reference_kind", ["persisted", "marker", "saved-sentence"])
def test_moved_home_keeps_offloaded_text_and_images(png_block: dict[str, object], tmp_path: Path, png_bytes: bytes, reference_kind: str) -> None:
    """Copied homes resolve original absolute references through the captured session inventory."""
    import shutil

    original: Path = tmp_path / "original"
    builder: SessionBuilder = SessionBuilder(original / "projects/project/session.jsonl")
    output: Path = builder.path.with_suffix("") / "tool-results/nested/result.txt"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"complete output\r\nprogress\r" * 100)
    preview: str = f"Output saved to: {output}" if reference_kind == "marker" else "preview"
    if reference_kind == "saved-sentence":
        preview = f"Error: result exceeds maximum allowed tokens. Output has been saved to {output}.\nFormat: text"
    builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": "call", "content": [
        {"type": "text", "text": preview},
        png_block,
    ]}]}, toolUseResult={"persistedOutputPath": str(output)} if reference_kind == "persisted" else {})
    copied: Path = tmp_path / "copied"
    shutil.copytree(original, copied)
    shutil.rmtree(original)
    session: Session = parse_session(copied / builder.path.relative_to(original))
    result = next(row.payload for row in session.main if isinstance(row.payload, ToolResult))
    image = next(row.payload for row in session.main if isinstance(row.payload, Image))
    assert result.text == "complete output\r\nprogress\r" * 100
    assert image.blob == png_bytes
    assert image.call_id == "call"
    assert session.properties["n_inlined_outputs"] == 1


@pytest.mark.parametrize("reference", [
    "/old/other-session/tool-results/nested/result.txt",
    "/old/unrelated/result.txt",
    "/old/session-123/tool-results/../tool-results/nested/result.txt",
    "/old/session-123/tool-results/link.txt",
    "/old/session-123/tool-results/missing/result.txt",
])
def test_moved_output_rejects_unrelated_or_escaping_paths(session_builder: SessionBuilder, reference: str) -> None:
    """A matching basename cannot bypass session identity, relative path, or containment."""
    results: Path = session_builder.path.with_suffix("") / "tool-results"
    (results / "nested").mkdir(parents=True)
    (results / "nested/result.txt").write_text("must not inline")
    outside: Path = results.parent / "outside.txt"
    outside.write_text("outside")
    (results / "link.txt").symlink_to(outside)
    session_builder.add("user", message={"content": [{"type": "tool_result", "content": "preview"}]},
                        toolUseResult={"persistedOutputPath": reference})
    session: Session = parse_session(session_builder.path)
    assert next(row.payload.text for row in session.main if isinstance(row.payload, ToolResult)) == "preview"
    assert session.properties["n_inlined_outputs"] == 0
    assert session.skipped["offloaded-output-missing"] == 1


def test_synthetic_models_and_api_errors(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Synthetic assistant records never label models, and API errors use an error lifecycle row."""
    session_builder.add("user", message={"content": "question"})
    session_builder.add("assistant", isApiErrorMessage=True, message={"id": "error", "model": "<synthetic>", "content": [{"type": "text", "text": "API unavailable"}]})
    session_builder.add("assistant", message={"id": "empty", "model": "<synthetic>", "content": [{"type": "text", "text": "No response requested."}]})
    session_builder.add("assistant", message={"id": "answer", "model": "real-model", "content": [{"type": "text", "text": "answer"}]})
    session: Session = parse_session(session_builder.path)
    assert session.properties["models"] == "real-model"
    assert all(row.model != "<synthetic>" for row in session.main)
    assert aggregate_turns(session.main)[0].model == "real-model"
    entities = read_entities(write_session_rrd(session, tmp_path / "errors.rrd").path)
    assert entities["/lifecycle/api_errors"]["TextLog:text"].to_pylist() == [["API unavailable"]]
    assert entities["/lifecycle/api_errors"]["TextLog:level"].to_pylist() == [["ERROR"]]
    assert entities["/conversation/assistant"]["TextLog:text"].to_pylist() == [["No response requested."], ["answer"]]


@pytest.mark.parametrize("emits_image", [True, False])
def test_tool_image_metadata_keeps_keys_without_repeating_emitted_bytes(png_base64: str, png_block: dict[str, object], session_builder: SessionBuilder, png_bytes: bytes, tmp_path: Path, emits_image: bool) -> None:
    """Only image bytes with a media row are replaced; text and unknown metadata remain exact."""

    import orjson

    encoded: str = png_base64
    metadata: dict[str, object] = {"type": "image", "file": {"base64": encoded, "path": "picture.png", "width": 1},
                                 "stdout": "full stdout", "structuredPatch": ["patch"], "future": {"key": 7}}
    content: list[dict[str, object]] = [{"type": "text", "text": "preview"}]
    if emits_image:
        content.append(png_block)
    session_builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": "read", "content": content}]}, toolUseResult=metadata)
    entities = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "metadata.rrd").path)
    expected: dict[str, object] = {**metadata, "file": {"base64": f"<image {len(png_bytes)} bytes, stored as media/images>" if emits_image else encoded,
                                                       "path": "picture.png", "width": 1}}
    metadata_json = entities["/tools"]["tool_use_result_json"].to_pylist()[0][0]
    assert isinstance(metadata_json, str)
    assert orjson.loads(metadata_json) == expected
    if emits_image:
        assert entities["/media/images"]["EncodedImage:blob"].to_pylist() == [[list(png_bytes)]]


def test_tool_search_references_keep_names(session_builder: SessionBuilder) -> None:
    """ToolSearch results in children retain names even without toolUseResult metadata."""
    child: Path = session_builder.path.with_suffix("") / "subagents/agent-search.jsonl"
    session_builder.add("user", message={"content": "search"})
    session_builder.add("user", path=child, message={"content": [{"type": "tool_result", "tool_use_id": "search", "content": [
        {"type": "tool_reference", "tool_name": "Monitor"},
        {"type": "tool_reference", "tool_name": "Read"},
    ]}]})
    result = next(row.payload for row in parse_session(session_builder.path).subagents["search"] if isinstance(row.payload, ToolResult))
    assert result.text == "Monitor\nRead"


def test_workflow_sidecars_keep_content_and_change_fingerprint(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Workflow runs and scripts participate in conversion and change detection."""
    import orjson

    from agent_traces.claude import session_source
    from agent_traces.sources import fingerprint

    session_builder.add("user", message={"content": "run workflow"})
    directory = session_builder.path.with_suffix("") / "workflows"
    (directory / "scripts").mkdir(parents=True)
    run = directory / "wf_run.json"
    run.write_bytes(orjson.dumps({"runId": "run", "timestamp": "2026-09-18T20:00:02Z", "status": "failed",
        "error": "worker failed", "result": {"partial": "kept"}, "phases": [{"title": "inspect", "detail": "files"}],
        "logs": ["started"], "totalTokens": 17}))
    script = directory / "scripts/program.js"
    script.write_bytes(b"const answer = 42;\r\n")
    source = session_source(session_builder.path)
    assert {run, script} <= set(source.inputs)
    before = fingerprint(source.inputs)
    entities = read_entities(write_session_rrd(source.parse(), tmp_path / "workflow.rrd").path)
    row = entities["/lifecycle/workflows"]
    assert row["TextLog:level"].to_pylist() == [["ERROR"]]
    assert orjson.loads(row["TextLog:text"].to_pylist()[0][0])["result"] == {"partial": "kept"}
    assert metadata_values(row, 'run_id') == [["run"]]
    assert metadata_values(row, 'total_tokens') == [[17]]
    assert '"inspect"' in row["TextLog:text"].to_pylist()[0][0]
    assert entities["/lifecycle/workflow_scripts"]["TextLog:text"].to_pylist() == [["const answer = 42;\r\n"]]
    script.write_text("const answer = 43;")
    assert fingerprint(session_source(session_builder.path).inputs) != before
    before = fingerprint(source.inputs)
    run.write_text(run.read_text().replace('"failed"', '"completed"'))
    assert fingerprint(session_source(session_builder.path).inputs) != before


def test_child_metadata_is_inventoried_and_recorded(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Claude agent labels survive the numbered layout and take part in the fingerprint."""
    from agent_traces.claude import session_source
    from agent_traces.sources import fingerprint

    session_builder.add("user", message={"content": "delegate"})
    child: Path = session_builder.path.with_suffix("") / "subagents/agent-review.jsonl"
    session_builder.add("user", path=child, message={"content": "review"})
    metadata = child.with_suffix(".meta.json")
    metadata.write_text('{"agentType":"reviewer","description":"Check the changes"}')
    source = session_source(session_builder.path)
    assert metadata in source.inputs
    before = fingerprint(source.inputs)
    entities = read_entities(write_session_rrd(source.parse(), tmp_path / "labels.rrd").path)
    assert entities["/__properties/agents"]["agent_type"].to_pylist() == [["reviewer"]]
    assert entities["/__properties/agents"]["description"].to_pylist() == [["Check the changes"]]
    metadata.write_text('{"agentType":"reviewer","description":"Check tests"}')
    assert fingerprint(session_source(session_builder.path).inputs) != before


@pytest.mark.parametrize(("reference", "persisted", "present", "missing"), [
    ("a file and the path is printed.", False, False, 0),
    ("./", False, False, 0),
    ("result.txt", True, False, 1),
    ("tool-results/result.txt", False, False, 1),
    ("tool-results/result.txt", False, True, 0),
])
def test_missing_outputs_count_references_not_prose(session_builder: SessionBuilder, reference: str, persisted: bool, present: bool, missing: int) -> None:
    """Only actual output references count as lost previews; present files inline."""
    results = session_builder.path.with_suffix("") / "tool-results"
    if present:
        results.mkdir(parents=True)
        (results / "result.txt").write_text("full output")
    session_builder.add("user", toolUseResult={"persistedOutputPath": reference} if persisted else {},
                        message={"content": [{"type": "tool_result", "content": f"Output saved to {reference}"}]})
    session = parse_session(session_builder.path)
    assert session.skipped["offloaded-output-missing"] == missing
    result = next(row.payload for row in session.main if isinstance(row.payload, ToolResult))
    assert result.text == ("full output" if present else f"Output saved to {reference}")


def test_mixed_result_parts_keep_order(session_builder: SessionBuilder) -> None:
    """Text and tool references retain their source order with line separators."""
    session_builder.add("user", message={"content": [{"type": "tool_result", "content": [
        {"type": "text", "text": "before"}, {"type": "tool_reference", "tool_name": "Read"}, {"type": "text", "text": "after"},
    ]}]})
    result = next(row.payload for row in parse_session(session_builder.path).main if isinstance(row.payload, ToolResult))
    assert result.text == "before\nRead\nafter"


@pytest.mark.parametrize("timed_child", [False, True])
def test_workflow_only_main_uses_whole_session_time(session_builder: SessionBuilder, tmp_path: Path, timed_child: bool) -> None:
    """Untimed workflow scripts use the first child or workflow time, never the epoch."""
    session_builder.add("file-history-snapshot", timestamp=None)
    workflows = session_builder.path.with_suffix("") / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "scripts").mkdir()
    (workflows / "scripts/script.js").write_text("const result = 1;")
    (workflows / "wf_run.json").write_text('{"timestamp":"2026-09-18T20:00:02Z"}')
    if timed_child:
        session_builder.add("user", path=session_builder.path.with_suffix("") / "subagents/agent-child.jsonl",
                            timestamp="2026-09-18T20:00:01Z", message={"content": "first"})
    entities = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "workflow-time.rrd").path)
    expected = 1789761601000000000 if timed_child else 1789761602000000000
    assert entities["/lifecycle/workflow_scripts"]["wall"].cast(pa.int64()).to_pylist() == [expected]
    assert entities["/__properties"]["RecordingInfo:start_time"].combine_chunks().values.cast(pa.int64()).to_pylist() == [expected]
