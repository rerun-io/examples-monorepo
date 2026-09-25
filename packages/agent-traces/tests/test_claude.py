"""Tests at the streaming parser and session boundary."""

from pathlib import Path

from agent_traces.claude import iter_records, parse_session
from agent_traces.claude_records import Record, TextBlock
from agent_traces.events import Image, Lifecycle, Prompt, Session, ToolResult
from agent_traces.sources import DamagedLine
from tests.conftest import SessionBuilder


def test_streams_typed_records_and_preserves_line_indices(session_builder: SessionBuilder) -> None:
    """Unknown fields are allowed and file order remains available."""
    session_builder.add("user", message={"role": "user", "content": "hello"}, future_field=True)
    session_builder.add("assistant", message={"id": "m1", "content": [{"type": "text", "text": "hi"}]})
    records = iter_records(session_builder.path)
    assert iter(records) is records
    first_line = next(records)
    assert not isinstance(first_line, DamagedLine)
    first: Record = first_line.record
    assert first.message is not None
    assert first.message.content == [TextBlock(text="hello")]
    session: Session = parse_session(session_builder.path)
    assert session.session_id == "session-123"
    assert session.profile == "claude-alt"
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
    import pytest

    session_builder.add("user", message={"content": "valid"})
    with session_builder.path.open("ab") as stream:
        stream.write(b"{broken\n")
    records = iter_records(session_builder.path)
    first_line = next(records)
    assert not isinstance(first_line, DamagedLine)
    assert first_line.record.type == "user"
    assert next(records) == DamagedLine(session_builder.path, 2)
    session_builder.path.write_bytes(b'{"type": "user", "message": {"content": 123}}\n')
    with pytest.raises(ValueError, match=r"session-123.jsonl:1"):
        list(iter_records(session_builder.path))


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
    assert session.n_inlined_outputs == 2
    assert results_emitted[0].agent_id == "child"
    assert '"future":42' in results_emitted[0].raw_json


def test_inlines_list_result_text_without_removing_images(session_builder: SessionBuilder) -> None:
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
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
                    ],
                }
            ]
        },
    )
    session: Session = parse_session(session_builder.path)
    assert isinstance(session.main[0].payload, ToolResult)
    assert session.main[0].payload.text == "complete output"
    assert isinstance(session.main[1].payload, Image)
    assert session.main[1].payload.blob == b"\x00"
    assert session.main[1].payload.media_type == "image/png"
    assert session.main[1].payload.source == "tool_result"
    assert session.n_inlined_outputs == 1


def test_invalid_record_shape_and_timestamp_name_the_source(session_builder: SessionBuilder) -> None:
    """Malformed record envelopes and timestamps have actionable diagnostics."""
    import pytest

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
    import pytest

    from agent_traces.claude_records import TextBlock, UnknownBlock

    session_builder.add("user", message={"content": "hello"})
    session_builder.add("assistant", message={"content": [{"type": "future", "text": 42}, {}]})
    session: Session = parse_session(session_builder.path)
    assert [row.payload.text for row in session.main if isinstance(row.payload, Prompt)] == ["hello"]
    records = [line.record for line in iter_records(session_builder.path) if not isinstance(line, DamagedLine)]
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
    from pathlib import Path

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
    import pytest

    from agent_traces.claude import parse_timestamp_ns

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
    assert [timed.values["subtype"] for timed in session.main if isinstance(timed.payload, Lifecycle)] == ["hook_success"]


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
    assert session.n_inlined_outputs == 1
    result = session.main[-1].payload
    assert isinstance(result, ToolResult) and result.text == "ok \ufffd bad"


def test_session_sources_include_sorted_recursive_outputs(session_builder: SessionBuilder) -> None:
    """Discovery includes ignored inputs but parses only child transcripts."""
    from agent_traces.claude import session_source

    session_builder.add("user", message={"content": "main"})
    root: Path = session_builder.path.with_suffix("")
    for name in ["z", "a"]:
        session_builder.add("user", path=root / f"subagents/agent-{name}.jsonl", message={"content": name})
    (root / "tool-results/pdf-id").mkdir(parents=True)
    (root / "tool-results/z.txt").write_text("output")
    (root / "tool-results/pdf-id/page.jpg").write_bytes(b"image")
    (root / "tool-results/agent-ignored.jsonl").write_text("not a transcript")
    (root / "subagents/agent-directory.jsonl").mkdir()
    assert list(session_source(session_builder.path).inputs) == [
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
    assert session.n_inlined_outputs == 0


def test_flat_events_carry_turn_and_assistant_identity(session_builder: SessionBuilder) -> None:
    """A boundary precedes an image-first prompt; tool-only messages retain identity."""
    from agent_traces.events import AssistantText, Thinking, TimedRecord, ToolCall, TurnBoundary, UsageSample

    session_builder.add("assistant", message={"id": "before", "content": "preamble"})
    session_builder.add("user", uuid="prompt-uuid", promptId="prompt-id", message={"content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
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
    assert [type(event.payload) for event in assistant_events] == [UsageSample, Thinking, AssistantText, ToolCall, ToolCall]
    assert all(event.message_id == "m1" for event in assistant_events)


def test_parser_reads_only_inventoried_files(session_builder: SessionBuilder) -> None:
    """Children and offloaded outputs added after inventory wait for the next conversion."""
    from agent_traces.claude import session_source
    from agent_traces.events import ToolResult

    output = session_builder.path.with_suffix("") / "tool-results/new.txt"
    session_builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": "t", "content": "preview"}]}, toolUseResult={"persistedOutputPath": str(output)})
    source = session_source(session_builder.path)
    output.parent.mkdir(parents=True)
    output.write_text("full output")
    session_builder.add("user", path=session_builder.path.with_suffix("") / "subagents/agent-late.jsonl", message={"content": "late"})
    session = source.parse()
    assert not session.subagents
    assert [event.payload.text for event in session.main if isinstance(event.payload, ToolResult)] == ["preview"]
    updated = session_source(session_builder.path).parse()
    assert set(updated.subagents) == {"late"}
    assert [event.payload.text for event in updated.main if isinstance(event.payload, ToolResult)] == ["full output"]


def test_damaged_lines_are_skipped_counted_and_warned(session_builder: SessionBuilder) -> None:
    """A line that is not valid JSON, in the main file or a subagent file, costs that line only."""
    import pytest

    from agent_traces.events import Prompt

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
