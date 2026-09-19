"""Tests at the streaming parser and session boundary."""

from collections.abc import Iterator

from agent_traces.claude import ClaudeSession, SourceRecord, iter_records, parse_session
from agent_traces.claude_records import Record, ResultContent, TextBlock, ToolResultBlock
from tests.conftest import SessionBuilder


def test_streams_typed_records_and_preserves_line_indices(session_builder: SessionBuilder) -> None:
    """Unknown fields are allowed and file order remains available."""
    session_builder.add("user", message={"role": "user", "content": "hello"}, future_field=True)
    session_builder.add("assistant", message={"id": "m1", "content": [{"type": "text", "text": "hi"}]})
    records: Iterator[SourceRecord] = iter_records(session_builder.path)
    assert iter(records) is records
    first: Record = next(records).record
    assert first.message is not None
    assert first.message.content == [TextBlock(text="hello")]
    session: ClaudeSession = parse_session(session_builder.path)
    assert session.session_id == "session-123"
    assert session.profile == "claude-alt"
    assert [row.file_index for row in session.main] == [0, 1]
    assert session.main[0].timestamp_ns == 1_789_761_600_000_000_000


def test_skips_noise_and_folds_subagents_without_losing_tool_results(session_builder: SessionBuilder) -> None:
    """Count noise by kind, retain typed results, and preserve child ids."""
    session_builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "done"}]})
    session_builder.add("attachment", attachment={"type": "total_tokens_reminder"})
    session_builder.add("queue-operation")
    session_builder.add("future-kind")
    session_builder.add("system", timestamp=None)
    session_builder.add("user", path=session_builder.path.with_suffix("") / "subagents/agent-child.jsonl", message={"content": "child"})
    session: ClaudeSession = parse_session(session_builder.path)
    assert session.skipped == {"total_tokens_reminder": 1, "queue-operation": 1, "future-kind": 1, "system": 1}
    assert len(session.main) == 1
    assert session.main[0].record.message is not None
    assert isinstance(session.main[0].record.message.content, list)
    assert isinstance(session.main[0].record.message.content[0], ToolResultBlock)
    assert session.main[0].record.message.content[0].tool_use_id == "t1"
    assert list(session.subagents) == ["child"]
    assert session.subagents["child"][0].file_index == 0


def test_bad_lines_report_source_and_line_without_reading_ahead(session_builder: SessionBuilder) -> None:
    """A valid first line is yielded before the bad second line is decoded."""
    import pytest

    session_builder.add("user", message={"content": "valid"})
    with session_builder.path.open("ab") as stream:
        stream.write(b"{broken\n")
    records: Iterator[SourceRecord] = iter_records(session_builder.path)
    assert next(records).record.type == "user"
    with pytest.raises(ValueError, match=r"session-123.jsonl:2"):
        next(records)
    session_builder.path.write_bytes(b'{"type": "user", "message": {"content": 123}}\n')
    with pytest.raises(ValueError, match=r"session-123.jsonl:1"):
        list(iter_records(session_builder.path))


def test_inlines_only_outputs_inside_session_tool_results(session_builder: SessionBuilder) -> None:
    """Both CLI output references and persisted paths resolve within the session."""
    from pathlib import Path

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
    session: ClaudeSession = parse_session(session_builder.path)
    actual: list[str | None] = []
    for row in session.main:
        assert row.record.message is not None
        assert isinstance(row.record.message.content, list)
        assert isinstance(row.record.message.content[0], ToolResultBlock)
        content: str | list[ResultContent] | None = row.record.message.content[0].content
        assert isinstance(content, str) or content is None
        actual.append(content)
    assert actual == ["full tool output", "full tool output", "missing", "outside"]
    assert session.n_inlined_outputs == 2
    assert session.main[0].record.toolUseResult is not None
    assert session.main[0].record.toolUseResult.agentId == "child"
    assert '"future":42' in session.main[0].tool_use_result_json


def test_inlines_list_result_text_without_removing_images(session_builder: SessionBuilder) -> None:
    """Persisted text replaces previews while image content stays attached."""
    from pathlib import Path

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
    session: ClaudeSession = parse_session(session_builder.path)
    record: Record = session.main[0].record
    assert record.message is not None and isinstance(record.message.content, list)
    assert isinstance(record.message.content[0], ToolResultBlock)
    content: str | list[ResultContent] | None = record.message.content[0].content
    assert isinstance(content, list)
    assert [part.type for part in content] == ["text", "image"]
    assert content[0].text == "complete output"
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
    session: ClaudeSession = parse_session(session_builder.path)
    assert session.main[0].timestamp_ns == 1_789_761_600_123_456_789


def test_content_boundary_normalizes_and_rejects_malformed_known_blocks(session_builder: SessionBuilder) -> None:
    """Known tags are strict; unknown tags retain only their identity."""
    import pytest

    from agent_traces.claude_records import TextBlock, UnknownBlock

    session_builder.add("user", message={"content": "hello"})
    session_builder.add("assistant", message={"content": [{"type": "future", "text": 42}, {}]})
    session: ClaudeSession = parse_session(session_builder.path)
    assert session.main[0].record.message is not None
    assert session.main[1].record.message is not None
    assert session.main[0].record.message.content == [TextBlock(text="hello")]
    assert session.main[1].record.message.content == [UnknownBlock(type="future"), UnknownBlock()]
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
    session: ClaudeSession = parse_session(session_builder.path)
    assert session.main[0].raw_json == ""
    assert '"unknown":{"nested":[null]}' in session.main[0].tool_use_result_json
    assert not hasattr(session.main[0].record, "raw_json")
    assert session.main[0].record.message is not None
    assert isinstance(session.main[0].record.message.content[0], ToolResultBlock)
    assert session.main[0].record.message.content[0].content == "full\ufffdoutput"


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
