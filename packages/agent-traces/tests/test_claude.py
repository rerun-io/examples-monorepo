"""Tests at the streaming parser and session boundary."""

from collections.abc import Iterator

from agent_traces.claude import ClaudeSession, iter_records, parse_session
from agent_traces.claude_records import Record, ResultContent
from tests.conftest import SessionBuilder


def test_streams_typed_records_and_preserves_line_indices(session_builder: SessionBuilder) -> None:
    """Unknown fields are allowed and file order remains available."""
    session_builder.add("user", message={"role": "user", "content": "hello"}, future_field=True)
    session_builder.add("assistant", message={"id": "m1", "content": [{"type": "text", "text": "hi"}]})
    records: Iterator[Record] = iter_records(session_builder.path)
    assert iter(records) is records
    first: Record = next(records)
    assert first.message is not None
    assert first.message.content == "hello"
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
    assert session.main[0].record.message.content[0].tool_use_id == "t1"
    assert list(session.subagents) == ["child"]
    assert session.subagents["child"][0].file_index == 0


def test_bad_lines_report_source_and_line_without_reading_ahead(session_builder: SessionBuilder) -> None:
    """A valid first line is yielded before the bad second line is decoded."""
    import pytest

    session_builder.add("user", message={"content": "valid"})
    with session_builder.path.open("ab") as stream:
        stream.write(b"{broken\n")
    records: Iterator[Record] = iter_records(session_builder.path)
    assert next(records).type == "user"
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
        message={"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "preview"}]},
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
        content: str | list[ResultContent] | None = row.record.message.content[0].content
        assert isinstance(content, str) or content is None
        actual.append(content)
    assert actual == ["full tool output", "full tool output", "missing", "outside"]
    assert session.n_inlined_outputs == 2
    assert session.main[0].record.toolUseResult is not None
    assert session.main[0].record.toolUseResult.agentId == "child"
    assert '"future":42' in session.main[0].record.tool_use_result_json


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
