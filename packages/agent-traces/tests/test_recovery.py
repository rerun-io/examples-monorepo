"""Recover complete records without guessing at invalid schemas."""

from pathlib import Path

import pytest

from agent_traces.apis import convert, convert_all
from agent_traces.events import AssistantText, Prompt
from tests.conftest import RolloutBuilder, SessionBuilder, parse_rollout, parse_session


@pytest.mark.parametrize("prefix,counts", [(b"\0\0", {"nul-padded-line": 1}), (b'{"type":"user","message":', {"merged-line": 1, "damaged-line": 1})])
def test_complete_trailing_record_is_recovered(session_builder: SessionBuilder, prefix: bytes, counts: dict[str, int]) -> None:
    """A full suffix survives NUL padding or an interrupted prefix."""
    session_builder.add("user", message={"content": "first"})
    session_builder.add("user", message={"content": "recovered"})
    lines: list[bytes] = session_builder.path.read_bytes().splitlines(keepends=True)
    session_builder.path.write_bytes(lines[0] + prefix + lines[1] + b'{"type":')
    with pytest.warns(UserWarning):
        session = parse_session(session_builder.path)
    assert [row.payload.text for row in session.main if isinstance(row.payload, Prompt)] == ["first", "recovered"]
    assert session.skipped == {**counts, "damaged-line": counts.get("damaged-line", 0) + 1}


def test_damaged_first_line_converts_through_both_commands(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Single-file detection finds the first decodable record; batch uses layout."""
    session_builder.add("user", message={"content": "kept"})
    session_builder.path.write_bytes(b"{truncated\n" + session_builder.path.read_bytes())
    with pytest.warns(UserWarning) as single_warnings:
        convert.main(convert.Config(session=session_builder.path, out=tmp_path / "single.rrd"))
    with pytest.warns(UserWarning) as batch_warnings:
        convert_all.main(convert_all.Config(home=session_builder.path.parents[2], out=tmp_path / "batch"))
    assert len(single_warnings) == len(batch_warnings) == 1
    assert str(single_warnings[0].message) == str(batch_warnings[0].message) == f"{session_builder.path}:1: damaged-line"
    assert (tmp_path / "single.rrd").is_file()
    assert (tmp_path / "batch/claude/session-123.rrd").is_file()


def test_merged_suffix_must_pass_the_provider_schema(session_builder: SessionBuilder) -> None:
    """A JSON object nested in a broken record is not enough to recover a record."""
    session_builder.add("user", message={"content": "kept"})
    with session_builder.path.open("ab") as stream:
        stream.write(b'{broken{"type":"user","message":{"content":123}}\n')
    with pytest.warns(UserWarning):
        session = parse_session(session_builder.path)
    assert session.skipped == {"damaged-line": 1}
    with session_builder.path.open("ab") as stream:
        stream.write(b'{"type":"user","message":{"content":123}}\n')
    with pytest.warns(UserWarning), pytest.raises(ValueError):
        parse_session(session_builder.path)


def test_codex_line_recovery_preserves_envelope_and_payload(tmp_path: Path) -> None:
    """Both providers use the same strict recovery rule."""
    builder = RolloutBuilder(tmp_path / ".codex/sessions/rollout.jsonl")
    builder.meta()
    builder.add("event_msg", type="agent_message", message="padded")
    builder.add("event_msg", type="agent_message", message="merged")
    lines = builder.path.read_bytes().splitlines(keepends=True)
    builder.path.write_bytes(lines[0] + b"\0" + lines[1] + b'{"truncated":' + lines[2])
    with pytest.warns(UserWarning):
        session = parse_rollout(builder.path)
    assert [row.payload.text for row in session.main if isinstance(row.payload, AssistantText)] == ["padded", "merged"]
    assert session.skipped == {"nul-padded-line": 1, "merged-line": 1, "damaged-line": 1}


def test_single_claude_detection_accepts_untimed_metadata(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """A valid title record can precede the first timed message."""
    session_builder.add("custom-title", timestamp=None, customTitle="title")
    session_builder.add("user", message={"content": "hello"})
    convert.main(convert.Config(session=session_builder.path, out=tmp_path / "metadata.rrd"))
    assert (tmp_path / "metadata.rrd").is_file()


def test_codex_nul_padded_header_is_recovered(tmp_path: Path) -> None:
    """Header eligibility applies after removing NUL padding."""
    builder = RolloutBuilder(tmp_path / ".codex/sessions/rollout.jsonl")
    builder.meta()
    builder.path.write_bytes(b"\0\0" + builder.path.read_bytes())
    with pytest.warns(UserWarning):
        session = parse_rollout(builder.path)
    assert session.session_id == "thread"
    assert session.skipped == {"nul-padded-line": 1}
