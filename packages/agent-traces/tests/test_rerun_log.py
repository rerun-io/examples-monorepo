"""Read saved recordings to test the public writer boundary."""

from pathlib import Path

import pyarrow as pa
import pytest
from rerun.chunk import RrdReader

from agent_traces.claude import parse_session
from agent_traces.rerun_log import write_session_rrd
from tests.conftest import SessionBuilder


def read_entities(path: Path) -> dict[str, pa.Table]:
    """Read entity columns using the Rerun 0.37 file reader."""
    batches: dict[str, list[pa.RecordBatch]] = {}
    for chunk in RrdReader(path).stream().to_chunks():
        batches.setdefault(str(chunk.entity_path), []).append(chunk.to_record_batch())
    return {
        entity: pa.concat_tables([pa.Table.from_batches([batch]) for batch in parts], promote_options="default") for entity, parts in batches.items()
    }


def test_conversations_have_only_wall_time_and_keep_identity(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Text and thinking are sorted by time without an implicit log timeline."""
    session_builder.add("user", uuid="u1", parentUuid="p1", promptId="prompt", message={"content": "hello"})
    session_builder.add("user", message={"content": [{"type": "text", "text": "second"}]})
    session_builder.add("assistant", requestId="request", message={"id": "m1", "model": "model", "content": [{"type": "text", "text": "later"}]})
    session_builder.add(
        "assistant",
        timestamp="2026-09-18T20:00:01.500Z",
        message={
            "id": "m1",
            "content": [{"type": "text", "text": "earlier"}, {"type": "thinking", "thinking": "reason"}, {"type": "text", "text": "same time"}],
        },
    )
    session_builder.add("user", isCompactSummary=True, message={"content": "summary"})
    session_builder.add("user", path=session_builder.path.with_suffix("") / "subagents/agent-child.jsonl", message={"content": "child"})
    out: Path = write_session_rrd(parse_session(session_builder.path), tmp_path / "session.rrd")
    entities: dict[str, pa.Table] = read_entities(out)
    timelines: set[str] = {
        field.name for table in entities.values() for field in table.schema if (field.metadata or {}).get(b"rerun:kind") == b"index"
    }
    assert timelines == {"wall"}
    assert entities["/conversation/user"].num_rows == 2
    assert entities["/conversation/assistant"]["TextLog:text"].to_pylist() == [["earlier"], ["same time"], ["later"]]
    assert entities["/conversation/thinking"]["TextLog:level"].to_pylist() == [["DEBUG"]]
    assert entities["/conversation/compaction"]["TextLog:text"].to_pylist() == [["summary"]]
    assert entities["/conversation/user"]["uuid"].to_pylist()[0] == ["u1"]
    assert entities["/conversation/assistant"]["request_id"].to_pylist()[-1] == ["request"]
    assert entities["/conversation/assistant"]["wall"].cast(pa.int64()).to_pylist() == [1789761601500000000, 1789761601500000000, 1789761602000000000]
    assert entities["/agents/child/conversation/user"].num_rows == 1
    assert RrdReader(out).recordings()[0].recording_id == "session-123"


def test_tool_results_join_calls_and_preserve_images(session_builder: SessionBuilder, png_bytes: bytes, tmp_path: Path) -> None:
    """Tool users are not prompts; both image sources and MCP paths survive."""
    import base64
    import math

    image: dict[str, object] = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png_bytes).decode()},
    }
    session_builder.add(
        "assistant",
        message={"content": [{"type": "tool_use", "id": "t1", "name": "mcp__server__tool", "input": {"command": "echo", "nested": [1, True, None]}}]},
    )
    session_builder.add(
        "user",
        message={"content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "result"}, image], "is_error": True}]},
        toolUseResult={"agentId": "child"},
    )
    session_builder.add("user", message={"content": [image]})
    session_builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": "unknown", "content": "no call"}]})
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "tools.rrd"))
    tool: pa.Table = entities["/tools/mcp/server/tool"]
    assert tool.num_rows == 2
    assert tool["phase"].to_pylist() == [["call"], ["result"]]
    assert tool["TextLog:level"].to_pylist() == [["INFO"], ["ERROR"]]
    assert tool["input_json"].to_pylist()[0] == ['{"command":"echo","nested":[1,true,null]}']
    assert tool["result_text"].to_pylist()[-1] == ["result"]
    assert tool["elapsed_ms"].to_pylist()[-1] == [1000.0]
    assert tool["agent_id"].to_pylist()[-1] == ["child"]
    assert entities["/tools/elapsed_ms/mcp/server/tool"]["Scalars:scalars"].to_pylist() == [[1000.0]]
    assert math.isnan(entities["/tools/unknown"]["elapsed_ms"].to_pylist()[0][0])
    images: pa.Table = entities["/media/images"]
    assert images.num_rows == 2
    assert images["source"].to_pylist() == [["tool_result"], ["user"]]
    assert images["tool_use_id"].to_pylist() == [["t1"], [""]]
    assert images["EncodedImage:blob"].to_pylist() == [[list(png_bytes)], [list(png_bytes)]]
    assert "/conversation/user" not in entities


def test_usage_counts_first_row_per_message_id_per_agent(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Split assistant blocks repeat usage; missing counters become zero."""
    session_builder.add(
        "assistant",
        message={
            "id": "m1",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "cache_creation": {"ephemeral_5m_input_tokens": 1, "ephemeral_1h_input_tokens": 2},
                "output_tokens_details": {"thinking_tokens": 3},
            },
            "content": [],
        },
    )
    session_builder.add("assistant", message={"id": "m1", "usage": {"input_tokens": 99}, "content": []})
    session_builder.add("assistant", message={"id": "m2", "content": []})
    session_builder.add(
        "assistant",
        path=session_builder.path.with_suffix("") / "subagents/agent-child.jsonl",
        message={"id": "m1", "usage": {"input_tokens": 7}, "content": []},
    )
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "usage.rrd"))
    expected: dict[str, int] = {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_tokens": 3,
        "cache_creation_tokens": 2,
        "cache_creation_5m_tokens": 1,
        "cache_creation_1h_tokens": 2,
        "thinking_tokens": 3,
    }
    for name, value in expected.items():
        assert entities[f"/usage/{name}"]["Scalars:scalars"].to_pylist() == [[value], [0]]
    assert entities["/agents/child/usage/input_tokens"]["Scalars:scalars"].to_pylist() == [[7]]


def test_lifecycle_and_flat_recording_properties(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Lifecycle details and skipped metadata survive as rows and properties."""
    import hashlib
    import socket

    session_builder.add("user", cwd="/workspace", gitBranch="main", version="2.1.9", message={"content": "hello"})
    session_builder.add(
        "assistant", version="2.1.8", message={"model": "z-model", "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}]}
    )
    session_builder.add("assistant", message={"model": "a-model", "content": []})
    session_builder.add("system", subtype="api_error", level="warn", content="retry", future_detail=42)
    session_builder.add("system", subtype="compact_boundary")
    for subtype in ["queued_command", "command_permissions", "hook_success", "edited_text_file", "auto_mode"]:
        session_builder.add("attachment", attachment={"type": subtype, "text": "detail", "unknown": True})
    session_builder.add("attachment", attachment={"type": "skill_listing"})
    session_builder.add("pr-link", prNumber=123, prUrl="https://example.test/pr/123", prRepository="org/repo")
    session_builder.add("custom-title", timestamp=None, customTitle="old title")
    session_builder.add("ai-title", timestamp=None, aiTitle="last title")
    session_builder.add("cost-state", timestamp=None, totalCostUSD=1.0)
    session_builder.add("cost-state", timestamp=None, totalCostUSD=2.5)
    out: Path = write_session_rrd(parse_session(session_builder.path), tmp_path / "properties.rrd")
    entities: dict[str, pa.Table] = read_entities(out)
    system: pa.Table = entities["/lifecycle/system"]
    assert system["TextLog:text"].to_pylist() == [["retry"], ["compact_boundary"]]
    assert system["TextLog:level"].to_pylist() == [["WARN"], ["INFO"]]
    assert '"future_detail":42' in system["extra_json"].to_pylist()[0][0]
    assert entities["/lifecycle/attachments"].num_rows == 5
    assert entities["/lifecycle/attachments"]["TextLog:text"].to_pylist()[0] == ["queued_command: detail"]
    assert entities["/lifecycle/pr_links"]["pr_number"].to_pylist() == [[123]]
    props: pa.Table = entities["/__properties/session"]
    expected: dict[str, object] = {
        "session_id": "session-123",
        "profile": "claude-alt",
        "host": socket.gethostname(),
        "cwd": "/workspace",
        "git_branch": "main",
        "cli_versions": "2.1.8,2.1.9",
        "title": "last title",
        "models": "a-model,z-model",
        "n_subagents": 0,
        "n_tool_calls": 1,
        "n_images": 0,
        "n_inlined_outputs": 0,
        "total_cost_usd": 2.5,
        "source_path": str(session_builder.path.resolve()),
        "source_sha256": hashlib.sha256(session_builder.path.read_bytes()).hexdigest(),
    }
    for key, value in expected.items():
        assert props[key].to_pylist() == [[value]]
    assert entities["/__properties/skipped"]["cost-state"].to_pylist() == [[2]]
    assert entities["/__properties/skipped"]["skill_listing"].to_pylist() == [[1]]
    assert entities["/__properties"]["RecordingInfo:name"].drop_null().to_pylist() == [["claude session- last title"]]


def test_multiple_tool_entities_keep_sparse_columns_aligned(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Different tool row counts must not depend on SDK inference history."""
    for index, tool in enumerate(["Bash", "Read", "Read"]):
        session_builder.add("assistant", message={"content": [{"type": "tool_use", "id": str(index), "name": tool, "input": {}}]})
        session_builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": str(index), "content": "ok"}]})
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "multiple.rrd"))
    assert entities["/tools/Read"].num_rows == 4
    assert entities["/tools/Read"]["elapsed_ms"].to_pylist()[1::2] == [[1000.0], [1000.0]]


@pytest.mark.parametrize(
    ("timestamp", "expected_ns"),
    [("2026-09-18T15:00:00.123456789-05:00", 1_789_761_600_123_456_789), ("1969-12-31T23:59:59.999999999Z", -1)],
)
def test_all_row_families_preserve_nanoseconds(
    session_builder: SessionBuilder, png_bytes: bytes, tmp_path: Path, timestamp: str, expected_ns: int
) -> None:
    """Text, scalar, and image columns keep exact timestamps across the epoch."""
    import base64

    session_builder.add(
        "assistant", timestamp=timestamp, message={"id": "m1", "content": [{"type": "text", "text": "precise"}], "usage": {"input_tokens": 7}}
    )
    session_builder.add(
        "user",
        timestamp=timestamp,
        message={
            "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png_bytes).decode()}}]
        },
    )
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "precise.rrd"))
    for entity in ["/conversation/assistant", "/usage/input_tokens", "/media/images"]:
        assert entities[entity]["wall"].cast(pa.int64()).to_pylist() == [expected_ns]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure_point", ["send_columns", "flush"])
def test_failed_write_does_not_publish_or_leave_temporary_files(
    session_builder: SessionBuilder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool, failure_point: str
) -> None:
    """Publication preserves an existing destination until all writes succeed."""
    import rerun as rr

    session_builder.add("user", message={"content": "hello"})
    out: Path = tmp_path / "atomic.rrd"
    if existing:
        out.write_bytes(b"existing recording")

    def fail(*_args: object, **_kwargs: object) -> None:
        """Simulate an emission failure after the sink is attached."""
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(rr.RecordingStream, failure_point, fail)
    with pytest.raises(RuntimeError, match="injected write failure"):
        write_session_rrd(parse_session(session_builder.path), out)
    if existing:
        assert out.read_bytes() == b"existing recording"
    else:
        assert not out.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_raw_tool_metadata_round_trips(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Unknown nested metadata and non-object JSON survive on result rows."""
    payloads: list[object] = [{"unknown": {"nested": [1, True, None]}}, "output", ["a", {"b": 2}], None]
    for payload in payloads:
        session_builder.add("user", toolUseResult=payload, message={"content": [{"type": "tool_result", "content": "ok"}]})
    session_builder.add("user", message={"content": [{"type": "tool_result", "content": "ok"}]})
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "raw.rrd"))
    assert entities["/tools/unknown"]["tool_use_result_json"].to_pylist() == [
        ['{"unknown":{"nested":[1,true,null]}}'],
        ['"output"'],
        ['["a",{"b":2}]'],
        [""],
        [""],
    ]
def test_turns_follow_prompts_in_file_order(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Turn totals exclude preamble and children, and deduplicate split usage."""
    session_builder.add("assistant", message={"id": "before", "content": []})
    session_builder.add("user", promptId="p1", message={"content": "first"})
    session_builder.add(
        "assistant",
        message={
            "id": "m1",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "output_tokens_details": {"thinking_tokens": 1},
            },
            "content": [{"type": "tool_use", "id": "t", "name": "Bash"}],
        },
    )
    session_builder.add("assistant", message={"id": "m1", "usage": {"input_tokens": 99}, "content": []})
    session_builder.add(
        "user", message={"content": [{"type": "text", "text": "result label"}, {"type": "tool_result", "tool_use_id": "t", "content": "done"}]}
    )
    session_builder.add("user", isCompactSummary=True, message={"content": "summary"})
    session_builder.add("user", message={"content": [{"type": "text", "text": "second"}, {"type": "text", "text": "line"}]})
    session_builder.add("assistant", message={"id": "m2", "usage": {"output_tokens": 8}, "content": []})
    session_builder.add("user", path=session_builder.path.with_suffix("") / "subagents/agent-child.jsonl", message={"content": "child"})
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "turns.rrd"))
    turns: pa.Table = entities["/turns"]
    assert turns["TextLog:text"].to_pylist() == [["first"], ["second\nline"]]
    for name, expected in {
        "turn_index": [0, 1],
        "file_index": [1, 6],
        "prompt_id": ["p1", ""],
        "elapsed_ms": [4000.0, 1000.0],
        "n_tool_calls": [1, 0],
        "n_assistant_messages": [1, 1],
        "n_images": [0, 0],
        "input_tokens": [10, 0],
        "output_tokens": [4, 8],
        "cache_read_tokens": [3, 0],
        "cache_creation_tokens": [2, 0],
        "thinking_tokens": [1, 0],
    }.items():
        assert turns[name].to_pylist() == [[value] for value in expected]
    assert entities["/__properties/session"]["n_turns"].to_pylist() == [[2]]
    for name, expected in {"elapsed_ms": [4000.0, 1000.0], "output_tokens": [4.0, 8.0], "tool_calls": [1.0, 0.0]}.items():
        assert entities[f"/turns/{name}"]["Scalars:scalars"].to_pylist() == [[value] for value in expected]
    assert not any(path.startswith("/agents/child/turns") for path in entities)


def test_turn_images_and_nonprompt_users(session_builder: SessionBuilder, png_bytes: bytes, tmp_path: Path) -> None:
    """Images count within turns; result-only and untimed users start no turn."""
    import base64

    image: dict[str, object] = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png_bytes).decode()},
    }
    session_builder.add("user", timestamp=None, message={"content": "untimed"})
    session_builder.add("user", message={"content": [{"type": "text", "text": "prompt"}, image]})
    session_builder.add("user", message={"content": [{"type": "tool_result", "content": [image]}]})
    session_builder.add("user", message={"content": [image]})
    session_builder.add("system", timestamp="2026-09-18T20:00:02.500Z", content="last in file")
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_session(session_builder.path), tmp_path / "images.rrd"))
    assert entities["/turns"]["n_images"].to_pylist() == [[3]]
    assert entities["/turns"]["elapsed_ms"].to_pylist() == [[1500.0]]
    assert entities["/turns"]["TextLog:level"].to_pylist() == [["INFO"]]
    assert entities["/turns"]["TextLog:color"].to_pylist() == [[0x8AB4F8FF]]
    assert entities["/__properties/session"]["n_turns"].to_pylist() == [[1]]
