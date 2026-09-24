"""Codex parser and recording contracts use synthetic rollouts."""

from pathlib import Path

import pyarrow as pa
import pytest

from tests.conftest import RolloutBuilder
from tests.test_rerun_log import read_entities


def test_codex_explicit_turn_and_reasoning(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Task boundaries and encrypted reasoning survive the shared writer."""
    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta(cwd="/synthetic", git={"branch": "test"}, originator="exec")
    rollout_builder.add("event_msg", type="thread_settings_applied", thread_settings={"reasoning_effort": "high"})
    rollout_builder.add("turn_context", turn_id="turn", model="test-model")
    rollout_builder.add("event_msg", type="task_started", turn_id="turn", started_at=1789761600)
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "hello"}])
    rollout_builder.add("response_item", type="reasoning", encrypted_content="abcdef")
    rollout_builder.item("Reasoning", id="reasoning")
    rollout_builder.item("AgentMessage", id="answer", content=[{"type": "text", "text": "done"}])
    rollout_builder.add("event_msg", type="task_complete", turn_id="turn", duration_ms=1250, completed_at=1789761601)
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "codex.rrd"))
    assert entities["/conversation/thinking"]["TextLog:text"].to_pylist() == [["<encrypted reasoning, 6 bytes>"]]
    assert entities["/conversation/thinking"]["TextLog:level"].to_pylist() == [["DEBUG"]]
    assert entities["/turns"]["elapsed_ms"].to_pylist() == [[1250.0]]
    assert entities["/turns"]["model"].to_pylist() == [["test-model"]]
    assert entities["/turns"]["effort"].to_pylist() == [["high"]]
    assert entities["/__properties/session"]["agent"].to_pylist() == [["codex"]]


@pytest.mark.parametrize("version", ["0.149.9", "0.32.0"])
def test_version_floor(rollout_builder: RolloutBuilder, version: str) -> None:
    """Old CLIs are skipped before their payload schemas are interpreted."""
    from agent_traces.codex import SkipRollout, parse_rollout

    rollout_builder.meta(version=version)
    rollout_builder.add("event_msg", type="item_completed", item="old-schema")
    with pytest.raises(SkipRollout, match=f"codex-cli-{version}"):
        parse_rollout(rollout_builder.path)


def test_history_without_completed_items_is_skipped(rollout_builder: RolloutBuilder) -> None:
    """Response history alone cannot serve as the completion source of truth."""
    from agent_traces.codex import SkipRollout, parse_rollout

    rollout_builder.meta()
    rollout_builder.add("response_item", type="message", content=[{"type": "text", "text": "history"}])
    with pytest.raises(SkipRollout, match="no-item_completed"):
        parse_rollout(rollout_builder.path)


def test_usage_deduplicates_responses_and_prefers_records(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Only per-response usage contributes; final totals become properties."""
    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn")
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "go"}])
    rollout_builder.add(
        "event_msg", type="token_count", info={"last_token_usage": {"input_tokens": 999}, "total_token_usage": {"input_tokens": 9999}}
    )
    rollout_builder.item("AgentMessage", id="one", content=[{"type": "text", "text": "reply"}])
    for response, input_tokens in [("one", 10), ("one", 10), ("two", 20)]:
        rollout_builder.add(
            "token_usage_record",
            turn_id="turn",
            response_id=response,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": 3,
                "cached_input_tokens": 2,
                "cache_write_input_tokens": 1,
                "reasoning_output_tokens": 1,
            },
            thread_token_usage={"input_tokens": 30, "output_tokens": 6},
        )
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "usage.rrd"))
    assert entities["/usage/input_tokens"]["Scalars:scalars"].to_pylist() == [[10.0], [20.0]]
    assert entities["/turns"]["input_tokens"].to_pylist() == [[30]]
    assert entities["/__properties/session"]["total_input_tokens"].to_pylist() == [[30]]
    assert entities["/__properties/session"]["total_cost_usd"].to_pylist() == [[None]]


def test_legacy_usage_deduplicates_within_each_turn(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Identical last-response counters count again in a different turn."""
    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    for turn in ["one", "two"]:
        rollout_builder.add("event_msg", type="task_started", turn_id=turn)
        rollout_builder.item("UserMessage", turn_id=turn, content=[{"type": "text", "text": turn}])
        for _ in range(2):
            rollout_builder.add(
                "event_msg",
                type="token_count",
                info={"last_token_usage": {"input_tokens": 7, "output_tokens": 4}, "total_token_usage": {"input_tokens": 999}},
            )
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "fallback.rrd"))
    assert entities["/usage/input_tokens"]["Scalars:scalars"].to_pylist() == [[7.0], [7.0]]
    assert entities["/turns"]["input_tokens"].to_pylist() == [[7], [7]]


@pytest.mark.parametrize(
    "native,name,kind,fields",
    [
        ("CommandExecution", "exec", "shell", {"command": ["bash", "-lc", "echo synthetic"], "aggregated_output": "synthetic", "exit_code": 0}),
        ("FileChange", "apply_patch", "file_edit", {"changes": {"test.txt": {"type": "add", "unified_diff": "+test"}}}),
        ("McpToolCall", "mcp/server/tool", "mcp", {"server": "server", "tool": "tool", "arguments": {"a": 1}}),
        ("WebSearch", "web_search", "web_search", {}),
        ("Plan", "update_plan", "plan", {}),
        ("SubAgentActivity", "spawn_agent", "subagent", {"kind": "spawn_agent", "agent_thread_id": "child"}),
        ("ImageView", "view_image", "image", {"path": "/synthetic/image.png"}),
        ("Extension", "custom", "other", {"kind": "custom"}),
    ],
)
def test_tool_items_share_native_entities(
    rollout_builder: RolloutBuilder, tmp_path: Path, native: str, name: str, kind: str, fields: dict[str, object]
) -> None:
    """Every completed native tool gets call, result, kind, and elapsed rows."""
    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    rollout_builder.item(native, turn_id="turn", id="item", **fields)
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "tool.rrd"))
    assert entities[f"/tools/{name}"]["kind"].to_pylist() == [[kind], [kind]]
    assert entities[f"/tools/{name}"]["phase"].to_pylist() == [["call"], ["result"]]
    elapsed: float = entities[f"/tools/elapsed_ms/{name}"]["Scalars:scalars"].to_pylist()[0][0]
    assert elapsed != elapsed  # NaN: no raw call/output pair, and the item's own stamps are logging times, not execution


def test_raw_tool_response_is_joined_by_arguments(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Item IDs can differ from raw call IDs; raw JSON remains verbatim."""
    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    raw: str = '{ "cmd": "echo synthetic", "extra": [true, null] }'
    rollout_builder.add("response_item", type="function_call", call_id="raw-call", name="exec_command", arguments=raw)
    rollout_builder.item("CommandExecution", id="different", command=["bash", "-lc", "echo synthetic"], aggregated_output="preview")
    rollout_builder.add("response_item", type="function_call_output", call_id="raw-call", output="full result")
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "raw.rrd"))
    assert entities["/tools/exec"]["input_json"].to_pylist()[0] == [raw]
    assert entities["/tools/exec"]["tool_use_result_json"].to_pylist()[1] == ['"full result"']


def test_images_from_response_and_local_files(rollout_builder: RolloutBuilder, tmp_path: Path, png_bytes: bytes) -> None:
    """Both image sources are decoded without replaying raw message text."""
    import base64

    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    local: Path = tmp_path / "image.png"
    local.write_bytes(png_bytes)
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn")
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "images"}])
    rollout_builder.add(
        "response_item",
        type="message",
        content=[
            {"type": "text", "text": "not a second prompt"},
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(png_bytes).decode()},
        ],
    )
    rollout_builder.add("event_msg", type="user_message", local_images=[str(local), str(tmp_path / "missing.png")])
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "images.rrd"))
    assert entities["/media/images"].num_rows == 2
    assert entities["/turns"]["n_images"].to_pylist() == [[2]]
    assert entities["/conversation/user"].num_rows == 1


def test_subagents_fold_recursively_and_orphans_keep_parent(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Home-wide parent identifiers, including archived children, define the tree."""
    from agent_traces.codex import parse_rollout, session_sources
    from agent_traces.events import Session

    rollout_builder.meta()
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "root"}])
    home: Path = tmp_path / ".codex-alt"
    child: RolloutBuilder = RolloutBuilder(home / "archived_sessions/rollout-child.jsonl")
    child.meta("child", parent_thread_id="thread", source={"subagent": {"thread_spawn": {}}})
    child.item("AgentMessage", content=[{"type": "text", "text": "child"}])
    grandchild: RolloutBuilder = RolloutBuilder(home / "sessions/rollout-grandchild.jsonl")
    grandchild.meta("grandchild", parent_thread_id="child", source={"subagent": {}})
    grandchild.item("Reasoning")
    orphan: RolloutBuilder = RolloutBuilder(home / "sessions/rollout-orphan.jsonl")
    orphan.meta("orphan", parent_thread_id="missing", source={"subagent": {}})
    orphan.item("Reasoning")
    parsed: Session = parse_rollout(rollout_builder.path)
    assert set(parsed.subagents) == {"child", "grandchild"}
    assert set(session_sources(rollout_builder.path)) == {rollout_builder.path, child.path, grandchild.path}
    assert parse_rollout(orphan.path).parent_thread == "missing"


def test_convert_detects_codex(rollout_builder: RolloutBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The single-session command uses the envelope provider."""
    from agent_traces.apis.convert import Config, main

    rollout_builder.meta()
    rollout_builder.item("Reasoning")
    out: Path = tmp_path / "single.rrd"
    main(Config(session=rollout_builder.path, out=out))
    assert "entities=1 rows=1" in capsys.readouterr().out
    assert read_entities(out)["/__properties/session"]["agent"].to_pylist() == [["codex"]]


def test_batch_detects_codex_and_fingerprints_children(
    rollout_builder: RolloutBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str], png_bytes: bytes
) -> None:
    """Children fold once; child and local image changes invalidate the parent."""
    from agent_traces.apis.convert_all import Config, main
    from agent_traces.manifest import load_manifest

    home: Path = tmp_path / ".codex-alt"
    rollout_builder.meta()
    rollout_builder.item("Reasoning")
    child: RolloutBuilder = RolloutBuilder(home / "archived_sessions/rollout-child.jsonl")
    child.meta("child", parent_thread_id="thread", source={"subagent": {}})
    child.item("Reasoning")
    old: RolloutBuilder = RolloutBuilder(home / "sessions/rollout-old.jsonl")
    old.meta("old", version="0.149.9")
    old.item("Reasoning")
    image: Path = tmp_path / "local.png"
    image.write_bytes(png_bytes)
    child.add("event_msg", type="user_message", local_images=[str(image)])
    config: Config = Config(home=home, out=tmp_path / "out")
    main(config)
    output: str = capsys.readouterr().out
    assert "converted=1 skipped=2 failed=0" in output
    assert "codex-cli-0.149.9" in output
    assert set(load_manifest(tmp_path / "out/codex-alt/manifest.json").sessions) == {"thread"}
    main(config)
    assert "converted=0 skipped=3 failed=0" in capsys.readouterr().out
    child.item("Reasoning")
    main(config)
    assert "converted=1 skipped=2 failed=0" in capsys.readouterr().out
    image.write_bytes(png_bytes + b"changed")
    main(config)
    assert "converted=1 skipped=2 failed=0" in capsys.readouterr().out


def test_model_changes_after_task_start_and_unknown_events(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Context can arrive after task start; unknown events remain accounted for."""
    from agent_traces.codex import parse_rollout
    from agent_traces.events import Session
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    for turn, model in [("one", "first"), ("two", "second")]:
        rollout_builder.add("event_msg", type="task_started", turn_id=turn)
        rollout_builder.add("turn_context", turn_id=turn, model=model)
        rollout_builder.item("UserMessage", turn_id=turn, content=[{"type": "text", "text": "go"}])
        rollout_builder.item("AgentMessage", turn_id=turn, id=turn, content=[{"type": "text", "text": "done"}])
        rollout_builder.add("event_msg", type="task_complete", turn_id=turn, duration_ms=250)
    rollout_builder.add("event_msg", type="future_event")
    rollout_builder.item("FutureItem")
    session: Session = parse_rollout(rollout_builder.path)
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(session, tmp_path / "models.rrd"))
    assert entities["/turns"]["model"].to_pylist() == [["first"], ["second"]]
    assert entities["/turns"]["n_assistant_messages"].to_pylist() == [[1], [1]]
    assert session.skipped["future_event"] == 1
    assert session.skipped["FutureItem"] == 1


def test_batch_bad_rollout_keeps_progress_and_private_errors(
    rollout_builder: RolloutBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed first line cannot abort conversion of valid neighbors."""
    from agent_traces.apis.convert_all import Config, main

    rollout_builder.meta()
    rollout_builder.item("Reasoning")
    bad: Path = rollout_builder.path.with_name("rollout-bad.jsonl")
    bad.write_text('{"private": "must not appear", broken\n')
    main(Config(home=tmp_path / ".codex-alt", out=tmp_path / "out"))
    output: str = capsys.readouterr().out
    assert "converted=1 skipped=0 failed=1" in output
    assert "FAILED" in output
    assert "must not appear" not in output


def test_reasoning_matches_turn_and_order_with_missing_payload(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Reasoning completions remain visible even when raw ciphertext is missing."""
    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    rollout_builder.add("turn_context", turn_id="one", model="model")
    rollout_builder.item("Reasoning", turn_id="one")
    rollout_builder.item("Reasoning", turn_id="one")
    rollout_builder.add("response_item", type="reasoning", encrypted_content="abc", internal_chat_message_metadata_passthrough={"turn_id": "one"})
    rollout_builder.add("turn_context", turn_id="two", model="model")
    rollout_builder.add("response_item", type="reasoning", encrypted_content="abcdef", internal_chat_message_metadata_passthrough={"turn_id": "two"})
    rollout_builder.item("Reasoning", turn_id="two")
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "reasoning.rrd"))
    assert entities["/conversation/thinking"]["TextLog:text"].to_pylist() == [
        ["<encrypted reasoning, 3 bytes>"],
        ["<encrypted reasoning, 0 bytes>"],
        ["<encrypted reasoning, 6 bytes>"],
    ]


def test_legacy_metadata_null_provider_and_string_subagent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Historical metadata variants still reach the version-floor policy."""
    from agent_traces.apis.convert_all import Config, main

    builder: RolloutBuilder = RolloutBuilder(tmp_path / ".codex/sessions/rollout-old.jsonl")
    builder.add("session_meta", id="old", cli_version="0.149.9", model_provider=None, source={"subagent": "review"})
    main(Config(home=tmp_path / ".codex", out=tmp_path / "out"))
    output: str = capsys.readouterr().out
    assert "converted=0 skipped=1 failed=0" in output
    assert "codex-cli-0.149.9" in output


def test_legacy_unenveloped_history_is_skipped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Early history files have bare metadata and no completion events."""
    from agent_traces.apis.convert_all import Config, main

    path: Path = tmp_path / ".codex/sessions/rollout-old.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"id":"old","timestamp":"2025-01-01T00:00:00Z","instructions":null}\n')
    main(Config(home=tmp_path / ".codex", out=tmp_path / "out"))
    output: str = capsys.readouterr().out
    assert "converted=0 skipped=1 failed=0" in output
    assert "no-item_completed" in output


def test_null_response_content_does_not_drop_completed_items(rollout_builder: RolloutBuilder) -> None:
    """A tool response can have an explicit null message content field."""
    from agent_traces.codex import parse_rollout
    from agent_traces.events import Session

    rollout_builder.meta(version="0.150.0")
    rollout_builder.add("response_item", type="function_call_output", call_id="c", output="ok", content=None)
    rollout_builder.item("Reasoning")
    session: Session = parse_rollout(rollout_builder.path)
    assert len(session.main) == 1


def test_mcp_raw_arguments_match_without_json_whitespace(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Correlation compares argument values while preserving their source bytes."""
    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    arguments: str = '{ "query": "synthetic", "limit": 1 }'
    rollout_builder.add("response_item", type="function_call", name="mcp__search__find", call_id="raw", arguments=arguments)
    rollout_builder.item(
        "McpToolCall",
        id="native",
        server="search",
        tool="find",
        arguments={"limit": 1, "query": "synthetic"},
        status="failed",
        error={"message": "synthetic error"},
    )
    rollout_builder.add("response_item", type="function_call_output", call_id="raw", output="synthetic response")
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "mcp.rrd"))
    assert entities["/tools/mcp/search/find"]["input_json"].to_pylist()[0] == [arguments]
    assert entities["/tools/mcp/search/find"]["tool_use_result_json"].to_pylist()[1] == ['"synthetic response"']
    assert entities["/tools/mcp/search/find"]["TextLog:level"].to_pylist()[1] == ["ERROR"]


def test_task_completion_bounds_elapsed_without_duration(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Usage written after completion cannot lengthen an explicit task."""
    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn", started_at=1789761600)
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "go"}])
    rollout_builder.add("event_msg", type="task_complete", turn_id="turn", completed_at=1789761602)
    rollout_builder.add("token_usage_record", turn_id="turn", response_id="late", usage={"output_tokens": 2})
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "bounded.rrd"))
    # `started_at`/`completed_at` are Unix seconds; the turn is bounded by the task events' envelope timestamps (2 builder steps).
    assert entities["/turns"]["elapsed_ms"].to_pylist() == [[2000.0]]
    assert entities["/turns"]["output_tokens"].to_pylist() == [[2]]


def test_native_tool_details_survive_without_raw_call(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Tool-specific query and result fields remain available even without a join."""
    import orjson

    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    rollout_builder.item("WebSearch", id="search", query="synthetic query", action={"type": "search", "query": "synthetic query"}, results=[{"title": "synthetic result"}])
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "search.rrd"))
    raw: str = entities["/tools/web_search"]["input_json"].to_pylist()[0][0]
    assert orjson.loads(raw)["query"] == "synthetic query"
    result: str = entities["/tools/web_search"]["tool_use_result_json"].to_pylist()[1][0]
    assert orjson.loads(result)["results"] == [{"title": "synthetic result"}]


def test_turn_rows_sit_at_the_task_event_time(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """task_started carries Unix seconds; the turn row must land on the wall timeline at that instant, not near the epoch."""
    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn", started_at=1789761600)
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "go"}])
    rollout_builder.add("event_msg", type="task_complete", turn_id="turn", duration_ms=5, completed_at=1789761600)
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "turn-time.rrd"))
    wall_ns: int = entities["/turns"]["wall"].cast(pa.int64()).to_pylist()[0]
    assert wall_ns > 1_700_000_000 * 1_000_000_000  # 2023 or later, i.e. not 1970


def test_tool_elapsed_comes_from_the_raw_call_output_span(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """The item's own stamps are logging times ~1 ms apart; the raw call → output envelope span is the real wall time."""
    from agent_traces.codex import parse_rollout
    from agent_traces.rerun_log import write_session_rrd

    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn")
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "go"}])
    rollout_builder.add("response_item", type="custom_tool_call", name="exec", call_id="c1", input="{\"cmd\":\"ls\"}")  # builder step 3
    rollout_builder.add("event_msg", type="agent_message", message="working")  # step 4
    rollout_builder.add("response_item", type="custom_tool_call_output", call_id="c1", output=[{"type": "text", "text": "ok"}])  # step 5
    rollout_builder.item("CommandExecution", id="c1", command=["bash", "-lc", "ls"], status="completed", exit_code=0, aggregated_output="ok",
                         )
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "elapsed.rrd"))
    assert entities["/tools/elapsed_ms/exec"]["Scalars:scalars"].to_pylist() == [[2000.0]]


def test_message_identity_and_late_usage_across_turns(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Repeated message IDs count per turn while response usage counts per transcript."""
    from agent_traces.codex import parse_rollout
    from agent_traces.events import AssistantText, UsageSample
    from agent_traces.rerun_log import write_session_rrd
    from agent_traces.turns import aggregate_turns

    rollout_builder.meta()
    for turn in ["one", "two"]:
        rollout_builder.add("event_msg", type="task_started", turn_id=turn)
        rollout_builder.item("UserMessage", turn_id=turn, id="human", content=[{"type": "text", "text": turn}])
        for _ in range(2):
            rollout_builder.item("AgentMessage", turn_id=turn, id="message", content=[{"type": "text", "text": "reply"}])
        rollout_builder.item("AgentMessage", turn_id=turn, content=[{"type": "text", "text": "no identity"}])
        rollout_builder.add("event_msg", type="task_complete", turn_id=turn, duration_ms=125)
        rollout_builder.add("token_usage_record", turn_id=turn, response_id="shared", usage={"input_tokens": 7, "output_tokens": 3})
    session = parse_rollout(rollout_builder.path)
    assert [event.message_id for event in session.main if isinstance(event.payload, AssistantText)] == ["message", "message", "", "message", "message", ""]
    samples: list[UsageSample] = [event.payload for event in session.main if isinstance(event.payload, UsageSample)]
    assert len(samples) == 1
    assert samples[0].usage.cache_creation_5m_tokens is None
    assert samples[0].usage.cache_creation_1h_tokens is None
    turns = aggregate_turns(session.main)
    assert [turn.n_assistant_messages for turn in turns] == [1, 1]
    assert [turn.elapsed_ms for turn in turns] == [125.0, 125.0]
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(session, tmp_path / "identities.rrd"))
    assert entities["/turns"]["input_tokens"].to_pylist() == [[7], [0]]
    assert entities["/usage/input_tokens"]["Scalars:scalars"].to_pylist() == [[7.0]]
    assert "/usage/cache_creation_5m_tokens" not in entities
    assert "/usage/cache_creation_1h_tokens" not in entities
