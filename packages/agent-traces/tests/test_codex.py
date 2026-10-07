"""Codex parser and recording contracts use synthetic rollouts."""

import re
from pathlib import Path

import pyarrow as pa
import pytest

from agent_traces import codex
from agent_traces import codex_records as cr
from agent_traces.apis.convert import Config as ConvertConfig
from agent_traces.apis.convert import main as convert_main
from agent_traces.apis.convert_all import Config as ConvertAllConfig
from agent_traces.apis.convert_all import main as convert_all_main
from agent_traces.codex import session_source as codex_session_source
from agent_traces.errors import SkipSession
from agent_traces.events import AssistantText, Execution, Prompt, Session, Thinking, ToolCall, ToolResult, UsageSample
from agent_traces.manifest import load_manifest
from agent_traces.rerun_log import PROPERTY_TYPES, write_session_rrd
from agent_traces.turns import aggregate_turns
from tests.conftest import RolloutBuilder, agent_rows, metadata_values, parse_rollout, read_entities


def test_codex_explicit_turn_and_reasoning(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Task boundaries and encrypted reasoning survive the shared writer."""
    rollout_builder.meta(cwd="/synthetic", git={"branch": "test"}, originator="exec")
    rollout_builder.add("event_msg", type="thread_settings_applied", thread_settings={"reasoning_effort": "high"})
    rollout_builder.add("turn_context", turn_id="turn", model="test-model")
    rollout_builder.add("event_msg", type="task_started", turn_id="turn", started_at=1789761600)
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "hello"}])
    rollout_builder.add("response_item", type="reasoning", encrypted_content="abcdef")
    rollout_builder.item("Reasoning", id="reasoning")
    rollout_builder.item("AgentMessage", id="answer", content=[{"type": "text", "text": "done"}])
    rollout_builder.add("event_msg", type="task_complete", turn_id="turn", duration_ms=1250, completed_at=1789761601)
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "codex.rrd").path)
    assert entities["/conversation/thinking"]["TextLog:text"].to_pylist() == [["<encrypted reasoning, 6 bytes>"]]
    assert entities["/conversation/thinking"]["TextLog:level"].to_pylist() == [["DEBUG"]]
    assert entities["/turns"]["elapsed_ms"].to_pylist() == [[3000.0]]
    assert entities["/turns"]["model"].to_pylist() == [["test-model"]]
    assert entities["/turns"]["effort"].to_pylist() == [["high"]]
    assert entities["/__properties/session"]["agent"].to_pylist() == [["codex"]]


@pytest.mark.parametrize("version", ["0.149.9", "0.32.0"])
def test_version_floor(rollout_builder: RolloutBuilder, version: str) -> None:
    """Old CLIs are skipped before their payload schemas are interpreted."""
    rollout_builder.meta(version=version)
    rollout_builder.add("event_msg", type="item_completed", item="old-schema")
    with pytest.raises(SkipSession, match=f"codex-cli-{version}"):
        parse_rollout(rollout_builder.path)


def test_usage_deduplicates_responses_and_prefers_records(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Only per-response usage contributes; final totals become properties."""
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
            thread_token_usage={"input_tokens": 30, "cached_input_tokens": 4, "output_tokens": 6},
        )
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "usage.rrd").path)
    assert entities["/usage/input_tokens"]["Scalars:scalars"].to_pylist() == [[8.0], [18.0]]
    assert entities["/turns"]["input_tokens"].to_pylist() == [[26]]
    assert entities["/__properties/session"]["total_input_tokens"].to_pylist() == [[26]]
    assert entities["/__properties/session"]["total_cache_read_tokens"].to_pylist() == [[4]]
    assert entities["/__properties/session"]["total_cost_usd"].to_pylist() == [[None]]


def test_legacy_usage_deduplicates_within_each_turn(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Identical last-response counters count again in a different turn."""
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
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "fallback.rrd").path)
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
    """Native execution items keep their own identities without fabricating model calls."""
    rollout_builder.meta()
    rollout_builder.item(native, turn_id="turn", id="item", **fields)
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "tool.rrd").path)
    expected: dict[str, str] = {"CommandExecution": "command", "FileChange": "file_change", "McpToolCall": "mcp", "WebSearch": "web_search", "Plan": "update_plan", "SubAgentActivity": "subagent", "ImageView": "image_view", "Extension": "extension"}
    assert metadata_values(entities[f"/executions/{expected[native]}"], 'item_id') == [["item"]]
    assert not any(entity.startswith("/tools/") for entity in entities)


def test_images_from_response_and_local_files(rollout_builder: RolloutBuilder, tmp_path: Path, png_bytes: bytes) -> None:
    """Both image sources are decoded without replaying raw message text."""
    import base64


    local: Path = tmp_path / "image.png"
    local.write_bytes(png_bytes)
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn")
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "images"}])
    rollout_builder.add("turn_context", turn_id="turn", model="image-model")
    rollout_builder.add("event_msg", type="thread_settings_applied", thread_settings={"reasoning_effort": "high"})
    rollout_builder.add(
        "response_item",
        type="message",
        content=[
            {"type": "text", "text": "not a second prompt"},
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(png_bytes).decode()},
        ],
    )
    rollout_builder.add("event_msg", type="user_message", local_images=[str(local), str(tmp_path / "missing.png")])
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "images.rrd").path)
    assert entities["/media/images"].num_rows == 2
    assert entities["/media/images"]["model"].to_pylist() == [["image-model"], ["image-model"]]
    assert entities["/media/images"]["effort"].to_pylist() == [["high"], ["high"]]
    assert entities["/turns"]["n_images"].to_pylist() == [[2]]
    assert entities["/conversation/user"].num_rows == 1


def test_subagents_fold_recursively_and_orphans_keep_parent(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Home-wide parent identifiers, including archived children, define the tree."""
    rollout_builder.meta()
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "root"}])
    home: Path = tmp_path / ".codex"
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
    assert set(codex_session_source(rollout_builder.path).inputs) == {rollout_builder.path, child.path, grandchild.path}
    assert parse_rollout(orphan.path).properties["parent_thread"] == "missing"


def test_convert_detects_codex(rollout_builder: RolloutBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The single-session command uses the envelope provider."""
    rollout_builder.meta()
    rollout_builder.item("Reasoning")
    out: Path = tmp_path / "single.rrd"
    convert_main(ConvertConfig(session=rollout_builder.path, out=out))
    assert "entities=1 rows=1" in capsys.readouterr().out
    assert read_entities(out)["/__properties/session"]["agent"].to_pylist() == [["codex"]]


def test_batch_detects_codex_and_fingerprints_children(
    rollout_builder: RolloutBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str], png_bytes: bytes
) -> None:
    """Children fold once; child and local image changes invalidate the parent."""
    home: Path = tmp_path / ".codex"
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
    config: ConvertAllConfig = ConvertAllConfig(home=home, out=tmp_path / "out")
    convert_all_main(config)
    output: str = capsys.readouterr().out
    assert "converted=1 skipped=2 failed=0" in output
    assert "codex-cli-0.149.9" in output
    assert set(load_manifest(tmp_path / "out/codex/manifest.json").sessions) == {"thread"}
    convert_all_main(config)
    assert "converted=0 skipped=3 failed=0" in capsys.readouterr().out
    child.item("Reasoning")
    convert_all_main(config)
    assert "converted=1 skipped=2 failed=0" in capsys.readouterr().out
    image.write_bytes(png_bytes + b"changed")
    convert_all_main(config)
    assert "converted=1 skipped=2 failed=0" in capsys.readouterr().out


def test_model_changes_after_task_start_and_unknown_events(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Context can arrive after task start; unknown events remain accounted for."""
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
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(session, tmp_path / "models.rrd").path)
    assert entities["/turns"]["model"].to_pylist() == [["first"], ["second"]]
    assert entities["/turns"]["n_assistant_messages"].to_pylist() == [[1], [1]]
    assert session.skipped["event_msg/future_event"] == 1
    assert session.skipped["event_msg/item_completed/FutureItem"] == 1


def test_batch_bad_rollout_keeps_progress_and_private_errors(
    rollout_builder: RolloutBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed first line cannot abort conversion of valid neighbors."""
    rollout_builder.meta()
    rollout_builder.item("Reasoning")
    bad: Path = rollout_builder.path.with_name("rollout-bad.jsonl")
    bad.write_text('{"private": "must not appear", broken\n')
    convert_all_main(ConvertAllConfig(home=tmp_path / ".codex", out=tmp_path / "out"))
    output: str = capsys.readouterr().out
    assert "converted=1 skipped=0 failed=1" in output
    assert f"FAILED {bad}:" in output
    assert "must not appear" not in output


def test_reasoning_matches_turn_and_order_with_missing_payload(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Reasoning completions remain visible even when raw ciphertext is missing."""
    rollout_builder.meta()
    rollout_builder.add("turn_context", turn_id="one", model="model")
    rollout_builder.item("Reasoning", turn_id="one")
    rollout_builder.item("Reasoning", turn_id="one")
    rollout_builder.add("response_item", type="reasoning", encrypted_content="abc", internal_chat_message_metadata_passthrough={"turn_id": "one"})
    rollout_builder.add("turn_context", turn_id="two", model="model")
    rollout_builder.add("response_item", type="reasoning", encrypted_content="abcdef", internal_chat_message_metadata_passthrough={"turn_id": "two"})
    rollout_builder.item("Reasoning", turn_id="two")
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "reasoning.rrd").path)
    assert entities["/conversation/thinking"]["TextLog:text"].to_pylist() == [
        ["<encrypted reasoning, 3 bytes>"],
        ["<encrypted reasoning, size unknown>"],
        ["<encrypted reasoning, 6 bytes>"],
    ]


def test_legacy_metadata_null_provider_and_string_subagent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Historical metadata variants still reach the version-floor policy."""
    builder: RolloutBuilder = RolloutBuilder(tmp_path / ".codex/sessions/rollout-old.jsonl")
    builder.add("session_meta", id="old", cli_version="0.149.9", model_provider=None, source={"subagent": "review"})
    convert_all_main(ConvertAllConfig(home=tmp_path / ".codex", out=tmp_path / "out"))
    output: str = capsys.readouterr().out
    assert "converted=0 skipped=1 failed=0" in output
    assert "codex-cli-0.149.9" in output


def test_legacy_unenveloped_history_is_skipped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Early history files have bare metadata and no completion events."""
    path: Path = tmp_path / ".codex/sessions/rollout-old.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"id":"old","timestamp":"2025-01-01T00:00:00Z","instructions":null}\n')
    convert_all_main(ConvertAllConfig(home=tmp_path / ".codex", out=tmp_path / "out"))
    output: str = capsys.readouterr().out
    assert "converted=0 skipped=1 failed=0" in output
    assert "legacy-rollout" in output


def test_null_response_content_does_not_drop_completed_items(rollout_builder: RolloutBuilder) -> None:
    """A tool response can have an explicit null message content field."""
    rollout_builder.meta(version="0.150.0")
    rollout_builder.add("response_item", type="function_call_output", call_id="c", output="ok", content=None)
    rollout_builder.item("Reasoning")
    session: Session = parse_rollout(rollout_builder.path)
    assert len(session.main) == 2


def test_usage_activity_extends_turn_after_task_completion(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Elapsed time runs from the human prompt through the last usage activity."""
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn", started_at=1789761600)
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "go"}])
    rollout_builder.add("event_msg", type="task_complete", turn_id="turn", completed_at=1789761602)
    rollout_builder.add("token_usage_record", turn_id="turn", response_id="late", usage={"output_tokens": 2})
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "bounded.rrd").path)
    # The prompt and last usage row are two builder steps apart.
    assert entities["/turns"]["elapsed_ms"].to_pylist() == [[2000.0]]
    assert entities["/turns"]["output_tokens"].to_pylist() == [[2]]


def test_native_tool_details_survive_without_raw_call(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Tool-specific query and result fields remain available even without a join."""
    import orjson


    rollout_builder.meta()
    rollout_builder.item("WebSearch", id="search", query="synthetic query", action={"type": "search", "query": "synthetic query"}, results=[{"title": "synthetic result"}])
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "search.rrd").path)
    raw = entities["/executions/web_search"]["input_json"].to_pylist()[0][0]
    assert isinstance(raw, str)
    assert orjson.loads(raw)["query"] == "synthetic query"
    result = entities["/executions/web_search"]["native_json"].to_pylist()[0][0]
    assert isinstance(result, str)
    assert orjson.loads(result)["results"] == [{"title": "synthetic result"}]


def test_turn_rows_sit_at_the_task_event_time(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """task_started carries Unix seconds; the turn row must land on the wall timeline at that instant, not near the epoch."""
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn", started_at=1789761600)
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "go"}])
    rollout_builder.add("event_msg", type="task_complete", turn_id="turn", duration_ms=5, completed_at=1789761600)
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "turn-time.rrd").path)
    wall_ns: int = entities["/turns"]["wall"].cast(pa.int64()).to_pylist()[0]
    assert wall_ns > 1_700_000_000 * 1_000_000_000  # 2023 or later, i.e. not 1970


def test_tool_elapsed_does_not_use_the_raw_call_output_span(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """The envelope includes orchestration and cannot establish command elapsed."""
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn")
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "go"}])
    rollout_builder.add("response_item", type="custom_tool_call", name="exec", call_id="c1", input="{\"cmd\":\"ls\"}")  # builder step 3
    rollout_builder.add("event_msg", type="agent_message", message="working")  # step 4
    rollout_builder.add("response_item", type="custom_tool_call_output", call_id="c1", output=[{"type": "text", "text": "ok"}])  # step 5
    rollout_builder.item("CommandExecution", id="c1", command=["bash", "-lc", "ls"], status="completed", exit_code=0, aggregated_output="ok",
                         )
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "elapsed.rrd").path)
    assert "/elapsed/tools/exec" not in entities


def test_message_identity_and_late_usage_across_turns(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Repeated message IDs count per turn while response usage counts per transcript."""
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
    assert [event.message_id for event in session.main if isinstance(event.payload, AssistantText)] == ["message", "", "message", ""]
    samples: list[UsageSample] = [event.payload for event in session.main if isinstance(event.payload, UsageSample)]
    assert len(samples) == 1
    assert samples[0].usage.cache_creation_5m_tokens is None
    assert samples[0].usage.cache_creation_1h_tokens is None
    turns = aggregate_turns(session.main)
    assert [turn.n_assistant_messages for turn in turns] == [1, 1]
    assert [turn.elapsed_ms for turn in turns] == [5000.0, 3000.0]
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(session, tmp_path / "identities.rrd").path)
    assert entities["/turns"]["input_tokens"].to_pylist() == [[7], [0]]
    assert entities["/usage/input_tokens"]["Scalars:scalars"].to_pylist() == [[7.0]]
    assert "/usage/cache_creation_5m_tokens" not in entities
    assert "/usage/cache_creation_1h_tokens" not in entities


@pytest.mark.parametrize("parent_state", ["version", "failed"])
def test_excluded_parent_does_not_claim_folded_children(rollout_builder: RolloutBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str], parent_state: str) -> None:
    """A child has no recording when its known owner is excluded or fails."""
    rollout_builder.meta(version="0.149.0" if parent_state == "version" else "0.153.4")
    if parent_state == "failed":
        with rollout_builder.path.open("ab") as stream:  # valid JSON with an invalid envelope still fails the rollout
            stream.write(b'{"timestamp":"not-a-time","type":"event_msg","payload":{"type":"task_started","turn_id":"turn"}}\n')
    child = RolloutBuilder(tmp_path / ".codex/archived_sessions/child.jsonl")
    child.meta("child", parent_thread_id="thread")
    child.item("Reasoning")
    convert_all_main(ConvertAllConfig(home=tmp_path / ".codex", out=tmp_path / "out"))
    output = capsys.readouterr().out
    assert "folded-subagent" not in output
    assert ("parent-failed" if parent_state == "failed" else "parent-skipped") in output
    assert ("converted=0 skipped=1 failed=1" if parent_state == "failed" else "converted=0 skipped=2 failed=0") in output
    assert not list((tmp_path / "out").rglob("*.rrd"))
    if parent_state == "failed":
        assert f"FAILED {rollout_builder.path}:" in output


@pytest.mark.parametrize("joined", [False, True])
def test_command_input_and_full_output_once(rollout_builder: RolloutBuilder, tmp_path: Path, joined: bool) -> None:
    """A command output appears once across the saved tool text fields."""
    rollout_builder.meta()
    output: str = "unique-output-marker\n" * 1000
    if joined:
        rollout_builder.add("response_item", type="custom_tool_call", name="exec", call_id="raw", input='text(await tools.exec_command({cmd:"cat example"}));')
    rollout_builder.item("CommandExecution", id="native", command=["bash", "-lc", "cat example"], aggregated_output=output)
    if joined:
        rollout_builder.add("response_item", type="custom_tool_call_output", call_id="raw", output=output)
    table: pa.Table = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "once.rrd").path)["/executions/command"]
    input_json = table["input_json"].to_pylist()[0][0]
    assert isinstance(input_json, str)
    assert "cat example" in input_json
    assert "aggregated_output" not in input_json
    assert "unique-output-marker" not in input_json
    strings: str = "".join(value for name in table.column_names for cell in table[name].to_pylist() if isinstance(cell, list) for value in cell if isinstance(value, str))
    assert strings.count("unique-output-marker") == 1000
    assert output in strings


def test_consumed_records_are_not_skipped(rollout_builder: RolloutBuilder) -> None:
    """Consumed settings, user image carriers and duplicate usage are accounted for."""
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="thread_settings_applied", thread_settings={"reasoning_effort": "high"})
    rollout_builder.add("event_msg", type="user_message", local_images=["missing.png"])
    for _ in range(2):
        rollout_builder.add("event_msg", type="token_count", info={"last_token_usage": {"input_tokens": 7}})
    rollout_builder.add("token_usage_record", response_id="r", usage={"input_tokens": 7})
    rollout_builder.item("ContextCompaction")
    rollout_builder.item("UnknownFutureItem")
    session = parse_rollout(rollout_builder.path)
    assert session.skipped == {"event_msg/item_completed/UnknownFutureItem": 1}
    assert any(isinstance(r.payload, Prompt) and r.payload.role == "compaction" for r in session.main)


def test_new_tag_uses_one_definition(rollout_builder: RolloutBuilder, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a modeled tool tag requires no second name table."""
    monkeypatch.setitem(cr.ITEM_TYPES, "FutureTool", cr.ItemTag(cr.OtherTool, "future"))
    rollout_builder.meta()
    rollout_builder.item("FutureTool", id="new")
    session = parse_rollout(rollout_builder.path)
    assert [r.payload.kind for r in session.main if isinstance(r.payload, Execution)] == ["future"]


def test_replayed_metadata_preserves_every_child(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """First headers own identities; replayed parent headers cannot replace children."""
    rollout_builder.meta("owner", cwd="/owner")
    rollout_builder.meta("replayed", cwd="/replayed")
    rollout_builder.item("Reasoning")
    for name in ("one", "two", "three"):
        child = RolloutBuilder(tmp_path / ".codex/archived_sessions" / f"{name}.jsonl")
        child.meta(name, parent_thread_id="owner")
        child.meta("owner")
        child.item("AgentMessage", content=[{"type": "text", "text": name}])
    source = codex_session_source(rollout_builder.path)
    session = source.parse()
    assert source.session_id == session.session_id == "owner"
    assert session.properties["cwd"] == "/owner"
    assert set(session.subagents) == {"one", "two", "three"}
    for name, events in session.subagents.items():
        assert [r.payload.text for r in events if isinstance(r.payload, AssistantText)] == [name]
    entities = read_entities(write_session_rrd(session, tmp_path / "children.rrd").path)
    assert {row[0] for row in entities["/conversation/assistant"]["agent_id"].to_pylist()} == {"one", "two", "three"}


def test_old_version_rejected_before_body_decode(rollout_builder: RolloutBuilder, monkeypatch: pytest.MonkeyPatch) -> None:
    """Version rejection happens before any unsupported body reaches the decoder."""
    rollout_builder.meta(version="0.149.9")
    rollout_builder.item("Reasoning")
    original = codex.orjson.loads

    def no_bodies(data: bytes) -> object:
        assert b'"event_msg"' not in data, "old rollout body was decoded"
        return original(data)

    monkeypatch.setattr(codex.orjson, "loads", no_bodies)
    with pytest.raises(SkipSession, match="codex-cli-0.149.9"):
        codex.session_source(rollout_builder.path).parse()


def test_damaged_rollout_line_is_skipped_counted_and_warned(rollout_builder: RolloutBuilder) -> None:
    """A damaged line after the header costs that line only; the rest of the rollout converts."""
    rollout_builder.meta()
    rollout_builder.item("Reasoning")
    with rollout_builder.path.open("ab") as stream:
        stream.write(b'{"type":"event_msg","pay\n')
    rollout_builder.item("Reasoning")
    with pytest.warns(UserWarning, match=rf"{re.escape(str(rollout_builder.path))}:3: "):
        session = parse_rollout(rollout_builder.path)
    assert session.skipped["damaged-line"] == 1
    assert sum(isinstance(e.payload, Thinking) for e in session.main) == 2


def test_valid_json_that_is_not_an_object_still_fails_the_rollout(rollout_builder: RolloutBuilder) -> None:
    """Only a line that is not valid JSON counts as damaged; a JSON value of the wrong shape is a format change."""
    rollout_builder.meta()
    rollout_builder.item("Reasoning")
    with rollout_builder.path.open("ab") as stream:
        stream.write(b"[]\n")
    with pytest.raises(ValueError, match=r"rollout-thread.jsonl:3: expected a JSON object"):
        parse_rollout(rollout_builder.path)


@pytest.mark.parametrize(("parent_model", "child_model", "expected"), [
    ("", "model-child", "model-child"), ("model-parent", "", "model-parent"),
    ("", "", ""), ("shared", "shared", "shared"), ("model-parent", "model-child", "model-child,model-parent"),
])
def test_tree_metadata_merges_facts_without_empty_names(rollout_builder: RolloutBuilder, tmp_path: Path,
                                                       parent_model: str, child_model: str, expected: str) -> None:
    """Parent and child model sets are joined once, without an empty comma entry."""
    rollout_builder.meta(version="0.153.4")
    rollout_builder.add("turn_context", model=parent_model)
    child: RolloutBuilder = RolloutBuilder(tmp_path / ".codex/sessions/child.jsonl")
    child.meta("child", version="0.154.0", parent_thread_id="thread")
    child.add("turn_context", model=child_model)
    entities = read_entities(write_session_rrd(codex_session_source(rollout_builder.path).parse(), tmp_path / "metadata.rrd").path)
    assert entities["/__properties/session"]["models"].to_pylist() == [[expected]]
    assert entities["/__properties/session"]["cli_versions"].to_pylist() == [["0.153.4,0.154.0"]]


@pytest.mark.parametrize("native", [False, True])
def test_child_message_copies_with_different_turn_owners(rollout_builder: RolloutBuilder, tmp_path: Path, native: bool) -> None:
    """Same-time representations share a canonical source despite inherited turn metadata."""
    rollout_builder.meta()
    child = RolloutBuilder(tmp_path / ".codex/sessions/child.jsonl")
    child.meta("child", parent_thread_id="thread")
    child.add("turn_context", turn_id="child-turn")
    if native:
        child.item("AgentMessage", turn_id="child-turn", id="native-answer", content=[{"type": "text", "text": "verdict"}])
    else:
        child.add("event_msg", type="agent_message", message="verdict")
    child.index -= 1
    child.add("response_item", type="message", role="assistant", id="response-answer",
              internal_chat_message_metadata_passthrough={"turn_id": "inherited-turn"}, content=[{"type": "output_text", "text": "verdict"}])
    child.add("turn_context", turn_id="next-turn")
    child.add("event_msg", type="agent_message", message="verdict")
    session = parse_rollout(rollout_builder.path)
    replies = [row for row in session.subagents["child"] if isinstance(row.payload, AssistantText)]
    assert len(replies) == 2
    assert replies[0].timestamp_ns < replies[1].timestamp_ns
    entities = read_entities(write_session_rrd(session, tmp_path / "child.rrd").path)
    assert agent_rows(entities["/conversation/assistant"], "child")["TextLog:text"].to_pylist() == [["[a0] verdict"], ["[a0] verdict"]]


def test_model_calls_join_only_by_call_id(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """A script stays one call; executions join it only with an explicit key."""
    rollout_builder.meta()
    script: str = 'await tools.exec_command({cmd: "echo test"});'
    rollout_builder.add("response_item", type="custom_tool_call", name="exec", call_id="model-call", input=script)
    rollout_builder.item("CommandExecution", id="model-call", command=["echo", "test"], aggregated_output="first")
    rollout_builder.item("CommandExecution", id="execution", call_id="model-call", command=["echo", "test"], aggregated_output="second")
    rollout_builder.add("response_item", type="custom_tool_call_output", call_id="model-call", output="raw output")
    entities = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "calls.rrd").path)
    calls = entities["/tools"]
    assert calls["call_id"].to_pylist() == [["model-call"], ["model-call"]]
    assert calls["input_json"].to_pylist()[0] == [script]
    assert calls["TextLog:text"].to_pylist()[-1][0].endswith("raw output")
    executions = entities["/executions/command"]
    assert metadata_values(executions, 'item_id') == [["model-call"], ["execution"]]
    assert executions["call_id"].to_pylist() == [[], ["model-call"]]
    assert entities["/__properties/session"]["n_tool_calls"].to_pylist() == [[1]]
    assert not any("elapsed_ms/" in name for name in entities)


def test_child_without_native_items_keeps_messages_once(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Response messages supply children and duplicate native messages only once."""
    rollout_builder.meta()
    rollout_builder.item("AgentMessage", id="answer", content=[{"type": "text", "text": "reply"}])
    rollout_builder.add("response_item", type="message", role="assistant", id="answer", content=[{"type": "output_text", "text": "reply"}])
    child = RolloutBuilder(tmp_path / ".codex/sessions/child.jsonl")
    child.meta("child", parent_thread_id="thread")
    child.add("response_item", type="message", role="assistant", content=[{"type": "output_text", "text": "child reply"}])
    entities = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "children.rrd").path)
    assert agent_rows(entities["/conversation/assistant"], "")["TextLog:text"].to_pylist() == [["reply"]]
    assert agent_rows(entities["/conversation/assistant"], "child")["TextLog:text"].to_pylist() == [["[a0] child reply"]]


def test_context_inter_agent_and_compaction_are_retained(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Context stays readable, and unknown response subtypes remain counted."""
    rollout_builder.meta(base_instructions={"text": "base instruction"})
    rollout_builder.add("response_item", type="message", role="developer", content=[{"type": "input_text", "text": "developer instruction"}])
    rollout_builder.add("response_item", type="message", role="user", content=[{"type": "input_text", "text": "<environment_context>workspace</environment_context>"}])
    rollout_builder.add("response_item", type="agent_message", author="sender", recipient="recipient", content=[{"type": "text", "text": "agent task"}])
    rollout_builder.item("ContextCompaction")
    rollout_builder.add("compacted", message="", replacement_history=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "summary text"}]}])
    rollout_builder.add("response_item", type="reasoning", summary=[{"type": "summary_text", "text": "reasoning summary"}])
    rollout_builder.add("response_item", type="future_response")
    session = parse_rollout(rollout_builder.path)
    entities = read_entities(write_session_rrd(session, tmp_path / "context.rrd").path)
    for entity, text in {
        "context/base_instructions": "base instruction", "context/developer": "developer instruction",
        "context/environment": "<environment_context>workspace</environment_context>", "conversation/inter_agent": "agent task",
        "conversation/compaction": "summary text", "conversation/thinking": "reasoning summary",
    }.items():
        assert entities[f"/{entity}"]["TextLog:text"].to_pylist() == [[text]]
    assert session.skipped == {"response_item/future_response": 1}


def test_event_messages_are_a_deduplicated_fallback(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Event-only messages survive; a response copy does not duplicate a row."""
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="user_message", message="request")
    rollout_builder.add("event_msg", type="agent_message", message="reply")
    rollout_builder.add("response_item", type="message", role="assistant", content=[{"type": "output_text", "text": "reply"}])
    rollout_builder.add("event_msg", type="agent_message", message="event-only reply")
    entities = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "fallback.rrd").path)
    assert entities["/conversation/user"]["TextLog:text"].to_pylist() == [["request"]]
    assert entities["/conversation/assistant"]["TextLog:text"].to_pylist() == [["reply"], ["event-only reply"]]


def test_totals_use_last_available_snapshot_or_null(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Unknown usage is distinct from zero, and response totals take precedence."""
    rollout_builder.meta()
    assert parse_rollout(rollout_builder.path).properties["total_input_tokens"] is None
    rollout_builder.add("event_msg", type="token_count", info={"total_token_usage": {"input_tokens": 15, "output_tokens": 4}})
    assert parse_rollout(rollout_builder.path).properties["total_input_tokens"] == 15
    rollout_builder.add("token_usage_record", response_id="response", thread_token_usage={"input_tokens": 25, "output_tokens": 7})
    rollout_builder.add("event_msg", type="token_count", info={"total_token_usage": {"input_tokens": 99, "output_tokens": 9}})
    session = parse_rollout(rollout_builder.path)
    entities = read_entities(write_session_rrd(session, tmp_path / "totals.rrd").path)
    assert entities["/__properties/session"]["total_input_tokens"].to_pylist() == [[25]]
    assert entities["/__properties/session"]["total_output_tokens"].to_pylist() == [[7]]


def test_project_filter_uses_codex_working_directory(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Day directory names do not determine project matches."""
    rollout_builder.meta(cwd="/workspace/selected-project")
    rollout_builder.add("event_msg", type="agent_message", message="reply")
    home: Path = tmp_path / ".codex"
    convert_all_main(ConvertAllConfig(home=home, out=tmp_path / "out", project="selected-project"))
    assert (tmp_path / "out/codex/thread.rrd").is_file()
    convert_all_main(ConvertAllConfig(home=home, out=tmp_path / "excluded", project="unrelated"))
    assert not list((tmp_path / "excluded").rglob("*.rrd"))


def test_function_call_keeps_raw_arguments_without_native_items(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Raw function calls need no native completion or argument interpretation."""
    rollout_builder.meta()
    arguments: str = '{ "query": "example" }'
    rollout_builder.add("response_item", type="function_call", name="search", call_id="call", arguments=arguments)
    rollout_builder.add("response_item", type="function_call_output", call_id="call", output={"answer": "found"})
    entities = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "function.rrd").path)
    assert entities["/tools"]["call_id"].to_pylist() == [["call"], ["call"]]
    assert entities["/tools"]["input_json"].to_pylist()[0] == [arguments]
    assert entities["/tools"]["TextLog:text"].to_pylist()[-1][0].endswith('{"answer":"found"}')


def test_streamed_outputs_keep_each_part_and_deduplicate_explicit_ids(rollout_builder: RolloutBuilder) -> None:
    """One model call can produce many outputs, including repeated text at later times."""
    rollout_builder.meta()
    rollout_builder.add("response_item", type="function_call", call_id="call", name="exec_command", arguments="{}")
    rollout_builder.add("response_item", type="function_call", call_id="call", name="exec_command", arguments="{}")
    for identity, text in [("first", "running"), ("first", "running"), ("second", "failed: exit 3"), (None, "running"), (None, "running")]:
        rollout_builder.add("response_item", type="function_call_output", id=identity, call_id="call", output=text)
    session = parse_rollout(rollout_builder.path)
    calls = [row for row in session.main if isinstance(row.payload, ToolCall)]
    outputs = [row for row in session.main if isinstance(row.payload, ToolResult)]
    assert len(calls) == 1
    assert [row.payload.text for row in outputs if isinstance(row.payload, ToolResult)] == ["running", "failed: exit 3", "running", "running"]
    assert [row.file_index for row in outputs] == [3, 5, 6, 7]
    assert len({row.timestamp_ns for row in outputs}) == 4
    assert {row.payload.call_id for row in outputs if isinstance(row.payload, ToolResult)} == {"call"}


@pytest.mark.parametrize("prefix", ["<recommended_plugins>", "# AGENTS.md instructions", "<codex_internal_context", "<turn_aborted>", "<skill>", "<chat-history-summary>"])
def test_codex_harness_prompts_do_not_name_turns(rollout_builder: RolloutBuilder, prefix: str) -> None:
    """Harness instructions stay visible without becoming human requests."""
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="context")
    rollout_builder.add("response_item", type="message", role="user", content=[{"type": "text", "text": prefix + " instructions"}])
    rollout_builder.add("event_msg", type="task_started", turn_id="human")
    rollout_builder.add("response_item", type="message", role="user", content=[{"type": "text", "text": "<task>hello</task>"}])
    session = parse_rollout(rollout_builder.path)
    prompts = [row.payload for row in session.main if isinstance(row.payload, Prompt)]
    assert [prompt.role for prompt in prompts] == ["injected", "human"]
    assert [turn.prompt for turn in aggregate_turns(session.main)] == ["<task>hello</task>"]


def test_effort_updates_from_context_and_settings(rollout_builder: RolloutBuilder) -> None:
    """Each source sees the most recent effort from either update channel."""
    rollout_builder.meta()
    rollout_builder.add("turn_context", turn_id="turn", effort="high")
    rollout_builder.item("AgentMessage", id="first", content=[{"type": "text", "text": "first"}])
    rollout_builder.add("event_msg", type="thread_settings_applied", thread_settings={"reasoning_effort": "low"})
    rollout_builder.item("AgentMessage", id="second", content=[{"type": "text", "text": "second"}])
    rollout_builder.add("turn_context", turn_id="turn", effort="medium")
    rollout_builder.add("event_msg", type="thread_settings_applied", thread_settings={})
    rollout_builder.item("AgentMessage", id="third", content=[{"type": "text", "text": "third"}])
    assert [row.effort for row in parse_rollout(rollout_builder.path).main if isinstance(row.payload, AssistantText)] == ["high", "low", "medium"]


@pytest.mark.parametrize("native_count", [0, 1, 2])
def test_response_reasoning_preserves_each_occurrence(rollout_builder: RolloutBuilder, native_count: int) -> None:
    """Native reasoning replaces one corresponding response, never the whole turn."""
    rollout_builder.meta()
    rollout_builder.add("turn_context", turn_id="turn")
    for index, secret in enumerate(["one", "second"]):
        rollout_builder.add("response_item", type="reasoning", encrypted_content=secret)
        if index < native_count:
            rollout_builder.item("Reasoning", id=f"reason-{index}")
    rows = [row for row in parse_rollout(rollout_builder.path).main if isinstance(row.payload, Thinking)]
    assert [row.payload.text for row in rows if isinstance(row.payload, Thinking)] == ["<encrypted reasoning, 3 bytes>", "<encrypted reasoning, 6 bytes>"]
    assert [row.file_index for row in rows] == ([2, 3] if native_count == 0 else [3, 4] if native_count == 1 else [3, 5])


def test_prompt_identity_only_marks_turn_start(rollout_builder: RolloutBuilder) -> None:
    """Completion and reasoning have turn provenance but no prompt identity."""
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn")
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "hello"}])
    rollout_builder.item("Reasoning")
    rollout_builder.add("event_msg", type="task_complete", turn_id="turn")
    assert [(row.file_index, row.prompt_id) for row in parse_rollout(rollout_builder.path).main if row.prompt_id] == [(1, "turn")]


def test_child_header_and_execution_replays_are_unique(rollout_builder: RolloutBuilder) -> None:
    """A replayed child header and completed native item each emit once."""
    rollout_builder.meta("parent")
    child = RolloutBuilder(rollout_builder.path.parent / "rollout-child.jsonl")
    for _ in range(2):
        child.meta("child", parent_thread_id="parent", base_instructions={"text": "instructions"})
        child.item("CommandExecution", id="command", stdout="done", status="completed")
    rows = parse_rollout(rollout_builder.path).subagents["child"]
    assert len(rows) == 2


def test_provider_properties_have_declared_types(rollout_builder: RolloutBuilder) -> None:
    """Every property emitted by this provider belongs to the shared catalog schema."""
    rollout_builder.meta()
    session: Session = parse_rollout(rollout_builder.path)
    assert session.properties.keys() <= PROPERTY_TYPES.keys()


def test_unknown_event_keeps_local_images(rollout_builder: RolloutBuilder, png_bytes: bytes, tmp_path: Path) -> None:
    """Local images follow one policy even when their enclosing event is unknown."""
    image: Path = tmp_path / "local.png"
    image.write_bytes(png_bytes)
    rollout_builder.meta(cwd=str(tmp_path))
    rollout_builder.add("event_msg", type="future_event", local_images=["local.png"])
    session: Session = parse_rollout(rollout_builder.path)
    entities = read_entities(write_session_rrd(session, tmp_path / "local.rrd").path)
    assert metadata_values(entities["/media/images"], 'file_index') == [[1]]
    assert entities["/media/images"]["EncodedImage:blob"].to_pylist() == [[list(png_bytes)]]
    assert session.skipped == {"event_msg/future_event": 1}
    assert str(image) in session.extra_inputs


@pytest.mark.parametrize("tag", ["function_call_output", "custom_tool_call_output"])
def test_tool_output_images_are_media(rollout_builder: RolloutBuilder, tmp_path: Path, png_bytes: bytes, tag: str) -> None:
    """Tool output images retain bytes and call provenance without base64 in text."""
    import base64

    import orjson

    encoded: str = base64.b64encode(png_bytes).decode()
    rollout_builder.meta()
    rollout_builder.add("response_item", type="function_call", call_id="view", name="view_image", arguments="{}")
    output: list[dict[str, str]] = [
        {"type": "input_text", "text": "a red pixel"},
        {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}", "detail": "original"},
    ]
    for _ in range(2):
        rollout_builder.add("response_item", type=tag, id="result", call_id="view", output=output)
    entities = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "image-output.rrd").path)
    images = entities["/media/images"]
    assert images["EncodedImage:blob"].to_pylist() == [[list(png_bytes)]]
    assert images["call_id"].to_pylist() == [["view"]]
    assert metadata_values(images, 'source') == [["tool_result"]]
    assert entities["/__properties/session"]["n_images"].to_pylist() == [[1]]
    text = entities["/tools"]["TextLog:text"].to_pylist()[1][0]
    assert encoded not in text
    assert orjson.loads(text.split("  ", 1)[1]) == [output[0], {**output[1], "image_url": f"<image {len(png_bytes)} bytes, stored as media/images>"}]


def test_encrypted_inter_agent_content_has_placeholder(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Opaque messages retain their presence and byte size without revealing ciphertext."""
    rollout_builder.meta()
    rollout_builder.add("response_item", type="agent_message", author="parent", recipient="child", content=[
        {"type": "text", "text": "Payload:"}, {"type": "encrypted_content", "encrypted_content": "opaque"},
    ])
    entities = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "inter-agent.rrd").path)
    assert entities["/conversation/inter_agent"]["TextLog:text"].to_pylist() == [["Payload:\n<encrypted payload, 6 bytes>"]]


def test_output_replay_identity_keeps_native_tag(rollout_builder: RolloutBuilder) -> None:
    """The same output ID in different native tags denotes distinct output parts."""
    rollout_builder.meta()
    for tag in ("function_call_output", "custom_tool_call_output"):
        for _ in range(2):
            rollout_builder.add("response_item", type=tag, id="shared", call_id="call", output=tag)
    results = [row.payload.text for row in parse_rollout(rollout_builder.path).main if isinstance(row.payload, ToolResult)]
    assert results == ["function_call_output", "custom_tool_call_output"]


def test_codex_child_labels_use_shared_agent_table(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """Codex child roles and names survive the shared ordinal layout."""
    rollout_builder.meta()
    child = RolloutBuilder(rollout_builder.path.with_name("rollout-child.jsonl"))
    child.meta("child", parent_thread_id="thread", agent_role="reviewer", agent_nickname="Checker")
    child.add("response_item", type="message", role="assistant", content=[{"type": "output_text", "text": "checked"}])
    entities = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "labels.rrd").path)
    assert agent_rows(entities["/conversation/assistant"], "child")["agent_id"].to_pylist() == [["child"]]
    assert entities["/__properties/agents"]["agent_type"].to_pylist() == [["reviewer"]]
    assert entities["/__properties/agents"]["description"].to_pylist() == [["Checker"]]


def test_child_codex_text_families_show_attribution(tmp_path: Path) -> None:
    """The shared writer decorates executions, context and inter-agent child text."""
    from collections import Counter

    from agent_traces.events import ContextText, Execution, InterAgent, Session, TimedRecord

    records = [TimedRecord(Execution("command", "item", "output", "{}", "{}"), 1, 0),
               TimedRecord(ContextText("developer", "instructions"), 2, 1), TimedRecord(InterAgent("message"), 3, 2)]
    session = Session("child-families", "synthetic", tmp_path / "source.jsonl", [], {"child": records}, Counter(), agent="codex")
    entities = read_entities(write_session_rrd(session, tmp_path / "families.rrd").path)
    for family in ("/executions/command", "/context/developer", "/conversation/inter_agent"):
        assert entities[family]["TextLog:text"].to_pylist()[0][0].startswith("[a0] ")


@pytest.mark.parametrize(("input_tokens", "uncached"), [(0, 0), (3, 0), (7, 2)])
def test_codex_cached_input_is_separate_and_uncached_is_nonnegative(rollout_builder: RolloutBuilder, tmp_path: Path,
                                                                 input_tokens: int, uncached: int) -> None:
    """Requests, turns and session totals expose uncached input separately from cache reads."""
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn")
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "go"}])
    counters: dict[str, int] = {"input_tokens": input_tokens, "cached_input_tokens": 5}
    rollout_builder.add("token_usage_record", turn_id="turn", response_id="response", usage=counters, thread_token_usage=counters)
    entities = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "tokens.rrd").path)
    assert entities["/usage/input_tokens"]["Scalars:scalars"].to_pylist() == [[float(uncached)]]
    assert entities["/usage/cache_read_tokens"]["Scalars:scalars"].to_pylist() == [[5.0]]
    assert entities["/turns"]["input_tokens"].to_pylist() == [[uncached]]
    assert entities["/turns"]["cache_read_tokens"].to_pylist() == [[5]]
    assert entities["/__properties/session"]["total_input_tokens"].to_pylist() == [[uncached]]
    assert entities["/__properties/session"]["total_cache_read_tokens"].to_pylist() == [[5]]


@pytest.mark.parametrize(("name", "kind"), [("exec", "shell"), ("functions.exec_command", "shell"), ("js", "shell"),
    ("apply_patch", "file_edit"), ("spawn_agent", "subagent"), ("send_message", "subagent"), ("wait", "subagent"),
    ("view_image", "image"), ("update_plan", "plan"), ("web_search", "web_search"), ("mcp__server__tool", "mcp"), ("unknown", "other")])
def test_codex_model_tools_use_shared_kinds(rollout_builder: RolloutBuilder, name: str, kind: str) -> None:
    """Calls and results retain names while sharing the provider-neutral kind vocabulary."""
    rollout_builder.meta()
    rollout_builder.add("response_item", type="function_call", name=name, call_id="call", arguments="{}")
    rollout_builder.add("response_item", type="function_call_output", call_id="call", output="done")
    rows = parse_rollout(rollout_builder.path).main
    assert [row.payload.kind for row in rows if isinstance(row.payload, (ToolCall, ToolResult))] == [kind, kind]


def test_native_execution_counts_as_turn_activity(rollout_builder: RolloutBuilder, tmp_path: Path) -> None:
    """A native tool result can be the final turn activity without a model-tool output."""
    rollout_builder.meta()
    rollout_builder.add("event_msg", type="task_started", turn_id="turn")
    rollout_builder.item("UserMessage", content=[{"type": "text", "text": "go"}])
    rollout_builder.add("event_msg", type="item_completed", turn_id="turn", completed_at_ms=1789761603000,
                        item={"type": "CommandExecution", "id": "command", "aggregated_output": "done", "exit_code": 0})
    entities = read_entities(write_session_rrd(parse_rollout(rollout_builder.path), tmp_path / "native.rrd").path)
    assert entities["/turns"]["elapsed_ms"].to_pylist() == [[1000.0]]
