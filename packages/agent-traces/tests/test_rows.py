"""Neutral rows retain provenance and select only usable views."""

from collections import Counter
from pathlib import Path

import pyarrow as pa
import pytest

from agent_traces.blueprint import catalog_blueprint
from agent_traces.events import Prompt, Session, TimedRecord, ToolCall, ToolResult, TurnBoundary, Usage, UsageSample
from agent_traces.rerun_log import write_session_rrd
from agent_traces.rows import collect_rows
from tests.conftest import SessionBuilder, agent_rows, metadata_values, parse_session, read_entities


def test_rows_keep_typed_provenance_and_full_current_message(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Tool and usage rows can join the prompt; the current document is complete."""
    session_builder.add("user", uuid="turn", promptId="prompt", message={"content": "start"})
    text: str = "answer\n" * 100
    session_builder.add("assistant", promptId="prompt", effort="high", message={"id": "message", "model": "model", "content": [
        {"type": "text", "text": text}, {"type": "tool_use", "id": "call", "name": "Read", "input": {}}
    ], "usage": {"output_tokens": 12}})
    session_builder.add("user", message={"content": [{"type": "tool_result", "tool_use_id": "missing", "content": "result"}]})
    session_builder.add("user", path=session_builder.path.with_suffix("") / "subagents/agent-child.jsonl", message={"content": "child"})
    session = parse_session(session_builder.path)
    rows = collect_rows(session)
    assert rows.texts["tools"][0].values["turn_id"] == "turn"
    assert rows.scalars["usage/output_tokens"][0].values == {
        "file_index": 1, "agent_id": "", "turn_id": "turn", "prompt_id": "prompt", "message_id": "message", "model": "model", "effort": "high"
    }
    assert "elapsed/tools/unknown" not in rows.scalars
    assert rows.texts["conversation/user"][-1].values["agent_id"] == "child"
    assert rows.texts["conversation/user"][-1].text == "[a0] child"
    assert rows.texts["conversation/user"][-1].color != rows.texts["conversation/user"][0].color
    assert rows.texts["turns"][0].values["turn_id"] == "turn"
    entities = read_entities(write_session_rrd(session, tmp_path / "current.rrd").path)
    documents = entities["/conversation/current"]["TextDocument:text"].to_pylist()
    assert documents[-1][0].endswith(text)
    assert documents[-1][0].startswith("assistant | ")
    assert metadata_values(agent_rows(entities["/tools"], ""), 'message_id') [0] == ["message"]
    assert entities["/usage/output_tokens"]["turn_id"].to_pylist() == [["turn"]]


def test_saved_blueprints_select_children_and_omit_empty_views(session_builder: SessionBuilder, tmp_path: Path) -> None:
    """Recording capabilities trim views; the catalog blueprint keeps the full layout."""
    from rerun.chunk import RrdReader


    session_builder.add("user", message={"content": "hello"})
    recording = write_session_rrd(parse_session(session_builder.path), tmp_path / "views.rrd").path
    catalog = tmp_path / "catalog.rbl"
    catalog_blueprint().save("agent_traces", catalog)
    for path, full in ((recording, False), (catalog, True)):
        reader = RrdReader(path)
        names: set[str] = set()
        expressions: list[str] = []
        for chunk in reader.stream(store=reader.blueprints()[0]).to_chunks():
            batch = chunk.to_record_batch()
            if "ViewContents:query" in batch.schema.names:
                expressions.extend(expression for row in batch.column("ViewContents:query").to_pylist() for expression in row)
            if "ViewBlueprint:display_name" in batch.schema.names:
                names.update(name for row in batch.column("ViewBlueprint:display_name").to_pylist() for name in row)
        assert expressions
        assert all("**" not in expression.removesuffix("/**") for expression in expressions)
        assert {"Conversation", "Current message"} <= names
        assert ("Images" in names) is full
        assert ("Tool elapsed (ms)" in names) is full
        assert names == ({"Conversation", "Thinking", "Current message", "Tools", "Executions", "Context", "Lifecycle",
                          "Images", "Tokens per request", "Cache tokens", "Tool elapsed (ms)", "Turns"} if full else
                         {"Conversation", "Current message", "Turns"})


def test_scalar_components_exclude_provider_extras(tmp_path: Path) -> None:
    """Plots contain one measurement plus provenance; text retains result metadata."""
    provenance: dict[str, str] = {"turn_id": "turn", "prompt_id": "prompt", "message_id": "message", "model": "model", "effort": "high"}
    result: TimedRecord = TimedRecord(ToolResult("Read", "call", "output", '{"detail":"kept on text"}', "file_read", 250.0),
                         2_000_000_000, 2, {"provider_count": 99, "provider_text": "extra"}, **provenance)
    usage: TimedRecord = TimedRecord(UsageSample(Usage(input_tokens=12, output_tokens=3)), 3_000_000_000, 3,
                        {"provider_count": 88, "provider_text": "usage extra"}, **provenance)
    session: Session = Session("synthetic", "claude", tmp_path / "source.jsonl", [
        TimedRecord(TurnBoundary("start"), 1_000_000_000, 0, turn_id="turn", prompt_id="prompt"),
        TimedRecord(Prompt("request"), 1_000_000_000, 0, turn_id="turn"), result, usage,
    ], {"child": [result, usage]}, Counter())
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(session, tmp_path / "scalars.rrd").path)
    for path, table in entities.items():
        if "Scalars:scalars" not in table.column_names:
            continue
        components: set[str] = {field.name for field in table.schema if (field.metadata or {}).get(b"rerun:kind") == b"data"}
        expected: set[str] = {"Scalars:scalars", "turn_id", "agent_id", "metadata_json", "model", "effort"}
        if not path.startswith("/turns/"):
            assert all(value == ["message"] for value in metadata_values(table, "message_id"))
        if path.startswith("/elapsed/tools/"):
            expected.update({"call_id", "tool"})
            assert all(value == ["call"] for value in table["call_id"].to_pylist())
            assert all(value == [250.0] for value in table["Scalars:scalars"].to_pylist())
        if path.startswith("/usage/"):
            expected.add("model")
        assert components == expected, path
    assert agent_rows(entities["/tools"], "")["tool_use_result_json"].to_pylist() == [['{"detail":"kept on text"}']]
    assert metadata_values(agent_rows(entities["/tools"], ""), 'provider_count') == [[99]]
    assert entities["/usage/input_tokens"]["Scalars:scalars"].to_pylist() == [[12.0]]


def test_tool_timing_and_images_share_one_call_column(tmp_path: Path) -> None:
    """Tool text, timing and images join without mixing scalar and text families."""
    from agent_traces.events import Image, ToolCall

    records = [
        TimedRecord(ToolCall("Read", "call", "{}", "file_read"), 1, 0),
        TimedRecord(ToolResult("Read", "call", "done", "{}", "file_read", 25.0), 2, 1),
        TimedRecord(Image(b"image", "image/png", "call", "tool_result"), 2, 1),
    ]
    session = Session("synthetic", "test", tmp_path / "source.jsonl", records, {"child": records}, Counter())
    entities = read_entities(write_session_rrd(session, tmp_path / "joins.rrd").path)
    for identity in ("", "child"):
        for prefix in ("tools", "elapsed/tools/file_read", "media/images"):
            table = agent_rows(entities[f"/{prefix}" + ("/children" if identity and prefix.startswith("elapsed/") else "")], identity)
            assert "tool_use_id" not in table.column_names
            assert all(value == ["call"] for value in table["call_id"].to_pylist())
        assert "Scalars:scalars" not in entities["/tools"].column_names
    assert not any(path.startswith("/tools/elapsed_ms") for path in entities)


def test_child_table_uses_time_order_and_keeps_agent_identity(tmp_path: Path) -> None:
    """The agent table orders by first time and ID while shared rows retain their original IDs."""
    from agent_traces.events import Image, Lifecycle, ToolCall

    records = [
        TimedRecord(Prompt("child"), 20, 0),
        TimedRecord(ToolCall("Read", "call", "{}", "file_read"), 21, 1),
        TimedRecord(ToolResult("Read", "call", "done", "{}", "file_read", 1.0, agent_id="spawned"), 22, 2),
        TimedRecord(UsageSample(Usage(input_tokens=3)), 23, 3),
        TimedRecord(Image(b"image", "image/png"), 24, 4),
        TimedRecord(Lifecycle("system", "ready"), 25, 5),
    ]
    session = Session("ordering", "synthetic", tmp_path / "source.jsonl", [], {
        "z": records, "b": [TimedRecord(Prompt("tie b"), 10, 0)],
        "a": [TimedRecord(Prompt("tie a"), 10, 0)], "empty": [],
    }, Counter())
    entities = read_entities(write_session_rrd(session, tmp_path / "ordinals.rrd").path)
    assert entities["/conversation/user"]["agent_id"].to_pylist() == [["b"], ["a"], ["z"]]
    for path, table in entities.items():
        if not path.startswith("/__properties"):
            assert set(value[0] for value in table["agent_id"].to_pylist()) <= {"a", "b", "z"}, path
    assert metadata_values(entities["/tools"], "child_agent_id")[-1] == ["spawned"]
    agents = entities["/__properties/agents"]
    assert agents["n"].to_pylist() == [[0, 1, 2, 3]]
    assert agents["agent_id"].to_pylist() == [["a", "b", "z", "empty"]]


def test_content_metadata_uses_one_string_column(tmp_path: Path) -> None:
    """Content schemas stay narrow while JSON preserves sparse metadata and native types."""
    import orjson

    from agent_traces.events import Image

    session = Session("metadata", "synthetic", tmp_path / "source.jsonl", [
        TimedRecord(TurnBoundary("start"), 1, 0, turn_id="turn"),
        TimedRecord(Prompt("question"), 1, 0, {"provider_flag": True}, turn_id="turn", prompt_id="prompt"),
        TimedRecord(ToolResult("Read", "call", "done", '{"result":7}', "file_read", 2.5), 2, 1,
                    {"provider_count": 7}, turn_id="turn", message_id="message", model="model", effort="high"),
        TimedRecord(UsageSample(Usage(input_tokens=3)), 3, 2, turn_id="turn"),
        TimedRecord(Image(b"image", "image/png", "call", "tool_result"), 4, 3, turn_id="turn"),
    ], {}, Counter())
    entities = read_entities(write_session_rrd(session, tmp_path / "metadata.rrd").path)
    for path, table in entities.items():
        if path.startswith("/__properties") or path == "/turns":
            continue
        components = {field.name for field in table.schema if (field.metadata or {}).get(b"rerun:kind") == b"data"}
        custom = {name for name in components if ":" not in name}
        assert custom <= {"turn_id", "call_id", "agent_id", "metadata_json", "tool", "phase", "kind", "is_error", "elapsed_ms", "input_json", "tool_use_result_json", "model", "effort"}, (path, custom)
        assert pa.types.is_string(table.schema.field("metadata_json").type.value_type)
        assert "metadata_json" in custom
    metadata = orjson.loads(entities["/tools"]["metadata_json"].to_pylist()[0][0])
    assert metadata == {"file_index": 1, "message_id": "message", "provider_count": 7,
                        "child_agent_id": ""}
    assert entities["/tools"]["turn_id"].to_pylist() == [["turn"]]
    assert entities["/tools"]["call_id"].to_pylist() == [["call"]]
    assert metadata_values(entities["/turns"], "file_index") == [[0]]
    assert entities["/turns"]["n_images"].to_pylist() == [[1]]


def test_children_share_paths_and_keep_typed_identity(tmp_path: Path) -> None:
    """More children add rows, not entity paths or columns, and do not replace current message."""
    main = [TimedRecord(Prompt("main"), 1, 0, turn_id="turn")]
    children = {f"child-{n}": [TimedRecord(Prompt(f"child {n}"), n + 2, 0, turn_id="turn")] for n in range(20)}
    session = Session("bounded", "synthetic", tmp_path / "source.jsonl", main, children, Counter())
    entities = read_entities(write_session_rrd(session, tmp_path / "bounded.rrd").path)
    assert not any("/agents/" in path for path in entities)
    assert entities["/conversation/user"].num_rows == 21
    assert entities["/conversation/user"]["agent_id"].to_pylist() == [[""], *[[f"child-{n}"] for n in range(20)]]
    assert entities["/conversation/current"].num_rows == 1
    assert entities["/conversation/current"]["agent_id"].to_pylist() == [[""]]


def test_tools_are_values_with_small_typed_query_columns(tmp_path: Path) -> None:
    """Different tool names share a schema; queries need neither text nor raw JSON."""
    import orjson

    from agent_traces.events import ToolCall

    records = [
        TimedRecord(ToolCall("Read", "read", '{"path":"large input"}', "file_read"), 1, 0),
        TimedRecord(ToolResult("Read", "read", "output", '{"large":"output"}', "file_read", 2.5, True), 2, 1),
        TimedRecord(ToolCall("custom", "other", "{}", "other"), 3, 2),
        TimedRecord(ToolResult("custom", "other", "unknown time", "{}", "other", None), 4, 3),
        TimedRecord(UsageSample(Usage(output_tokens=7)), 5, 4, model="main-model"),
    ]
    session = Session("tools", "synthetic", tmp_path / "source.jsonl", records,
                      {"child": [TimedRecord(UsageSample(Usage(output_tokens=3)), 6, 0, model="child-model"),
                                 TimedRecord(ToolResult("custom", "child-call", "done", "{}", "other", 1.0), 7, 1)]}, Counter())
    entities = read_entities(write_session_rrd(session, tmp_path / "tools.rrd").path)
    assert not any(path.startswith("/tools/") for path in entities)
    tools = entities["/tools"]
    for key, dtype in {"tool": pa.string(), "phase": pa.string(), "kind": pa.string(), "is_error": pa.bool_(),
                       "elapsed_ms": pa.float64(), "call_id": pa.string(), "agent_id": pa.string(),
                       "input_json": pa.string(), "tool_use_result_json": pa.string()}.items():
        assert tools.schema.field(key).type.value_type == dtype
    assert tools["tool"].to_pylist() == [["Read"], ["Read"], ["custom"], ["custom"], ["custom"]]
    assert tools["phase"].to_pylist() == [["call"], ["result"], ["call"], ["result"], ["result"]]
    assert tools["is_error"].to_pylist() == [[False], [True], [False], [False], [False]]
    assert tools["elapsed_ms"].to_pylist() == [[None], [2.5], [None], [None], [1.0]]
    assert tools["input_json"].to_pylist() == [['{"path":"large input"}'], [None], ["{}"], [None], [None]]
    assert tools["tool_use_result_json"].to_pylist()[1] == ['{"large":"output"}']
    for value in tools["metadata_json"].to_pylist():
        assert not {"tool", "phase", "kind", "is_error", "elapsed_ms", "input_json", "tool_use_result_json"} & orjson.loads(value[0]).keys()
    assert entities["/elapsed/tools/file_read"]["tool"].to_pylist() == [["Read"]]
    assert entities["/elapsed/tools/other/children"]["call_id"].to_pylist() == [["child-call"]]
    assert entities["/usage/output_tokens"]["model"].to_pylist() == [["main-model"]]
    assert entities["/usage/output_tokens/children"]["model"].to_pylist() == [["child-model"]]


def test_event_sequence_preserves_wall_ties_and_source_order(tmp_path: Path) -> None:
    """Every temporal row has a unique sequence, ordered by wall then source record."""
    from agent_traces.events import AssistantText, ToolCall

    session = Session("sequence", "synthetic", tmp_path / "source.jsonl", [
        TimedRecord(ToolCall("Read", "first", "{}", "file_read"), 20, 1),
        TimedRecord(AssistantText("middle"), 20, 1),
        TimedRecord(ToolCall("Read", "last", "{}", "file_read"), 20, 1),
        TimedRecord(UsageSample(Usage(output_tokens=2)), 10, 0, model="model"),
    ], {}, Counter())
    signatures = []
    for attempt in range(2):
        entities = read_entities(write_session_rrd(session, tmp_path / f"sequence-{attempt}.rrd").path)
        temporal = [(path, row) for path, table in entities.items() if not path.startswith("/__properties")
                    for row in table.to_pylist()]
        assert sorted(row["event"] for _, row in temporal) == list(range(len(temporal)))
        ordered = sorted(temporal, key=lambda item: item[1]["event"])
        assert [row["wall"] for _, row in ordered] == sorted(row["wall"] for _, row in temporal)
        text = [row["TextLog:text"][0] for _, row in ordered if "TextLog:text" in row]
        assert text == ["▶ Read  {}", "middle", "▶ Read  {}"]
        assert entities["/tools"]["wall"].cast(pa.int64()).to_pylist() == [20, 20]
        assert all("event" not in table.column_names for path, table in entities.items() if path.startswith("/__properties"))
        signatures.append([(path, row["event"], row["wall"]) for path, row in ordered])
    assert signatures[0] == signatures[1]


def test_declared_facts_are_typed_on_every_entity_and_absent_values_are_empty(tmp_path: Path) -> None:
    """A fact's type and null policy do not depend on its entity family."""
    session = Session("facts", "synthetic", tmp_path / "source.jsonl", [
        TimedRecord(Prompt("with facts"), 1, 0, {"tool": "Read", "phase": "call", "kind": "file_read", "is_error": False,
                    "elapsed_ms": 2.5, "input_json": "{}", "provider_note": "small"}, model="model"),
        TimedRecord(Prompt("without facts"), 2, 1),
    ], {}, Counter())
    table = read_entities(write_session_rrd(session, tmp_path / "facts.rrd").path)["/conversation/user"]
    for key, dtype, value in (("tool", pa.string(), "Read"), ("is_error", pa.bool_(), False),
                              ("elapsed_ms", pa.float64(), 2.5), ("model", pa.string(), "model"), ("input_json", pa.string(), "{}")):
        assert table.schema.field(key).type.value_type == dtype
        assert table[key].to_pylist() == [[value], []]
        assert metadata_values(table, key) == [[], []]
    assert metadata_values(table, "provider_note") == [["small"], []]


def test_every_row_gets_one_emission_order_including_turns(tmp_path: Path) -> None:
    """Derived documents, counters and turns all use the same insertion counter."""
    session = Session("emission", "synthetic", tmp_path / "source.jsonl", [
        TimedRecord(TurnBoundary("start"), 1, 0, turn_id="turn"),
        TimedRecord(Prompt("question"), 1, 0, turn_id="turn"),
        TimedRecord(UsageSample(Usage(input_tokens=2, output_tokens=1)), 1, 1, turn_id="turn"),
    ], {}, Counter())
    rows = collect_rows(session)
    orders = [row.event for family in (rows.texts, rows.documents, rows.scalars, rows.images)
              for batch in family.values() for row in batch]
    assert sorted(orders) == list(range(len(orders)))
    assert rows.texts["conversation/user"][0].event < rows.documents["conversation/current"][0].event
    assert rows.scalars["usage/output_tokens"][0].event < rows.texts["turns"][0].event


def test_turn_and_usage_effort_is_a_typed_query_fact(tmp_path: Path) -> None:
    """Effort levels can group turn and request tokens without decoding metadata."""
    session: Session = Session("effort", "claude", tmp_path / "source.jsonl", [
        TimedRecord(TurnBoundary("start"), 1, 0, turn_id="turn"),
        TimedRecord(Prompt("question"), 1, 0, turn_id="turn"),
        TimedRecord(UsageSample(Usage(output_tokens=12)), 2, 1, turn_id="turn", effort="high"),
    ], {}, Counter())
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(session, tmp_path / "effort.rrd").path)
    for entity in ("/turns", "/usage/output_tokens"):
        table: pa.Table = entities[entity]
        assert table.schema.field("effort").type == pa.list_(pa.string())
        assert table["effort"].to_pylist() == [["high"]]
        assert metadata_values(table, "effort") == [[]]


@pytest.mark.parametrize("payload", [ToolCall("custom", "call", "{}", "other"),
                                     ToolResult("custom", "call", "done", "{}", "other", None)])
def test_tools_schema_exists_with_only_calls_or_unknown_duration_results(tmp_path: Path, payload: ToolCall | ToolResult) -> None:
    """Partial and Codex-style tool streams expose the same query columns with typed nulls."""
    session: Session = Session("tools", "synthetic", tmp_path / "source.jsonl", [TimedRecord(payload, 1, 0)], {}, Counter())
    entities: dict[str, pa.Table] = read_entities(write_session_rrd(session, tmp_path / "tools.rrd").path)
    table: pa.Table = entities["/tools"]
    for name, dtype in {"tool": pa.string(), "phase": pa.string(), "kind": pa.string(), "call_id": pa.string(),
                        "agent_id": pa.string(), "input_json": pa.string(), "tool_use_result_json": pa.string(),
                        "is_error": pa.bool_(), "elapsed_ms": pa.float64()}.items():
        assert table.schema.field(name).type == pa.list_(dtype)
    assert table["elapsed_ms"].to_pylist() == [[None]]
    assert table["input_json"].to_pylist() == ([["{}"]] if isinstance(payload, ToolCall) else [[None]])
    assert table["tool_use_result_json"].to_pylist() == ([[None]] if isinstance(payload, ToolCall) else [["{}"]])
    assert not any(entity.startswith("/elapsed/") for entity in entities)


def test_next_prompt_timestamp_does_not_extend_previous_turn(tmp_path: Path) -> None:
    """An injected row at the next prompt joins that turn and cannot create a month-long duration."""
    from agent_traces.events import AssistantText

    next_prompt: int = 30 * 86400 * 1_000_000_000
    session: Session = Session("turns", "synthetic", tmp_path / "source.jsonl", [
        TimedRecord(TurnBoundary("start"), 0, 0, turn_id="one"),
        TimedRecord(Prompt("first"), 0, 0, turn_id="one"),
        TimedRecord(AssistantText("done"), 19 * 60 * 1_000_000_000, 1, turn_id="one"),
        TimedRecord(Prompt("injected", "injected"), next_prompt, 2, turn_id="one"),
        TimedRecord(TurnBoundary("start"), next_prompt, 3, turn_id="two"),
        TimedRecord(Prompt("second"), next_prompt, 3, turn_id="two"),
    ], {}, Counter())
    rows = collect_rows(session)
    assert [row.values["elapsed_ms"] for row in rows.texts["turns"]] == [19 * 60 * 1000.0, 0.0]
    assert rows.texts["conversation/injected"][0].values["turn_id"] == "two"


@pytest.mark.parametrize("spawned", [False, True])
def test_child_rows_inherit_spawn_turn_or_main_prompt_interval(tmp_path: Path, spawned: bool) -> None:
    """An explicit spawn call beats later timestamps; unlinked children use each row's interval."""
    main: list[TimedRecord] = [
        TimedRecord(TurnBoundary("start"), 10, 0, turn_id="one"),
        TimedRecord(Prompt("first"), 10, 0, turn_id="one"),
        TimedRecord(TurnBoundary("start"), 20, 2, turn_id="two"),
        TimedRecord(Prompt("second"), 20, 2, turn_id="two"),
    ]
    if spawned:
        main.extend([TimedRecord(ToolCall("Agent", "spawn", "{}", "subagent"), 11, 1, turn_id="one"),
                     TimedRecord(ToolResult("Agent", "spawn", "started", "{}", "subagent", None, agent_id="child"), 21, 3, turn_id="two")])
    session: Session = Session("children", "synthetic", tmp_path / "source.jsonl", main, {"child": [
        TimedRecord(Prompt("child"), 12, 0, turn_id="child-local"),
        TimedRecord(ToolCall("Read", "read", "{}", "file_read"), 22, 1),
        TimedRecord(UsageSample(Usage(output_tokens=3)), 23, 2),
    ]}, Counter())
    rows = collect_rows(session)
    assert next(row for row in rows.texts["conversation/user"] if row.values["agent_id"] == "child").values["turn_id"] == "one"
    assert rows.texts["tools"][-1].values["turn_id"] == ("one" if spawned else "two")
    assert rows.scalars["usage/output_tokens/children"][0].values["turn_id"] == ("one" if spawned else "two")
