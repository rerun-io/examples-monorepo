"""The pure turn boundary preserves input records and resets per-turn usage."""

from agent_traces.events import AssistantText, Prompt, Session, TimedRecord, TurnBoundary, Usage, UsageSample
from agent_traces.turns import Turn, aggregate_turns
from tests.conftest import SessionBuilder, parse_session


def test_turn_usage_resets_and_inputs_remain_unchanged(session_builder: SessionBuilder) -> None:
    """Repeated IDs count once per transcript; returned accumulators are independent."""
    session_builder.add("assistant", message={"id": "preamble", "content": "preamble", "usage": {"input_tokens": 99}})
    for prompt in ["first", "second"]:
        session_builder.add("user", uuid=prompt, message={"content": prompt})
        session_builder.add("assistant", message={"id": "same", "content": "reply", "usage": {"input_tokens": 7}})
        session_builder.add("assistant", message={"id": "same", "usage": {"input_tokens": 99}})
    session: Session = parse_session(session_builder.path)
    turns: list[Turn] = aggregate_turns(session.main)
    assert [turn.prompt for turn in turns] == ["first", "second"]
    assert [(turn.usage.input_tokens or 0) for turn in turns] == [0, 99]
    assert [turn.n_assistant_messages for turn in turns] == [1, 1]
    assert [turn.elapsed_ms for turn in turns] == [1000.0, 2000.0]
    assert [turn.file_index for turn in turns] == [1, 4]
    turns[0].usage = Usage(input_tokens=100)
    assert aggregate_turns(session.main)[1].usage.input_tokens == 99
    assert session.main == parse_session(session_builder.path).main


def test_empty_source_records_do_not_extend_turns(session_builder: SessionBuilder) -> None:
    """The finalized usage advances time; an unemitted URL image does not."""
    session_builder.add("user", message={"content": "first"})
    session_builder.add("assistant", message={"id": "m", "content": "reply"})
    session_builder.add("assistant", message={"id": "m", "content": []})
    session_builder.add("user", message={"content": [{"type": "image", "source": {"type": "url", "url": "https://example.test/image.png"}}]})
    turns: list[Turn] = aggregate_turns(parse_session(session_builder.path).main)
    assert len(turns) == 1
    assert turns[0].elapsed_ms == 2000.0
    assert turns[0].n_assistant_messages == 1


def test_turn_fold_uses_typed_identity_and_sums_reported_usage() -> None:
    """Late usage joins the next prompt interval; metadata cannot override identity."""
    records: list[TimedRecord] = [
        TimedRecord(AssistantText("preamble"), 0, 0, message_id="preamble"),
        TimedRecord(TurnBoundary("start"), 1_000_000, 1, {"prompt_id": "spoof"}, turn_id="one", prompt_id="p1"),
        TimedRecord(Prompt("first"), 1_000_000, 1, turn_id="one"),
        TimedRecord(AssistantText("reply"), 2_000_000, 2, {"message_id": "spoof"}, turn_id="one", message_id="m1"),
        TimedRecord(AssistantText("reply continued"), 3_000_000, 3, {"message_id": "different"}, turn_id="one", message_id="m1"),
        TimedRecord(TurnBoundary("complete", 12.0), 4_000_000, 4, turn_id="one"),
        TimedRecord(TurnBoundary("start"), 5_000_000, 5, turn_id="two"),
        TimedRecord(Prompt("second"), 5_000_000, 5, turn_id="two"),
        TimedRecord(UsageSample(Usage(input_tokens=7, cache_creation_5m_tokens=2)), 6_000_000, 6, turn_id="one"),
        TimedRecord(UsageSample(Usage(input_tokens=3, cache_creation_1h_tokens=4)), 7_000_000, 7, turn_id="one"),
    ]
    turns: list[Turn] = aggregate_turns(records)
    assert [turn.prompt for turn in turns] == ["first", "second"]
    assert turns[0].prompt_id == "p1"
    assert turns[0].elapsed_ms == 2.0
    assert turns[0].n_assistant_messages == 1
    assert turns[1].usage.input_tokens == 10
    assert turns[1].usage.cache_creation_5m_tokens == 2
    assert turns[1].usage.cache_creation_1h_tokens == 4
    assert turns[0].usage.output_tokens is None
    assert turns[0].usage == Usage()
    assert turns[1].elapsed_ms == 2.0


def test_interagent_ids_are_not_assistant_messages() -> None:
    """Peer messages must not inflate assistant response counts."""
    from agent_traces.events import InterAgent

    records = [
        TimedRecord(TurnBoundary("start"), 0, 0, turn_id="turn"),
        TimedRecord(AssistantText("answer"), 1, 1, turn_id="turn", message_id="assistant"),
        TimedRecord(InterAgent("peer"), 2, 2, turn_id="turn", message_id="peer"),
    ]
    assert aggregate_turns(records)[0].n_assistant_messages == 1
