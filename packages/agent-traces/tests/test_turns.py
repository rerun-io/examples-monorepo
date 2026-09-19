"""The pure turn boundary preserves input records and resets per-turn usage."""

from agent_traces.claude import ClaudeSession, parse_session
from agent_traces.turns import Turn, aggregate_turns
from tests.conftest import SessionBuilder


def test_turn_usage_resets_and_inputs_remain_unchanged(session_builder: SessionBuilder) -> None:
    """Repeated IDs count once in each turn; returned accumulators are independent."""
    session_builder.add("assistant", message={"id": "preamble", "content": "preamble", "usage": {"input_tokens": 99}})
    for prompt in ["first", "second"]:
        session_builder.add("user", message={"content": prompt})
        session_builder.add("assistant", message={"id": "same", "content": "reply", "usage": {"input_tokens": 7}})
        session_builder.add("assistant", message={"id": "same", "usage": {"input_tokens": 99}})
    session: ClaudeSession = parse_session(session_builder.path)
    turns: list[Turn] = aggregate_turns(session.main)
    assert [turn.prompt for turn in turns] == ["first", "second"]
    assert [turn.input_tokens for turn in turns] == [7, 7]
    assert [turn.n_assistant_messages for turn in turns] == [1, 1]
    assert [turn.elapsed_ms for turn in turns] == [2000.0, 2000.0]
    assert [turn.file_index for turn in turns] == [1, 4]
    turns[0].input_tokens = 100
    assert aggregate_turns(session.main)[0].input_tokens == 7
    assert session.main == parse_session(session_builder.path).main
