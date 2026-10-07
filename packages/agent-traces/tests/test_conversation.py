"""Representation selection retains occurrences while matching duplicate copies."""

from agent_traces import codex
from agent_traces import codex_records as cr


def test_equal_text_copies_are_consumed_one_to_one() -> None:
    """Different IDs across formats still match, but later repeats survive."""
    entries = [
        codex.ContextualRecord(codex.SourceRecord(cr.ItemCompleted( item=cr.MessageItem(type="AgentMessage", id="native", content=[cr.Content(type="text", text="same")])), "event_msg", 1, 0), "turn", "", ""),
        codex.ContextualRecord(codex.SourceRecord(cr.Message( role="assistant", id="copy", content=[cr.Content(type="text", text="same")]), "response_item", 1, 1), "inherited", "", ""),
        codex.ContextualRecord(codex.SourceRecord(cr.AgentMessage( message="same"), "event_msg", 1, 2), "turn", "", ""),
        codex.ContextualRecord(codex.SourceRecord(cr.Message( role="assistant", id="later", content=[cr.Content(type="text", text="same")]), "response_item", 2, 3), "turn", "", ""),
        codex.ContextualRecord(codex.SourceRecord(cr.AgentMessage( message="same"), "event_msg", 2, 4), "turn", "", ""),
    ]
    selected = codex.select_conversation(entries)
    assert [index for index, candidate in enumerate(selected) if candidate is not None] == [0, 3]
    assert selected == codex.select_conversation(entries)
