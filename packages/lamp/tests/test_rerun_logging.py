"""Rerun lifecycle logging tests for tracked people."""

import pytest
import rerun as rr

from lamptrack.rerun_logging import LivePeopleLogger


def test_ended_track_is_cleared_once_on_the_next_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A track ending at frame k is cleared at k+1 and never cleared again."""
    logged: list[tuple[str, object]] = []

    def capture_log(entity_path: str, archetype: object, *, static: bool = False) -> None:
        del static
        logged.append((entity_path, archetype))

    monkeypatch.setattr(rr, "log", capture_log)
    people = LivePeopleLogger()

    people.update({17})
    assert logged == []

    people.update(set())
    assert len(logged) == 1
    entity_path, archetype = logged[0]
    assert entity_path == "world/people/17"
    assert isinstance(archetype, rr.Clear)
    assert archetype.is_recursive.as_arrow_array().to_pylist() == [True]

    people.update(set())
    assert len(logged) == 1
