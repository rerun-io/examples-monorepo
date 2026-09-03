"""Rerun lifecycle logging tests for tracked people."""

import pytest
import rerun as rr

from lamptrack.rerun_logging import LivePeopleLogger, log_smpl_annotation_context
from lamptrack.third_party.lamp.core.types import SMPL_JOINT_NAMES, SMPL_SKELETON_EDGES


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


def test_smpl_annotation_context_names_all_24_joints_and_their_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """One static context under ``world/people`` draws every person's edges."""
    logged: list[tuple[str, object, bool]] = []
    monkeypatch.setattr(rr, "log", lambda entity_path, archetype, **kwargs: logged.append((entity_path, archetype, kwargs["static"])))

    log_smpl_annotation_context()

    assert len(logged) == 1
    entity_path, archetype, static = logged[0]
    assert (entity_path, static) == ("world/people", True)
    assert isinstance(archetype, rr.AnnotationContext)
    description = archetype.context.as_arrow_array().to_pylist()[0][0]["class_description"]
    assert [entry["label"] for entry in description["keypoint_annotations"]] == list(SMPL_JOINT_NAMES)
    assert len(description["keypoint_annotations"]) == 24
    assert len(description["keypoint_connections"]) == len(SMPL_SKELETON_EDGES)
