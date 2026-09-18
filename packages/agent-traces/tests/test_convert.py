"""Synthetic end-to-end CLI boundary."""

from pathlib import Path

import pytest

from tests.conftest import SessionBuilder
from tests.test_rerun_log import read_entities


def test_convert_writes_recording_and_prints_counts(session_builder: SessionBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The public entrypoint applies profile overrides and reports its output."""
    from agent_traces.apis.convert import Config, main

    session_builder.add("user", message={"content": "hello"})
    out: Path = tmp_path / "nested/session.rrd"
    main(Config(session=session_builder.path, out=out, profile="override"))
    assert out.is_file()
    assert read_entities(out)["/__properties/session"]["profile"].to_pylist() == [["override"]]
    output: str = capsys.readouterr().out
    assert "entities=5 rows=5" in output
    assert "conversation: 1" in output
    assert str(out) in output
