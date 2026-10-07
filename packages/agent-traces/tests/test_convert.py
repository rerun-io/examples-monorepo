"""Synthetic end-to-end CLI boundary."""

from pathlib import Path

import pytest

from agent_traces.apis.convert import Config as ConvertConfig
from agent_traces.apis.convert import main as convert_main
from tests.conftest import SessionBuilder, read_entities


def test_convert_writes_recording_and_prints_counts(session_builder: SessionBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The public entrypoint applies profile overrides and reports its output."""
    session_builder.add("user", message={"content": "hello"})
    out: Path = tmp_path / "nested/session.rrd"
    convert_main(ConvertConfig(session=session_builder.path, out=out, profile="override"))
    assert out.is_file()
    assert read_entities(out)["/__properties/session"]["profile"].to_pylist() == [["override"]]
    output: str = capsys.readouterr().out
    assert "entities=6 rows=6" in output
    assert "conversation: 2" in output
    assert str(out) in output
