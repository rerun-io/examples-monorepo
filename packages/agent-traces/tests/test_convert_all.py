"""Batch conversion through its public entrypoint and persisted outputs."""

from pathlib import Path

import pytest

from tests.conftest import SessionBuilder
from tests.test_rerun_log import read_entities


def test_batch_resumes_and_hashes_subagents(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Three sessions convert once; changed children and missing outputs rebuild."""
    from agent_traces.apis.convert_all import Config, Manifest, load_manifest, main

    home: Path = tmp_path / ".claude"
    for project, session_id in [("one", "a"), ("one", "b"), ("two", "c")]:
        SessionBuilder(home / "projects" / project / f"{session_id}.jsonl").add("user", message={"content": session_id})
    child: SessionBuilder = SessionBuilder(home / "projects/one/a/subagents/agent-child.jsonl")
    child.add("user", message={"content": "child"})
    SessionBuilder(home / "projects/one/agent-stray.jsonl").add("user", message={"content": "stray"})
    config: Config = Config(home=home, out=tmp_path / "out")
    main(config)
    assert "converted=3 skipped=0 failed=0" in capsys.readouterr().out
    manifest_path: Path = tmp_path / "out/claude/manifest.json"
    manifest: Manifest = load_manifest(manifest_path)
    assert set(manifest.sessions) == {"a", "b", "c"}
    assert {session_id: entry.n_rows for session_id, entry in manifest.sessions.items()} == {"a": 6, "b": 5, "c": 5}
    assert read_entities(tmp_path / "out/claude/a.rrd")["/__properties/session"]["profile"].to_pylist() == [["claude"]]
    main(config)
    assert "converted=0 skipped=3 failed=0" in capsys.readouterr().out
    child.add("assistant", message={"content": "new"})
    main(config)
    assert "converted=1 skipped=2 failed=0" in capsys.readouterr().out
    assert load_manifest(manifest_path).sessions["a"].n_rows == 7
    (tmp_path / "out/claude/b.rrd").unlink()
    main(config)
    assert "converted=1 skipped=2 failed=0" in capsys.readouterr().out

    late: SessionBuilder = SessionBuilder(home / "projects/one/a/subagents/agent-late.jsonl")
    late.add("user", message={"content": "late child"})
    main(config)
    assert "converted=1 skipped=2 failed=0" in capsys.readouterr().out
    assert load_manifest(manifest_path).sessions["a"].n_rows == 8


def test_batch_filters_and_profile(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Project, session, and UTC modification-date filters select main files."""
    import os

    from agent_traces.apis.convert_all import Config, load_manifest, main

    home: Path = tmp_path / ".claude"
    for project, session_id in [("one", "a"), ("one", "b"), ("two", "c")]:
        SessionBuilder(home / "projects" / project / f"{session_id}.jsonl").add("user", message={"content": session_id})
    main(Config(home=home, out=tmp_path / "project", project="on", profile="custom"))
    assert set(load_manifest(tmp_path / "project/custom/manifest.json").sessions) == {"a", "b"}
    assert read_entities(tmp_path / "project/custom/a.rrd")["/__properties/session"]["profile"].to_pylist() == [["custom"]]
    main(Config(home=home, out=tmp_path / "session", session_id="c"))
    assert set(load_manifest(tmp_path / "session/claude/manifest.json").sessions) == {"c"}
    os.utime(home / "projects/one/a.jsonl", (0, 0))
    main(Config(home=home, out=tmp_path / "date", since="2026-01-01"))
    assert set(load_manifest(tmp_path / "date/claude/manifest.json").sessions) == {"b", "c"}
    assert "failed=0" in capsys.readouterr().out


def test_bad_session_keeps_good_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Parser errors are reported while completed entries stay on disk."""
    from agent_traces.apis.convert_all import Config, load_manifest, main

    home: Path = tmp_path / ".claude"
    good: Path = home / "projects/one/a.jsonl"
    SessionBuilder(good).add("user", message={"content": "good"})
    bad: Path = good.with_name("b.jsonl")
    bad.write_text("{broken\n")
    SessionBuilder(good.with_name("c.jsonl")).add("user", message={"content": "also good"})
    main(Config(home=home, out=tmp_path / "out"))
    output: str = capsys.readouterr().out
    assert f"FAILED {bad}:" in output
    assert "converted=2 skipped=0 failed=1" in output
    assert set(load_manifest(tmp_path / "out/claude/manifest.json").sessions) == {"a", "c"}


@pytest.mark.parametrize(
    "content",
    [
        "{broken",
        '{"version":1,"sessions":{},"unknown":true}',
        '{"sessions":{"a":{"source_path":"a","source_sha256":"b","rrd":"a.rrd","converted_at":"now","n_rows":1,"unknown":true}}}',
    ],
)
def test_corrupt_manifest_names_path(tmp_path: Path, content: str) -> None:
    """Both malformed JSON and unknown fields fail at the owned-file boundary."""
    from agent_traces.apis.convert_all import Config, main

    manifest: Path = tmp_path / "out/claude/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(content)
    with pytest.raises(ValueError, match=str(manifest)):
        main(Config(home=tmp_path / ".claude", out=tmp_path / "out"))


def test_writer_failure_propagates_after_saving_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A writer error is not treated as a parser error; prior progress survives."""
    from agent_traces.apis import convert_all
    from agent_traces.claude import ClaudeSession
    from agent_traces.rerun_log import write_session_rrd

    home: Path = tmp_path / ".claude"
    for session_id in ["a", "b"]:
        SessionBuilder(home / "projects/one" / f"{session_id}.jsonl").add("user", message={"content": session_id})

    def fail_second(session: ClaudeSession, out: Path) -> Path:
        """Simulate a recording boundary failure after the first saved session."""
        if session.session_id == "b":
            raise ValueError("writer failure")
        return write_session_rrd(session, out)

    monkeypatch.setattr(convert_all, "write_session_rrd", fail_second)
    with pytest.raises(ValueError, match="writer failure"):
        convert_all.main(convert_all.Config(home=home, out=tmp_path / "out"))
    assert set(convert_all.load_manifest(tmp_path / "out/claude/manifest.json").sessions) == {"a"}
