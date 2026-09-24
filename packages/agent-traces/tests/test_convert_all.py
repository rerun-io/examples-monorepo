"""Batch conversion through its public entrypoint and persisted outputs."""

from pathlib import Path

import pytest

from tests.conftest import SessionBuilder
from tests.test_rerun_log import read_entities


def test_batch_resumes_and_hashes_subagents(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Three sessions convert once; changed children and missing outputs rebuild."""
    from agent_traces.apis.convert_all import Config, main
    from agent_traces.manifest import Manifest, load_manifest

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

    from agent_traces.apis.convert_all import Config, main
    from agent_traces.manifest import load_manifest

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
    from agent_traces.apis.convert_all import Config, main
    from agent_traces.manifest import load_manifest

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
        '{"version":2,"sessions":{},"unknown":true}',
        '{"version":2,"sessions":{"a":{"source_path":"a","source_sha256":"b","rrd":"a.rrd","converted_at":"now","host":"h","revision":1,"n_rows":1,"unknown":true}}}',
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
    from agent_traces.events import Session
    from agent_traces.manifest import load_manifest
    from agent_traces.rerun_log import WrittenRecording, write_session_rrd

    home: Path = tmp_path / ".claude"
    for session_id in ["a", "b"]:
        SessionBuilder(home / "projects/one" / f"{session_id}.jsonl").add("user", message={"content": session_id})

    def fail_second(session: Session, out: Path, *, host: str | None = None) -> WrittenRecording:
        """Simulate a recording boundary failure after the first saved session."""
        if session.session_id == "b":
            raise ValueError("writer failure")
        return write_session_rrd(session, out, host=host)

    monkeypatch.setattr(convert_all, "write_session_rrd", fail_second)
    with pytest.raises(ValueError, match="writer failure"):
        convert_all.main(convert_all.Config(home=home, out=tmp_path / "out"))
    assert set(load_manifest(tmp_path / "out/claude/manifest.json").sessions) == {"a"}


@pytest.mark.parametrize("change", ["rename_child", "edit_output", "add_output", "edit_page", "move_record"])
def test_batch_fingerprints_all_session_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str], change: str) -> None:
    """Input paths, file boundaries, and recursive offloaded files affect resume."""
    from agent_traces.apis.convert_all import Config, main

    home: Path = tmp_path / ".claude"
    session: SessionBuilder = SessionBuilder(home / "projects/one/a.jsonl")
    session.add("user", message={"content": "prompt"})
    child: Path = session.path.with_suffix("") / "subagents/agent-child.jsonl"
    SessionBuilder(child).add("assistant", message={"content": "child"})
    outputs: Path = session.path.with_suffix("") / "tool-results"
    (outputs / "pdf-id").mkdir(parents=True)
    (outputs / "x.txt").write_text("before")
    (outputs / "pdf-id/page.jpg").write_bytes(b"before")
    config: Config = Config(home=home, out=tmp_path / "out")
    main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    main(config)
    assert "converted=0 skipped=1 failed=0" in capsys.readouterr().out
    match change:
        case "rename_child":
            child.rename(child.with_name("agent-renamed.jsonl"))
        case "edit_output":
            (outputs / "x.txt").write_text("after")
        case "add_output":
            (outputs / "new.txt").write_text("new")
        case "edit_page":
            (outputs / "pdf-id/page.jpg").write_bytes(b"after")
        case "move_record":
            session.path.write_bytes(session.path.read_bytes() + child.read_bytes())
            child.write_bytes(b"")
    main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    main(config)
    assert "converted=0 skipped=1 failed=0" in capsys.readouterr().out


def test_missing_recording_retries_completed_entry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A matching manifest cannot skip a recording that no longer exists."""
    from agent_traces.apis.convert_all import Config, main
    from agent_traces.manifest import load_manifest

    home: Path = tmp_path / ".claude"
    SessionBuilder(home / "projects/one/a.jsonl").add("user", message={"content": "prompt"})
    config: Config = Config(home=home, out=tmp_path / "out")
    main(config)
    capsys.readouterr()
    manifest_path: Path = tmp_path / "out/claude/manifest.json"
    recording: Path = manifest_path.parent / load_manifest(manifest_path).sessions["a"].rrd
    recording.unlink()
    main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    assert read_entities(recording)["/turns"]["TextLog:text"].to_pylist() == [["prompt"]]


def test_appledouble_sidecars_are_not_sessions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A macOS copy leaves `._<session>.jsonl` resource forks beside real transcripts; they are neither converted nor failures."""
    home: Path = tmp_path / ".claude"
    builder: SessionBuilder = SessionBuilder(home / "projects" / "p" / "real.jsonl")
    builder.add("user", message={"content": "hello"})
    (home / "projects" / "p" / "._real.jsonl").write_bytes(b"\x00\x05\x16\x07 not json")
    from agent_traces.apis.convert_all import Config, main

    main(Config(home=home, out=tmp_path / "out"))
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out


@pytest.mark.parametrize("version", [1, 999])
def test_unsupported_manifest_version_requires_reconversion(tmp_path: Path, version: int) -> None:
    """Old or unknown schemas require an explicit reset, including nonempty v1 files."""
    from agent_traces.manifest import load_manifest

    path: Path = tmp_path / "manifest.json"
    path.write_text(f'{{"version":{version},"sessions":{{"old":{{}}}}}}')
    with pytest.raises(ValueError, match=rf"{path}.*delete.*convert everything again"):
        load_manifest(path)


def test_symlink_inventory_and_conversion_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """A renamed link keeps target identity, child dependencies, and cache settings."""
    from dataclasses import replace

    from agent_traces import manifest
    from agent_traces.apis import convert, convert_all

    target: Path = tmp_path / "original/projects/p/real.jsonl"
    SessionBuilder(target).add("user", message={"content": "main"})
    child = SessionBuilder(target.with_suffix("") / "subagents/agent-child.jsonl")
    child.add("user", message={"content": "child"})
    home: Path = tmp_path / ".claude"
    link: Path = home / "projects/p/alias.jsonl"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    config = convert_all.Config(home=home, out=tmp_path / "out", host="one")
    convert_all.main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    saved = manifest.load_manifest(tmp_path / "out/claude/manifest.json")
    assert set(saved.sessions) == {"real"}
    props = read_entities(tmp_path / "out/claude/real.rrd")["/__properties/session"]
    assert props["session_id"].to_pylist() == [["real"]]
    from rerun.chunk import RrdReader

    assert RrdReader(tmp_path / "out/claude/real.rrd").recordings()[0].recording_id == "real"
    assert props["source_sha256"].to_pylist() == [[saved.sessions["real"].source_sha256]]
    convert.main(convert.Config(session=link, out=tmp_path / "single.rrd"))
    assert read_entities(tmp_path / "single.rrd")["/__properties/session"]["source_sha256"].to_pylist() == props["source_sha256"].to_pylist()
    capsys.readouterr()
    convert_all.main(config)
    assert "converted=0 skipped=1 failed=0" in capsys.readouterr().out
    child.add("assistant", message={"content": "changed"})
    convert_all.main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    config = replace(config, host="other")
    convert_all.main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    monkeypatch.setattr(manifest, "CONVERSION_REVISION", manifest.CONVERSION_REVISION + 1)
    convert_all.main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    convert_all.main(config)
    assert "converted=0 skipped=1 failed=0" in capsys.readouterr().out


def test_manifest_failure_preserves_published_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Publication failure preserves the old manifest and removes the temporary file."""
    from agent_traces import writing
    from agent_traces.manifest import Manifest, save_manifest

    path = tmp_path / "manifest.json"
    save_manifest(Manifest(), path)
    original = path.read_bytes()

    def fail_replace(source: Path, target: Path) -> None:
        assert source.read_bytes()
        assert target == path
        raise OSError("replace failed")

    monkeypatch.setattr(writing.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        save_manifest(Manifest(), path)
    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


def test_claude_transcript_in_codex_home(tmp_path: Path) -> None:
    """Home layout chooses search paths; first-record detection chooses each parser."""
    from agent_traces.apis import convert, convert_all

    home = tmp_path / ".codex"
    path = home / "sessions/a.jsonl"
    SessionBuilder(path).add("user", message={"content": "hello"})
    convert.main(convert.Config(session=path, out=tmp_path / "single.rrd"))
    convert_all.main(convert_all.Config(home=home, out=tmp_path / "batch"))
    single = read_entities(tmp_path / "single.rrd")["/__properties/session"]
    batch = read_entities(tmp_path / "batch/codex/a.rrd")["/__properties/session"]
    for name in ("agent", "session_id", "source_sha256"):
        assert single[name].to_pylist() == batch[name].to_pylist()
