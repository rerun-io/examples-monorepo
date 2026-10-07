"""Batch conversion through its public entrypoint and persisted outputs."""

import subprocess
import sys
from pathlib import Path

import pytest

from agent_traces import manifest, writing
from agent_traces.apis import convert, convert_all
from agent_traces.apis.convert_all import Config as ConvertAllConfig
from agent_traces.apis.convert_all import main as convert_all_main
from agent_traces.events import Session
from agent_traces.manifest import Manifest, load_manifest, save_manifest
from agent_traces.rerun_log import WrittenRecording, write_session_rrd
from agent_traces.sources import fingerprint
from tests.conftest import RolloutBuilder, SessionBuilder, agent_rows, read_entities


def test_batch_resumes_and_hashes_subagents(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Three sessions convert once; changed children and missing outputs rebuild."""
    home: Path = tmp_path / ".claude"
    for project, session_id in [("one", "a"), ("one", "b"), ("two", "c")]:
        SessionBuilder(home / "projects" / project / f"{session_id}.jsonl").add("user", message={"content": session_id})
    child: SessionBuilder = SessionBuilder(home / "projects/one/a/subagents/agent-child.jsonl")
    child.add("user", message={"content": "child"})
    SessionBuilder(home / "projects/one/agent-stray.jsonl").add("user", message={"content": "stray"})
    config: ConvertAllConfig = ConvertAllConfig(home=home, out=tmp_path / "out")
    convert_all_main(config)
    assert "converted=3 skipped=0 failed=0" in capsys.readouterr().out
    manifest_path: Path = tmp_path / "out/claude/manifest.json"
    manifest: Manifest = load_manifest(manifest_path)
    assert set(manifest.sessions) == {"a", "b", "c"}
    assert {session_id: entry.n_rows for session_id, entry in manifest.sessions.items()} == {"a": 7, "b": 6, "c": 6}
    assert read_entities(tmp_path / "out/claude/a.rrd")["/__properties/session"]["profile"].to_pylist() == [["claude"]]
    convert_all_main(config)
    assert "converted=0 skipped=3 failed=0" in capsys.readouterr().out
    child.add("assistant", message={"content": "new"})
    convert_all_main(config)
    assert "converted=1 skipped=2 failed=0" in capsys.readouterr().out
    assert load_manifest(manifest_path).sessions["a"].n_rows == 8
    (tmp_path / "out/claude/b.rrd").unlink()
    convert_all_main(config)
    assert "converted=1 skipped=2 failed=0" in capsys.readouterr().out

    late: SessionBuilder = SessionBuilder(home / "projects/one/a/subagents/agent-late.jsonl")
    late.add("user", message={"content": "late child"})
    convert_all_main(config)
    assert "converted=1 skipped=2 failed=0" in capsys.readouterr().out
    assert load_manifest(manifest_path).sessions["a"].n_rows == 9


def test_batch_filters_and_profile(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Project, session, and UTC modification-date filters select main files."""
    import os


    home: Path = tmp_path / ".claude"
    for project, session_id in [("one", "a"), ("one", "b"), ("two", "c")]:
        SessionBuilder(home / "projects" / project / f"{session_id}.jsonl").add("user", message={"content": session_id})
    convert_all_main(ConvertAllConfig(home=home, out=tmp_path / "project", project="on", profile="custom"))
    assert set(load_manifest(tmp_path / "project/custom/manifest.json").sessions) == {"a", "b"}
    assert read_entities(tmp_path / "project/custom/a.rrd")["/__properties/session"]["profile"].to_pylist() == [["custom"]]
    convert_all_main(ConvertAllConfig(home=home, out=tmp_path / "session", session_id="c"))
    assert set(load_manifest(tmp_path / "session/claude/manifest.json").sessions) == {"c"}
    os.utime(home / "projects/one/a.jsonl", (0, 0))
    convert_all_main(ConvertAllConfig(home=home, out=tmp_path / "date", since="2026-01-01"))
    assert set(load_manifest(tmp_path / "date/claude/manifest.json").sessions) == {"b", "c"}
    assert "failed=0" in capsys.readouterr().out


def test_bad_session_keeps_good_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Parser errors are reported while completed entries stay on disk."""
    home: Path = tmp_path / ".claude"
    good: Path = home / "projects/one/a.jsonl"
    SessionBuilder(good).add("user", message={"content": "good"})
    bad: Path = good.with_name("b.jsonl")
    bad.write_text('{"type":"user","message":{"content":123}}\n')
    SessionBuilder(good.with_name("c.jsonl")).add("user", message={"content": "also good"})
    convert_all_main(ConvertAllConfig(home=home, out=tmp_path / "out"))
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
    (tmp_path / ".claude/projects").mkdir(parents=True)
    manifest: Path = tmp_path / "out/claude/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(content)
    with pytest.raises(ValueError, match=str(manifest)):
        convert_all_main(ConvertAllConfig(home=tmp_path / ".claude", out=tmp_path / "out"))


def test_writer_failure_is_reported_after_saving_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A writer error fails the batch summary; prior progress survives."""
    home: Path = tmp_path / ".claude"
    for session_id in ["a", "b"]:
        SessionBuilder(home / "projects/one" / f"{session_id}.jsonl").add("user", message={"content": session_id})

    def fail_second(session: Session, out: Path, *, host: str | None = None) -> WrittenRecording:
        """Simulate a recording boundary failure after the first saved session."""
        if session.session_id == "b":
            raise ValueError("writer failure")
        return write_session_rrd(session, out, host=host)

    monkeypatch.setattr(convert_all, "write_session_rrd", fail_second)
    summary = convert_all.main(convert_all.Config(home=home, out=tmp_path / "out"))
    assert summary.failed == 1
    assert summary.exit_code == 1
    assert set(load_manifest(tmp_path / "out/claude/manifest.json").sessions) == {"a"}


@pytest.mark.parametrize("change", ["rename_child", "edit_output", "add_output", "edit_page", "move_record"])
def test_batch_fingerprints_all_session_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str], change: str) -> None:
    """Input paths, file boundaries, and recursive offloaded files affect resume."""
    home: Path = tmp_path / ".claude"
    session: SessionBuilder = SessionBuilder(home / "projects/one/a.jsonl")
    session.add("user", message={"content": "prompt"})
    child: Path = session.path.with_suffix("") / "subagents/agent-child.jsonl"
    SessionBuilder(child).add("assistant", message={"content": "child"})
    outputs: Path = session.path.with_suffix("") / "tool-results"
    (outputs / "pdf-id").mkdir(parents=True)
    (outputs / "x.txt").write_text("before")
    (outputs / "pdf-id/page.jpg").write_bytes(b"before")
    config: ConvertAllConfig = ConvertAllConfig(home=home, out=tmp_path / "out")
    convert_all_main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    convert_all_main(config)
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
    convert_all_main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    convert_all_main(config)
    assert "converted=0 skipped=1 failed=0" in capsys.readouterr().out


def test_missing_recording_retries_completed_entry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A matching manifest cannot skip a recording that no longer exists."""
    home: Path = tmp_path / ".claude"
    SessionBuilder(home / "projects/one/a.jsonl").add("user", message={"content": "prompt"})
    config: ConvertAllConfig = ConvertAllConfig(home=home, out=tmp_path / "out")
    convert_all_main(config)
    capsys.readouterr()
    manifest_path: Path = tmp_path / "out/claude/manifest.json"
    recording: Path = manifest_path.parent / load_manifest(manifest_path).sessions["a"].rrd
    recording.unlink()
    convert_all_main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    assert read_entities(recording)["/turns"]["TextLog:text"].to_pylist() == [["prompt"]]


def test_appledouble_sidecars_are_not_sessions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A macOS copy leaves `._<session>.jsonl` resource forks beside real transcripts; they are neither converted nor failures."""
    home: Path = tmp_path / ".claude"
    builder: SessionBuilder = SessionBuilder(home / "projects" / "p" / "real.jsonl")
    builder.add("user", message={"content": "hello"})
    (home / "projects" / "p" / "._real.jsonl").write_bytes(b"\x00\x05\x16\x07 not json")

    convert_all_main(ConvertAllConfig(home=home, out=tmp_path / "out"))
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out


@pytest.mark.parametrize("version", [1, 999])
def test_unsupported_manifest_version_requires_reconversion(tmp_path: Path, version: int) -> None:
    """Old or unknown schemas require an explicit reset, including nonempty v1 files."""
    path: Path = tmp_path / "manifest.json"
    path.write_text(f'{{"version":{version},"sessions":{{"old":{{}}}}}}')
    with pytest.raises(ValueError, match=rf"{path}.*delete.*convert everything again"):
        load_manifest(path)


def test_symlink_inventory_and_conversion_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """A renamed link keeps target identity, child dependencies, and cache settings."""
    from dataclasses import replace


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
    """Single-file detection is independent of the home layout policy."""
    home = tmp_path / ".codex"
    path = home / "sessions/a.jsonl"
    SessionBuilder(path).add("user", message={"content": "hello"})
    convert.main(convert.Config(session=path, out=tmp_path / "single.rrd"))
    convert_all.main(convert_all.Config(home=home, out=tmp_path / "batch"))
    single = read_entities(tmp_path / "single.rrd")["/__properties/session"]
    assert single["agent"].to_pylist() == [["claude"]]
    assert not list((tmp_path / "batch").rglob("*.rrd"))


@pytest.mark.parametrize("target", ["main", "child", "image"])
def test_changes_during_conversion_cannot_certify_newer_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], png_bytes: bytes, target: str,
) -> None:
    """A change after an input read must cause a later run to rebuild the RRD."""
    home = tmp_path / ".codex"
    parent = RolloutBuilder(home / "sessions/main.jsonl")
    parent.meta("main")
    parent.item("AgentMessage", content=[{"type": "text", "text": "before"}])
    child = RolloutBuilder(home / "sessions/child.jsonl")
    child.meta("child", parent_thread_id="main")
    child.item("AgentMessage", content=[{"type": "text", "text": "before"}])
    image = tmp_path / "local.png"
    image.write_bytes(png_bytes)
    parent.add("event_msg", type="user_message", local_images=[str(image)])
    config = convert_all.Config(home=home, out=tmp_path / "out")

    def change_after_read(session: Session, out: Path, *, host: str | None = None) -> WrittenRecording:
        if target == "image":
            image.write_bytes(png_bytes + b"after")
        else:
            builder = parent if target == "main" else child
            builder.item("AgentMessage", content=[{"type": "text", "text": "after"}])
        return write_session_rrd(session, out, host=host)

    with monkeypatch.context() as patch:
        patch.setattr(convert_all, "write_session_rrd", change_after_read)
        convert_all.main(config)
    capsys.readouterr()
    convert_all.main(config)
    assert "converted=1 skipped=1 failed=0" in capsys.readouterr().out
    entities = read_entities(tmp_path / "out/codex/main.rrd")
    if target != "image":
        identity = "child" if target == "child" else ""
        assert agent_rows(entities["/conversation/assistant"], identity)["TextLog:text"].to_pylist() == ([["[a0] before"], ["[a0] after"]] if identity else [["before"], ["after"]])
    else:
        assert entities["/media/images"]["EncodedImage:blob"].to_pylist() == [[list(png_bytes + b"after")]]
    convert_all.main(config)
    assert "converted=0 skipped=2 failed=0" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["main", "child"])
@pytest.mark.parametrize("single", [False, True])
def test_transcript_append_during_parse_keeps_preparse_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str, single: bool,
) -> None:
    """An append at the decode boundary must not advance the certified input hash."""
    home = tmp_path / ".codex"
    parent = RolloutBuilder(home / "sessions/main.jsonl")
    parent.meta("main")
    parent.item("AgentMessage", content=[{"type": "text", "text": "main-before"}])
    child = RolloutBuilder(home / "sessions/child.jsonl")
    child.meta("child", parent_thread_id="main")
    child.item("AgentMessage", content=[{"type": "text", "text": "child-before"}])
    inputs = (parent.path, child.path)
    expected = fingerprint(inputs)
    builder = parent if target == "main" else child
    import orjson

    original = orjson.loads
    changed = False

    def append_during_decode(data: bytes) -> object:
        nonlocal changed
        decoded = original(data)
        if not changed and (target + "-before").encode() in data:
            changed = True
            builder.item("AgentMessage", content=[{"type": "text", "text": "after"}])
        return decoded

    config = convert_all.Config(home=home, out=tmp_path / "out")
    with monkeypatch.context() as patch:
        patch.setattr(orjson, "loads", append_during_decode)
        if single:
            convert.main(convert.Config(session=parent.path, out=tmp_path / "single.rrd"))
        else:
            convert_all.main(config)
    saved = tmp_path / "single.rrd" if single else tmp_path / "out/codex/main.rrd"
    props = read_entities(saved)["/__properties/session"]
    assert changed
    assert props["source_sha256"].to_pylist() == [[expected]]
    assert fingerprint(inputs) != expected
    if not single:
        assert load_manifest(saved.parent / "manifest.json").sessions["main"].source_sha256 == expected
        convert_all.main(config)
        assert load_manifest(saved.parent / "manifest.json").sessions["main"].source_sha256 == fingerprint(inputs)
        identity = "child" if target == "child" else ""
        assert agent_rows(read_entities(saved)["/conversation/assistant"], identity)["TextLog:text"].to_pylist()[-1] == (["[a0] after"] if identity else ["after"])


def test_image_replacement_at_read_boundary_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], png_bytes: bytes,
) -> None:
    """Newly discovered images are hashed from consumed bytes, never a later read."""
    home = tmp_path / ".codex"
    parent = RolloutBuilder(home / "sessions/main.jsonl")
    parent.meta("main")
    parent.item("Reasoning")
    image = tmp_path / "image.png"
    image.write_bytes(png_bytes)
    parent.add("event_msg", type="user_message", local_images=[str(image)])
    config = ConvertAllConfig(home=home, out=tmp_path / "out")
    original = Path.read_bytes

    def read_then_replace(path: Path) -> bytes:
        data = original(path)
        if path == image:
            path.write_bytes(png_bytes + b"newer")
        return data

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read_then_replace)
        convert_all_main(config)
    capsys.readouterr()
    saved = tmp_path / "out/codex/main.rrd"
    assert load_manifest(saved.parent / "manifest.json").sessions["main"].extra_inputs == (str(image),)
    assert read_entities(saved)["/media/images"]["EncodedImage:blob"].to_pylist() == [[list(png_bytes)]]
    convert_all_main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    assert read_entities(saved)["/media/images"]["EncodedImage:blob"].to_pylist() == [[list(png_bytes + b"newer")]]
    convert_all_main(config)
    assert "converted=0 skipped=1 failed=0" in capsys.readouterr().out


@pytest.mark.parametrize("first_target", ["first.png", "absent.png"])
def test_image_symlink_retarget_rebuilds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], png_bytes: bytes, first_target: str,
) -> None:
    """An image referenced through a symlink is re-read through that link, so retargeting it rebuilds."""
    home = tmp_path / ".codex"
    parent = RolloutBuilder(home / "sessions/main.jsonl")
    parent.meta("main")
    parent.item("Reasoning")
    (tmp_path / "first.png").write_bytes(png_bytes)
    (tmp_path / "second.png").write_bytes(png_bytes + b"second")
    link = tmp_path / "current.png"
    link.symlink_to(tmp_path / first_target)
    parent.add("event_msg", type="user_message", local_images=[str(link)])
    config = ConvertAllConfig(home=home, out=tmp_path / "out")
    convert_all_main(config)
    capsys.readouterr()
    link.unlink()
    link.symlink_to(tmp_path / "second.png")
    convert_all_main(config)
    assert "converted=1 skipped=0 failed=0" in capsys.readouterr().out
    saved = tmp_path / "out/codex/main.rrd"
    assert read_entities(saved)["/media/images"]["EncodedImage:blob"].to_pylist() == [[list(png_bytes + b"second")]]
    convert_all_main(config)
    assert "converted=0 skipped=1 failed=0" in capsys.readouterr().out


def test_concurrent_batches_wait_and_keep_both_entries(tmp_path: Path) -> None:
    """A writer holds the manifest transaction while another process waits."""
    import subprocess
    import sys
    import time


    for name in ("first", "second"):
        SessionBuilder(tmp_path / name / "projects/project" / f"{name}.jsonl").add("user", message={"content": name})
    worker: str = '''
import sys, time
from pathlib import Path
from agent_traces.apis import convert_all
root = Path(sys.argv[1])
name = sys.argv[2]
original = convert_all.write_session_rrd
if name == "first":
    def write(session, out, *, host=None):
        (root / "entered").touch()
        deadline = time.monotonic() + 10
        while not (root / "release").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("test release timed out")
            time.sleep(0.01)
        return original(session, out, host=host)
    convert_all.write_session_rrd = write
summary = convert_all.main(convert_all.Config(home=root/name, out=root/"out", profile="shared"))
raise SystemExit(summary.exit_code)
'''
    first = subprocess.Popen([sys.executable, "-c", worker, str(tmp_path), "first"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    second: subprocess.Popen[bytes] | None = None
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "entered").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "entered").exists()
        with (tmp_path / "second.log").open("wb") as output:
            second = subprocess.Popen([sys.executable, "-c", worker, str(tmp_path), "second"], stdout=output, stderr=subprocess.PIPE)
            deadline = time.monotonic() + 5
            while b"Waiting for manifest lock" not in (tmp_path / "second.log").read_bytes() and second.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert b"Waiting for manifest lock" in (tmp_path / "second.log").read_bytes()
            assert second.poll() is None
    finally:
        (tmp_path / "release").touch()
        first.communicate(timeout=15)
        if second is not None:
            second.communicate(timeout=15)
    assert first.returncode == 0
    assert second is not None and second.returncode == 0
    assert set(load_manifest(tmp_path / "out/shared/manifest.json").sessions) == {"first", "second"}


def test_failed_session_returns_failure_status(tmp_path: Path) -> None:
    """Successful entries persist even when the batch process exits one."""
    home: Path = tmp_path / ".claude"
    builder = SessionBuilder(home / "projects/project/good.jsonl")
    builder.add("user", message={"content": "good"})
    builder.path.with_name("bad.jsonl").write_text('{"type":"user","message":{"content":123}}\n')
    summary = convert_all.main(convert_all.Config(home=home, out=tmp_path / "api"))
    assert summary.failed == 1
    assert summary.converted == 1
    assert summary.exit_code == 1
    command: Path = Path(__file__).parents[1] / "tools/apps/convert_all.py"
    result = subprocess.run([sys.executable, str(command), "--home", str(home), "--out", str(tmp_path / "cli")], capture_output=True, text=True)
    assert result.returncode == 1
    assert "converted=1 skipped=0 failed=1" in result.stdout


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("single", [False, True])
def test_invalid_home_reports_missing_layout(tmp_path: Path, exists: bool, single: bool, capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid inputs fail through the summary and CLI without leaving output debris."""
    home: Path = tmp_path / "unknown"
    out: Path = tmp_path / "output"
    if exists:
        home.mkdir()
    if single:
        summary = convert.main(convert.Config(session=home, out=out / "session.rrd"))
    else:
        summary = convert_all_main(ConvertAllConfig(home=home, out=out))
    assert summary.failed == 1
    assert summary.exit_code == 1
    assert f"FAILED {home}:" in capsys.readouterr().out
    assert not out.exists()
    command: Path = Path(__file__).parents[1] / "tools/apps" / ("convert.py" if single else "convert_all.py")
    result = subprocess.run([sys.executable, str(command), "--session" if single else "--home", str(home), "--out", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert f"FAILED {home}:" in result.stdout
    assert "Traceback" not in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("layout", ["missing", "unrelated", "empty"])
def test_claude_discovery_requires_projects(tmp_path: Path, layout: str) -> None:
    """Claude discovery rejects missing layouts but accepts an empty projects directory."""
    from agent_traces.claude import discover

    home: Path = tmp_path / ".claude"
    if layout == "empty":
        (home / "projects").mkdir(parents=True)
        assert discover(home).sessions == []
        summary = convert_all_main(ConvertAllConfig(home=home, out=tmp_path / "output"))
        assert summary.failed == 0
        assert summary.converted == 0
    else:
        if layout == "unrelated":
            home.mkdir()
        with pytest.raises(ValueError, match="projects"):
            discover(home)


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_batch_rejects_transcript_as_home(tmp_path: Path, provider: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Both provider files fail batch conversion through the API and CLI."""
    path: Path = tmp_path / "transcript.jsonl"
    if provider == "claude":
        SessionBuilder(path).add("user", message={"content": "hello"})
    else:
        RolloutBuilder(path).meta()
    out: Path = tmp_path / "out"
    summary = convert_all_main(ConvertAllConfig(home=path, out=out))
    assert summary.failed == summary.exit_code == 1
    assert f"FAILED {path}:" in capsys.readouterr().out
    assert not out.exists()
    command: Path = Path(__file__).parents[1] / "tools/apps/convert_all.py"
    result = subprocess.run([sys.executable, str(command), "--home", str(path), "--out", str(out)], capture_output=True, text=True)
    assert result.returncode == 1
    assert f"FAILED {path}:" in result.stdout
    assert not out.exists()
