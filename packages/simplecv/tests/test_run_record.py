"""A layer's run record: what ran, from which code, where and for how long, read back from the catalog."""

import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

rr = pytest.importorskip("rerun", reason="rerun-sdk is required by this test module")
pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")

from simplecv.catalog_layers import atomic_recording, ensure_dataset, register_layer  # noqa: E402
from simplecv.run_record import RunRecord, capture_run_record, read_run_record, write_run_record  # noqa: E402


@dataclass(frozen=True, slots=True)
class _Config:
    segment: str
    stride: int = 10


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo: Path = tmp_path / "repo"
    (repo / "packages" / "stage").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "packages" / "stage" / "code.py").write_text("x = 1\n")
    (repo / "pixi.lock").write_text("lock\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return repo


def test_each_layer_keeps_its_own_record_on_the_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A segment's layers merge their property columns, so each record is its own group: both read back whole."""
    monkeypatch.setattr(sys, "argv", ["tools/apps/stage.py", "--segment", "seg", "--note", "two words"])
    started: datetime = datetime.now(UTC) - timedelta(seconds=2)
    records: dict[str, RunRecord] = {
        layer: capture_run_record(layer, "seg", config=_Config(segment="seg"), started=started) for layer in ("base", "masks")
    }
    paths: list[Path] = []
    for layer, record in records.items():
        path: Path = tmp_path / f"{layer}.rrd"
        with atomic_recording(path, application_id="simplecv-test", recording_id="seg", send_properties=layer == "base") as recording:
            write_run_record(recording, record)
            recording.log(f"/{layer}", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
        paths.append(path)
    with rr.server.Server() as server:
        entry = ensure_dataset(server.client(), "records")
        register_layer(entry, paths[:1], None)
        register_layer(entry, paths[1:], "masks")
        read: dict[str, RunRecord] = {layer: read_run_record(entry, "seg", layer) for layer in records}
        with pytest.raises(ValueError, match="no run record for layer 'cube'"):
            read_run_record(entry, "seg", "cube")
    assert read == records
    record: RunRecord = records["masks"]
    assert record.command == "python tools/apps/stage.py --segment seg --note 'two words'"
    assert record.config == "_Config(segment='seg', stride=10)"
    assert record.wall_s >= 2.0 and record.started_at == started.isoformat()
    assert len(record.git_commit) == 40 and record.peak_rss_mb > 0.0


def test_the_record_names_the_code_by_commit_and_a_hash_of_what_is_not_committed(tmp_path: Path) -> None:
    repo: Path = _repo(tmp_path)
    stage: Path = repo / "packages" / "stage"
    started: datetime = datetime.now(UTC)

    def capture() -> RunRecord:
        return capture_run_record("layer", "seg", config=None, started=started, cwd=stage)

    clean: RunRecord = capture()
    assert clean.git_commit == _git(repo, "rev-parse", "HEAD") and clean.git_dirty_sha256 == ""
    assert clean.cwd == "packages/stage" and len(clean.pixi_lock_sha256) == 64
    (stage / "code.py").write_text("x = 2\n")
    edited: str = capture().git_dirty_sha256
    (stage / "new.py").write_text("y = 1\n")
    added: str = capture().git_dirty_sha256
    (stage / "new.py").write_text("y = 2\n")
    assert len({edited, added, capture().git_dirty_sha256}) == 3 and edited != ""
