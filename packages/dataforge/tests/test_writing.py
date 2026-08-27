from pathlib import Path

import pytest
import rerun as rr

from dataforge.writing import atomic_recording, should_skip


def test_should_skip_existing_unless_forced(tmp_path: Path) -> None:
    target: Path = tmp_path / "base" / "x.rrd"
    assert not should_skip(target, force=False)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"done")
    assert should_skip(target, force=False)
    assert not should_skip(target, force=True)


def test_atomic_recording_publishes_only_on_success(tmp_path: Path) -> None:
    target: Path = tmp_path / "robocap__a__b.rrd"
    with atomic_recording(target, application_id="dataforge", recording_id="robocap__a__b") as recording:
        recording.log("/world", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    assert target.exists()
    assert target.stat().st_size > 0
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_recording_leaves_no_target_or_tmp_on_failure(tmp_path: Path) -> None:
    target: Path = tmp_path / "robocap__a__c.rrd"
    with pytest.raises(RuntimeError, match="boom"), atomic_recording(target, application_id="dataforge", recording_id="robocap__a__c"):
        raise RuntimeError("boom")
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_recording_creates_parent_dirs(tmp_path: Path) -> None:
    target: Path = tmp_path / "rrd" / "base" / "robocap__a__d.rrd"
    with atomic_recording(target, application_id="dataforge", recording_id="robocap__a__d") as recording:
        recording.log("/world", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    assert target.exists()
