from pathlib import Path

import pytest

from selfcap_soma.batch import Checkpoint, Config, Timings, cleanup_published

SEGMENT = "selfcap__2025-12-03__d31bf1da__ep001"


def test_cleanup_requires_verified_publication_and_retains_reports(tmp_path: Path) -> None:
    config = Config(dataset="test", spark_dir=Path("/remote/soma"), output=tmp_path, cleanup_after_publish=True)
    directory = tmp_path / "test" / SEGMENT
    directory.mkdir(parents=True)
    files = ["fit.rrd", SEGMENT + ".rrd", SEGMENT + ".pending.rrd", "report.json", "command.log", "evidence.rrd"]
    for name in files:
        (directory / name).write_text("keep or delete")
    state = Checkpoint("signature", "published", Timings())
    with pytest.raises(ValueError, match="verified"):
        cleanup_published(config, directory, SEGMENT, state)
    assert all((directory / name).exists() for name in files)
    state.readback_verified = True
    cleanup_published(config, directory, SEGMENT, state)
    assert state.cleaned
    assert not (directory / "fit.rrd").exists()
    assert not (directory / (SEGMENT + ".rrd")).exists()
    assert not (directory / (SEGMENT + ".pending.rrd")).exists()
    assert all((directory / name).exists() for name in ["report.json", "command.log", "evidence.rrd", "checkpoint.json"])


def test_cleanup_cannot_delete_through_a_symlink(tmp_path: Path) -> None:
    config = Config(dataset="test", spark_dir=Path("/remote/soma"), output=tmp_path, cleanup_after_publish=True)
    directory = tmp_path / "test" / SEGMENT
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.rrd"
    outside.write_text("outside")
    (directory / "fit.rrd").symlink_to(outside)
    state = Checkpoint("signature", "published", Timings(), readback_verified=True)
    with pytest.raises(ValueError, match="symlink"):
        cleanup_published(config, directory, SEGMENT, state)
    assert outside.read_text() == "outside"


def test_cleanup_never_runs_for_unpublished_or_read_only_calls(tmp_path: Path) -> None:
    from dataclasses import replace

    config = Config(dataset="test", spark_dir=Path("/remote/soma"), output=tmp_path, cleanup_after_publish=True)
    directory = tmp_path / "test" / SEGMENT
    directory.mkdir(parents=True)
    artifact = directory / "fit.rrd"
    artifact.write_text("retain")
    state = Checkpoint("signature", "export", Timings(), readback_verified=True)
    with pytest.raises(ValueError, match="verified successful publication"):
        cleanup_published(config, directory, SEGMENT, state)
    state.phase = "published"
    with pytest.raises(ValueError, match="verified successful publication"):
        cleanup_published(replace(config, publish=False), directory, SEGMENT, state)
    assert artifact.read_text() == "retain"


def test_failed_readback_clears_old_verification_and_prevents_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    import numpy as np

    from selfcap_soma.batch import verify_publication

    config = Config(dataset="test", spark_dir=Path("/remote/soma"), output=tmp_path, cleanup_after_publish=True)
    directory = tmp_path / "test" / SEGMENT
    directory.mkdir(parents=True)
    artifact = directory / "fit.rrd"
    artifact.write_text("retain")
    state = Checkpoint("signature", "published", Timings(), layer_sha256="expected", readback_verified=True)
    monkeypatch.setattr(subprocess, "run", lambda *args, **_kwargs: subprocess.CompletedProcess(args, 0, stdout="wrong remote.rrd"))
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_publication(config, SEGMENT, state, np.array([0.0, 1.0]))
    assert not state.readback_verified
    with pytest.raises(ValueError, match="verified"):
        cleanup_published(config, directory, SEGMENT, state)
    assert artifact.read_text() == "retain"
