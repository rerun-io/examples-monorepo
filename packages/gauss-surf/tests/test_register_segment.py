"""Extended gauss-surf catalog blueprint artifact tests."""

from pathlib import Path
from typing import Any

from gauss_surf.apis.register_segment import existing_recovery_rrds, recovery_rrds, write_extended_blueprints


def test_write_extended_blueprints_writes_both_orientations(tmp_path: Path) -> None:
    blueprint_paths: list[Path] = write_extended_blueprints(tmp_path, "arkitscenes-v2")

    assert [path.name for path in blueprint_paths] == [
        "arkitscenes-v2-gauss-surf-landscape.rbl",
        "arkitscenes-v2-gauss-surf-portrait.rbl",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in blueprint_paths)


def test_recovery_rrds_include_the_independent_splat_layers() -> None:
    paths: list[Path] = recovery_rrds("47115416")

    assert Path("data/splat_depth/47115416.rrd") in paths
    assert Path("data/splat_triage/47115416.rrd") in paths


def test_existing_recovery_rrds_skip_outputs_not_generated_yet(tmp_path: Path, monkeypatch: Any) -> None:
    """Fresh onboarding recovers PromptDA without requiring future pipeline layers."""
    monkeypatch.chdir(tmp_path)
    promptda_path: Path = Path("data/promptda/segment.rrd")
    promptda_path.parent.mkdir(parents=True)
    promptda_path.write_bytes(b"promptda")

    paths: list[Path] = existing_recovery_rrds("segment")

    assert paths == [promptda_path]
