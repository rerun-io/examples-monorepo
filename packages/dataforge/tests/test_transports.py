from pathlib import Path
from typing import Any

import pytest

from dataforge import transports
from dataforge.transports import gdrive_fetch, hf_fetch, http_fetch, local_verify


def test_local_verify_reports_missing_globs(tmp_path: Path) -> None:
    (tmp_path / "sess").mkdir()
    (tmp_path / "sess" / "video_dev0.mp4").write_bytes(b"x")
    missing: list[str] = local_verify(tmp_path, required=("sess/video_*.mp4", "sess/IMUWriter_*.db"))
    assert missing == ["sess/IMUWriter_*.db"]


def test_local_verify_ok_when_all_present(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"x")
    assert local_verify(tmp_path, required=("a.txt",)) == []


def test_local_verify_missing_root(tmp_path: Path) -> None:
    missing: list[str] = local_verify(tmp_path / "nope", required=("a.txt",))
    assert missing == ["a.txt"]


def test_unbuilt_transports_raise() -> None:
    for fetch in (http_fetch, gdrive_fetch):
        with pytest.raises(NotImplementedError):
            fetch()


@pytest.fixture
def recorded_snapshot(monkeypatch) -> dict[str, Any]:
    """Replace ``snapshot_download`` with a recorder that touches one file."""
    recorded: dict[str, Any] = {}

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        recorded["repo_id"] = repo_id
        recorded.update(kwargs)
        local_dir: Path = Path(kwargs["local_dir"])
        landed: Path = local_dir / "MI_valid_01" / "camera_calibration.json"
        landed.parent.mkdir(parents=True, exist_ok=True)
        landed.write_text("{}")
        return str(local_dir)

    monkeypatch.setattr(transports, "snapshot_download", fake_snapshot_download)
    return recorded


def test_hf_fetch_lands_files_at_local_dir(tmp_path: Path, recorded_snapshot: dict[str, Any]) -> None:
    returned: Path = hf_fetch("collabora/monado-slam-datasets", allow_patterns=("MI_valid_01/**",), local_dir=tmp_path)
    assert returned == tmp_path
    # local_dir mode, not the symlinked cache tree: the file sits at <local_dir>/<path-in-repo>.
    assert (tmp_path / "MI_valid_01" / "camera_calibration.json").is_file()
    assert recorded_snapshot["repo_id"] == "collabora/monado-slam-datasets"
    assert recorded_snapshot["repo_type"] == "dataset"
    assert recorded_snapshot["allow_patterns"] == ["MI_valid_01/**"]
    assert recorded_snapshot["local_dir"] == str(tmp_path)
    assert recorded_snapshot["revision"] is None


def test_hf_fetch_passes_the_revision_through(tmp_path: Path, recorded_snapshot: dict[str, Any]) -> None:
    hf_fetch("collabora/monado-slam-datasets", allow_patterns=("*.json",), local_dir=tmp_path, repo_type="model", revision="refs/pr/1")
    assert recorded_snapshot["repo_type"] == "model"
    assert recorded_snapshot["revision"] == "refs/pr/1"
