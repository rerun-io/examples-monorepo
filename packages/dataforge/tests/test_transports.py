from pathlib import Path

import pytest

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
    for fetch in (hf_fetch, http_fetch, gdrive_fetch):
        with pytest.raises(NotImplementedError):
            fetch()
