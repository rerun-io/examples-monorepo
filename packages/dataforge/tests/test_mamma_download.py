"""MAMMA download: verifying the release in place, and the fetch proof against a fake MPI server (no network, no credentials)."""

from pathlib import Path

import numpy as np
import pytest

from dataforge.datasets import mamma_download
from dataforge.datasets.mamma_download import fetch_sequence, verify_release
from dataforge.datasets.mamma_source import discover_sequences


def release(root: Path) -> Path:
    """Two markerless-dance sequences with every file in place, as small stand-ins."""
    for name in ("050825_Swing_A_1", "050825_Swing_B_1"):
        directory: Path = root / "mamma_markerless_dance" / name
        for sub in ("meta", "pred", "videos_crf16"):
            (directory / sub).mkdir(parents=True)
        np.savez(directory / "meta/global.npz", fps=np.array(30))
        for camera in ("IOI_01", "IOI_02"):
            np.savez(directory / f"meta/{camera}.npz", cam_name=np.array(camera))
            (directory / f"videos_crf16/{camera}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(64))
        np.savez(directory / "pred/params_00.npz", betas=np.zeros(16))
    return root


def test_a_complete_release_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(mamma_download.EXPECTED_SEQUENCES, "mamma_markerless_dance", 2)
    sequences, problems = verify_release(release(tmp_path), "markerless-dance")
    assert len(sequences) == 2 and problems == []


def test_verify_names_missing_empty_and_html_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(mamma_download.EXPECTED_SEQUENCES, "mamma_markerless_dance", 3)
    root: Path = release(tmp_path)
    (root / "mamma_markerless_dance/050825_Swing_A_1/videos_crf16/IOI_02.mp4").unlink()
    (root / "mamma_markerless_dance/050825_Swing_A_1/pred/params_00.npz").write_bytes(b"")
    (root / "mamma_markerless_dance/050825_Swing_B_1/videos_crf16/IOI_01.mp4").write_bytes(b"<!DOCTYPE html><html>login</html>")
    _, problems = verify_release(root, "markerless-dance")
    text: str = "\n".join(problems)
    assert "Swing_A_1: cameras without a video left out: ['IOI_02']" in text
    assert "Swing_A_1/pred/params_00.npz: empty" in text
    assert "Swing_B_1/videos_crf16/IOI_01.mp4: an HTML page" in text
    assert "mamma_markerless_dance: 2 sequences, the release has 3" in text


def test_the_fetch_proof_compares_every_file_with_the_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root: Path = release(tmp_path / "release")
    sequence = discover_sequences(root, "markerless-dance")[0][0]
    served: dict[str, bytes] = {}

    def post(url: str, data: dict[str, str], target: Path) -> None:
        assert data == {"username": "user", "password": "secret"}
        relative: str = url.split("sfile=datasets/", 1)[1]
        target.write_bytes(served.get(relative, (root / relative).read_bytes()))

    monkeypatch.setattr(mamma_download, "post_to_file", post)
    monkeypatch.setenv("MAMMA_USERNAME", "user")
    monkeypatch.setenv("MAMMA_PASSWORD", "secret")
    into: Path = tmp_path / "fetched"
    assert fetch_sequence(root, sequence, into) == []
    assert (into / "mamma_markerless_dance/050825_Swing_A_1/videos_crf16/IOI_01.mp4").is_file()

    served["mamma_markerless_dance/050825_Swing_A_1/meta/global.npz"] = b"something else"
    mismatches: list[str] = fetch_sequence(root, sequence, tmp_path / "again")
    assert mismatches == ["mamma_markerless_dance/050825_Swing_A_1/meta/global.npz: 14 bytes fetched differ from the release copy"]


def test_the_fetch_needs_the_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root: Path = release(tmp_path / "release")
    monkeypatch.delenv("MAMMA_USERNAME", raising=False)
    monkeypatch.delenv("MAMMA_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="MAMMA_USERNAME and MAMMA_PASSWORD"):
        fetch_sequence(root, discover_sequences(root, "markerless-dance")[0][0], tmp_path / "fetched")


@pytest.mark.parametrize("existing", [False, True])
def test_failed_post_cleans_staging_and_keeps_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool) -> None:
    root = release(tmp_path / "release")
    sequence = discover_sequences(root, "markerless-dance")[0][0]
    source = mamma_download.sequence_files(sequence)[0]
    into = tmp_path / "fetched"
    target = into / source.relative_to(root)
    target.parent.mkdir(parents=True)
    if existing:
        target.write_bytes(b"previous complete file")

    def post(url: str, data: dict[str, str], target: Path) -> None:
        target.write_bytes(b"partial transfer")
        raise OSError("connection lost")

    monkeypatch.setattr(mamma_download, "post_to_file", post)
    monkeypatch.setenv("MAMMA_USERNAME", "user")
    monkeypatch.setenv("MAMMA_PASSWORD", "secret")
    with pytest.raises(OSError, match="connection lost"):
        fetch_sequence(root, sequence, into)
    assert list(target.parent.iterdir()) == ([target] if existing else [])
    if existing:
        assert target.read_bytes() == b"previous complete file"
