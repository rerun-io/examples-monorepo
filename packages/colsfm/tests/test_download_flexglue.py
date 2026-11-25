"""`colsfm.apis.download_flexglue` into a model directory that every worktree on the host shares."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from colsfm.apis import download_flexglue

PAYLOAD: bytes = b"a flexglue graph"


@pytest.fixture
def one_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """One graph to fetch, served from memory."""
    name: str = "flexglue_descriptor.onnx"
    monkeypatch.setattr(download_flexglue, "MODELS", {name: hashlib.sha256(PAYLOAD).hexdigest()})
    monkeypatch.setattr(download_flexglue.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(PAYLOAD))
    return name


def test_a_download_stages_in_its_own_file_and_leaves_another_downloads_alone(tmp_path: Path, one_model: str) -> None:
    """Two worktrees share the model directory: a second download must not write into, rename or delete the file a
    first one is still filling (one fixed `.onnx.part` name gave a sha256 mismatch or a FileNotFoundError)."""
    other: Path = tmp_path / "flexglue_descriptor.onnx.part"
    other.write_bytes(b"half of another download")
    download_flexglue.main(download_flexglue.Config(output_dir=tmp_path))
    assert (tmp_path / one_model).read_bytes() == PAYLOAD
    assert other.read_bytes() == b"half of another download"
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([one_model, other.name]), "no staging file left"


def test_a_wrong_sum_keeps_nothing(tmp_path: Path, one_model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download_flexglue, "MODELS", {one_model: "0" * 64})
    with pytest.raises(RuntimeError, match="sha256"):
        download_flexglue.main(download_flexglue.Config(output_dir=tmp_path))
    assert list(tmp_path.iterdir()) == []
