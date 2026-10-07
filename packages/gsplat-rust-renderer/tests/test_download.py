"""Download the complete corpus through per-scene archives and reuse finished trees."""
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("huggingface_hub")

from gsplat_rust_renderer import nerfbaselines
from gsplat_rust_renderer.apis import download


def test_all_downloads_sixteen_scene_archives_and_reuses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both kinds cover every scene once, and a second run performs no downloads."""
    requests: list[tuple[str, str]] = []
    root = tmp_path / "extracted"

    def cached_archive(repo_id: str, filename: str, *, repo_type: str | None = None) -> str:
        kind = "data" if repo_id == nerfbaselines.DATA_REPO else "pretrained"
        scene = Path(filename).stem
        assert repo_id == (nerfbaselines.DATA_REPO if kind == "data" else nerfbaselines.PRETRAINED_REPO)
        assert repo_type == ("dataset" if kind == "data" else None)
        assert filename == (f"blender/{scene}.zip" if kind == "data" else f"3dgs-mcmc/blender/{scene}.zip")
        requests.append((kind, scene))
        archive_path = tmp_path / f"{kind}-{scene}.zip"
        with ZipFile(archive_path, "w") as archive:
            archive.writestr(f"{scene}/payload.txt" if kind == "data" else "payload.txt", f"{kind}/{scene}")
        return str(archive_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", cached_archive)
    monkeypatch.setattr(download, "download_and_extract", partial(nerfbaselines.download_and_extract, root=root))
    config = download.Config(kind="all", scene="all")
    download.main(config)
    expected = [(kind, scene) for kind in ("data", "pretrained") for scene in nerfbaselines.BLENDER_SCENES]
    assert len(expected) == 16
    assert {scene for _, scene in expected} == {"lego", "hotdog", "chair", "drums", "ficus", "materials", "mic", "ship"}
    assert requests == expected
    payloads = [root / kind / scene / "payload.txt" for kind, scene in expected]
    assert [path.read_text() for path in payloads] == [f"{kind}/{scene}" for kind, scene in expected]
    mtimes = [path.stat().st_mtime_ns for path in payloads]
    download.main(config)
    assert requests == expected
    assert [path.stat().st_mtime_ns for path in payloads] == mtimes
    assert not list(root.glob("*/.*-extract-*"))


def test_tandt_download_extracts_inner_tree_and_reuses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Python task keeps the upstream archive layout and does not fetch twice."""
    requests: list[str] = []

    def retrieve(url: str, filename: Path) -> None:
        requests.append(url)
        with ZipFile(filename, "w") as archive:
            archive.writestr("tandt/truck/sparse/0/cameras.bin", b"camera fixture")

    monkeypatch.setattr(nerfbaselines, "urlretrieve", retrieve)
    target = tmp_path / "tandt"
    assert nerfbaselines.download_tandt(target) == target
    assert (target / "truck/sparse/0/cameras.bin").read_bytes() == b"camera fixture"
    assert nerfbaselines.download_tandt(target) == target
    assert requests == ["https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"]
