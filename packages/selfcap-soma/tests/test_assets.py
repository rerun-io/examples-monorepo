"""SOMA body files come from a pinned Hugging Face revision unless a local directory overrides them."""

from pathlib import Path
from typing import Any

import pytest

from selfcap_soma import assets


def test_the_default_downloads_the_pinned_body_files(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(assets, "snapshot_download", lambda **kwargs: calls.append(kwargs) or "/cache/soma")
    assets.soma_assets.cache_clear()
    assert assets.soma_assets() == Path("/cache/soma")
    assert calls == [
        {
            "repo_id": "nvidia/SOMA-X",
            "revision": "f81a01796f652260ff930d5318d6c6b845a505ed",
            "allow_patterns": ["SOMA_neutral.npz", "SOMA_procedural_transforms.json", "SOMA_template_rig.usda", "LICENSE"],
        }
    ]
    assets.soma_assets.cache_clear()


def test_a_local_directory_overrides_the_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(**_: Any) -> str:
        raise AssertionError("a local directory must not download")

    monkeypatch.setattr(assets, "snapshot_download", refuse)
    assert assets.soma_assets(Path("/local/soma")) == Path("/local/soma")
