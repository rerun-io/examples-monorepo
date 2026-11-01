from pathlib import Path

import pytest

import selfcap_curate.apis as api_inputs
from selfcap_curate.apis.calibrate import Config, main


def test_calibrate_reads_the_named_catalog_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Like align, calibrate reads the aligned bases from a named dataset, e.g. a walkthrough's local catalog.
    opened: list[tuple[str, str]] = []

    class Client:
        def __init__(self, url: str) -> None:
            self.url = url

        def get_dataset(self, name: str) -> None:
            opened.append((self.url, name))
            raise LookupError("stop after the catalog lookup")

    monkeypatch.setattr(api_inputs, "CatalogClient", Client)
    config = Config(
        output_dir=tmp_path / "out", segment_ids=("selfcap__2025-12-03__d31bf1da__ep001",),
        catalog_url="rerun+http://127.0.0.1:51301", dataset="selfcap-e2e",
    )
    with pytest.raises(LookupError):
        main(config)
    assert opened == [("rerun+http://127.0.0.1:51301", "selfcap-e2e")]
    assert not (tmp_path / "out").exists()
