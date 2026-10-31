"""The main dev refresh replaces only its generated layers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from selfcap_curate.apis import dev_cycle

SEGMENT = "selfcap__2025-12-03__d31bf1da__ep001"


def test_main_dev_refresh_preserves_independent_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    dataset = client.get_dataset.return_value
    client.create_dataset.return_value = dataset
    names = ["base", "sapiens2", "calib", "egohand", "soma", "exo_landmarks"]
    dataset.segment_table.return_value.filter.return_value.select.return_value.to_arrow_table.return_value.to_pylist.return_value = [{
        "rerun_segment_id": SEGMENT,
        "rerun_layer_names": names,
        "rerun_storage_urls": [f"file:///source/{name}.rrd" for name in names],
    }]
    monkeypatch.setattr(dev_cycle, "CatalogClient", lambda _url: client)
    monkeypatch.setattr(dev_cycle.subprocess, "run", MagicMock())

    def post(config: dev_cycle.post.Config) -> None:
        for layer in ("align", "align/sapiens2", "keypoints", "calib/egohand", "align/exo_landmarks"):
            target = config.output_dir / layer / f"{SEGMENT}.rrd"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

    monkeypatch.setattr(dev_cycle.post, "main", post)
    dev_cycle.main(dev_cycle.Config(SEGMENT, "test", dataset="selfcap-post-dev", work_root=tmp_path))
    registered = [call.kwargs["layer_name"] for call in dataset.register.call_args_list]
    assert registered == ["base", "calib", "egohand_calib", "exo_landmarks"]


@pytest.mark.parametrize("destination", ["selfcap-post-ep1", "dataforge-selfcap"])
def test_dev_cycle_refuses_production(destination: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="destination"):
        dev_cycle.main(dev_cycle.Config(SEGMENT, "test", dataset=destination, work_root=tmp_path))
