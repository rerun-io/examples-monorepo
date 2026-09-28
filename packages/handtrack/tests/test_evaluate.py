"""Shards leave publication to the explicit aggregation step."""

import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from rerun.catalog import DatasetEntry

from handtrack.apis import evaluate
from handtrack.apis.run_pipeline import Networks, RunConfig


def test_only_aggregation_publishes_reports_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = RunConfig(output_root=tmp_path, random_weights=True, device="cpu", hand_modes=(), shards=4)
    monkeypatch.setattr(evaluate.rr.catalog, "CatalogClient", lambda _url: Mock(get_dataset=Mock(return_value=Mock(spec=DatasetEntry))))
    monkeypatch.setattr(evaluate, "select_segments", lambda config, entry: ())
    monkeypatch.setattr(evaluate, "load_networks", lambda config, device: Networks(None, None, "random", "random"))
    root = tmp_path / config.name
    reports = ("summary.json", "summary.md", "detnet_metrics.json")
    for shard in range(4):
        evaluate.main(evaluate.EvaluateConfig(replace(config, shard=shard)))
    assert all(not (root / name).exists() for name in reports)
    replaced: list[str] = []
    original = os.replace

    def record_replace(source: str, target: Path) -> None:
        assert Path(source).parent == target.parent
        assert Path(source).read_bytes()
        replaced.append(target.name)
        original(source, target)

    monkeypatch.setattr(os, "replace", record_replace)
    evaluate.main(evaluate.EvaluateConfig(config, aggregate_only=True))
    assert set(replaced) == set(reports)
    assert all((root / name).read_text() for name in reports)
