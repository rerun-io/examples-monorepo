"""Restore validates the destination before it can write to a catalog."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("rerun.catalog", reason="requires the dataforge catalog environment")

from serde.json import to_json

from dataforge.apis.catalog_snapshot import CatalogSnapshot, restore_catalog


def test_restore_refuses_the_snapshot_source_before_connecting(tmp_path: Path) -> None:
    path: Path = tmp_path / "snapshot.json"
    path.write_text(to_json(CatalogSnapshot("rerun+http://source:51235", [])))
    with pytest.raises(ValueError, match="must differ"):
        restore_catalog(path, "rerun+http://source:51235/")


def test_restore_rejects_unknown_dataset_names_before_connecting(tmp_path: Path) -> None:
    path: Path = tmp_path / "snapshot.json"
    path.write_text(to_json(CatalogSnapshot("rerun+http://source:51235", [])))
    with pytest.raises(ValueError, match="Unknown snapshot entries"):
        restore_catalog(path, "rerun+http://destination:51241", ("missing",))


def test_restore_allows_explicit_same_url_recovery_for_a_restarted_server(tmp_path: Path) -> None:
    path: Path = tmp_path / "snapshot.json"
    path.write_text(to_json(CatalogSnapshot("rerun+http://127.0.0.1:1", [])))
    with patch("dataforge.apis.catalog_snapshot.CatalogClient") as connect:
        restore_catalog(path, "rerun+http://127.0.0.1:1", allow_source_url=True)
        connect.assert_called_once_with("rerun+http://127.0.0.1:1")
