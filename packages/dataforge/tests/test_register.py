"""``dataforge-register`` layer fan-out, against a fake catalog client (no server)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest
import rerun.blueprint as rrb

from dataforge import paths
from dataforge.apis import register
from dataforge.apis.register import Config
from dataforge.datasets.msd import MsdConfig
from dataforge.datasets.robocap import RobocapConfig


@dataclass
class FakeRegistration:
    """Stand-in for the task ``DatasetEntry.register`` returns."""

    def wait(self) -> None:
        """The real call returns a task the caller must drive to completion."""


@dataclass
class FakeEntry:
    """Records every registration so a test can assert the per-layer fan-out."""

    registered: dict[str, list[str]] = field(default_factory=dict)
    blueprints: list[tuple[str, bool]] = field(default_factory=list)
    opened_as: tuple[str, str] = ("", "")
    """``(catalog url, dataset name)`` the client was asked for."""

    def register(self, uris: list[str], *, layer_name: str, on_duplicate: Any) -> FakeRegistration:
        self.registered[layer_name] = list(uris)
        return FakeRegistration()

    def default_blueprint(self) -> None:
        return None

    def default_segment_table_blueprint(self) -> None:
        return None

    def register_blueprint(self, uri: str, *, set_default: bool = False, segment_table: bool = False) -> None:
        self.blueprints.append((uri, segment_table))


@dataclass
class FakeClient:
    """``CatalogClient`` replacement that hands out one shared entry."""

    url: str
    shared: ClassVar[FakeEntry] = FakeEntry()
    """The entry every instance returns; the fixture replaces it per test."""

    def create_dataset(self, name: str, *, exist_ok: bool = False) -> FakeEntry:
        FakeClient.shared.opened_as = (self.url, name)
        return FakeClient.shared


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch) -> FakeEntry:
    """Point the output root at a tmp tree and swap the catalog types for fakes.

    Both names are patched, not just the client: ``main`` annotates its locals
    (``client: CatalogClient``, ``entry: DatasetEntry``) and beartype checks
    every one of those under ``PIXI_DEV_MODE``, so a fake that the annotation
    does not admit fails before the code under test runs.
    """
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    entry: FakeEntry = FakeEntry()
    monkeypatch.setattr(FakeClient, "shared", entry)
    monkeypatch.setattr(register, "CatalogClient", FakeClient)
    monkeypatch.setattr(register, "DatasetEntry", FakeEntry)
    return entry


def make_rrds(root: Path, layer: str, names: list[str]) -> list[Path]:
    """Create empty rrds for one layer, plus a decoy another dataset owns."""
    layer_root: Path = root / layer
    layer_root.mkdir(parents=True, exist_ok=True)
    (layer_root / "selfcap__other.rrd").write_bytes(b"")
    written: list[Path] = []
    for name in names:
        target: Path = layer_root / name
        target.write_bytes(b"")
        written.append(target)
    return written


def test_registers_base_and_gt_as_separate_layers(tmp_path: Path, catalog: FakeEntry) -> None:
    base: list[Path] = make_rrds(tmp_path, paths.BASE_LAYER, ["robocap__a.rrd", "robocap__b.rrd"])
    gt: list[Path] = make_rrds(tmp_path, paths.GT_LAYER, ["robocap__a.rrd"])
    register.main(Config(dataset=RobocapConfig()))
    assert catalog.registered[paths.BASE_LAYER] == [path.resolve().as_uri() for path in base]
    assert catalog.registered[paths.GT_LAYER] == [path.resolve().as_uri() for path in gt]
    # The dataset's registry key is its catalog name, and the decoy rrds are another dataset's.
    assert catalog.opened_as == ("rerun+http://127.0.0.1:51235", "robocap")


def test_gt_is_optional(tmp_path: Path, catalog: FakeEntry) -> None:
    make_rrds(tmp_path, paths.BASE_LAYER, ["robocap__a.rrd"])
    register.main(Config(dataset=RobocapConfig()))
    assert paths.GT_LAYER not in catalog.registered


def test_base_is_required(tmp_path: Path, catalog: FakeEntry) -> None:
    make_rrds(tmp_path, paths.GT_LAYER, ["robocap__a.rrd"])
    with pytest.raises(FileNotFoundError, match=paths.BASE_LAYER):
        register.main(Config(dataset=RobocapConfig()))


def test_blueprints_are_registered_once_each(tmp_path: Path, catalog: FakeEntry) -> None:
    make_rrds(tmp_path, paths.BASE_LAYER, ["robocap__a.rrd"])
    register.main(Config(dataset=RobocapConfig()))
    assert [segment_table for _, segment_table in catalog.blueprints] == [False, True]
    assert all(Path(paths.blueprint_path(tmp_path, "robocap", segment_table=table)).exists() for table in (False, True))


def test_reports_a_count_per_layer(tmp_path: Path, catalog: FakeEntry, capsys) -> None:
    make_rrds(tmp_path, paths.BASE_LAYER, ["robocap__a.rrd", "robocap__b.rrd"])
    make_rrds(tmp_path, paths.GT_LAYER, ["robocap__a.rrd"])
    register.main(Config(dataset=RobocapConfig()))
    printed: str = capsys.readouterr().out
    assert f"2 {paths.BASE_LAYER}" in printed
    assert f"1 {paths.GT_LAYER}" in printed


def test_a_per_device_dataset_registers_only_its_own_two_layers(tmp_path: Path, catalog: FakeEntry, capsys) -> None:
    """msd derives ``msd-<device>``, so one device's layers must not pick up another's."""
    make_rrds(tmp_path, paths.BASE_LAYER, ["msd-g2__MGO_others__MGO09.rrd", "msd-index__MIO_others__MIO09.rrd"])
    make_rrds(tmp_path, paths.GT_LAYER, ["msd-g2__MGO_others__MGO09.rrd", "msd-index__MIO_others__MIO09.rrd"])

    register.main(Config(dataset=MsdConfig(device="g2")))

    assert catalog.opened_as[1] == "msd-g2"
    for layer in (paths.BASE_LAYER, paths.GT_LAYER):
        assert [Path(uri).name for uri in catalog.registered[layer]] == ["msd-g2__MGO_others__MGO09.rrd"]
    printed: str = capsys.readouterr().out
    assert f"1 {paths.BASE_LAYER}" in printed
    assert f"1 {paths.GT_LAYER}" in printed


def test_default_blueprint_is_a_blueprint() -> None:
    """Guards the fake above: the real entry receives a saved rrb.Blueprint."""
    assert isinstance(RobocapConfig().setup().default_blueprint(), rrb.Blueprint)
