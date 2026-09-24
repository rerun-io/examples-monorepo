"""``dataforge-register`` layer fan-out, against a fake catalog client (no server)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pyarrow as pa
import pytest
import rerun.blueprint as rrb

pytest.importorskip("rerun.catalog", reason="catalog dependencies are optional outside the dataforge catalog environment")

from rerun.catalog import OnDuplicateSegmentLayer  # noqa: E402

from dataforge import paths  # noqa: E402
from dataforge.apis import register  # noqa: E402
from dataforge.apis.register import Config  # noqa: E402
from dataforge.datasets.msd import MsdConfig  # noqa: E402
from dataforge.datasets.robocap import RobocapConfig  # noqa: E402
from dataforge.datasets.show3d import Show3dConfig  # noqa: E402


@dataclass
class FakeRegistration:
    """Stand-in for the task ``DatasetEntry.register`` returns."""

    def wait(self) -> None:
        """The real call returns a task the caller must drive to completion."""


@dataclass
class FakeEntry:
    """Records every registration so a test can assert the per-layer fan-out."""

    registered: dict[str, list[str]] = field(default_factory=dict)
    duplicates: dict[str, Any] = field(default_factory=dict)
    """The ``on_duplicate`` policy each layer was registered under."""
    blueprints: list[tuple[str, bool]] = field(default_factory=list)
    opened_as: tuple[str, str] = ("", "")
    """``(catalog url, dataset name)`` the client was asked for."""
    blueprint_rows: dict[str, str] = field(default_factory=dict)
    """Registered blueprint entries (id -> storage url), as the hidden blueprint dataset lists them."""
    unregistered: list[str] = field(default_factory=list)
    """Blueprint entry ids dropped from the blueprint dataset."""

    def register(self, uris: list[str], *, layer_name: str, on_duplicate: Any) -> FakeRegistration:
        self.registered[layer_name] = list(uris)
        self.duplicates[layer_name] = on_duplicate
        return FakeRegistration()

    def default_blueprint(self) -> None:
        return None

    def default_segment_table_blueprint(self) -> None:
        return None

    def register_blueprint(self, uri: str, *, set_default: bool = False, segment_table: bool = False) -> None:
        self.blueprints.append((uri, segment_table))
        self.blueprint_rows[f"rec_new_{len(self.blueprints)}"] = uri  # the server lists new entries beside the old ones

    def blueprint_dataset(self) -> FakeEntry:
        """The real entry has a hidden per-dataset blueprint dataset; one object plays both here."""
        return self

    def unregister(self, *, segments_to_drop: list[str], layers_to_drop: list[str]) -> FakeRegistration:
        assert layers_to_drop == [], "a refresh drops whole blueprint entries, never single layers"
        self.unregistered.extend(segments_to_drop)
        return FakeRegistration()

    def segment_table(self) -> SimpleNamespace:
        """Just enough of the segment-table DataFrame: its column names, and the rows a blueprint dataset lists."""
        rows: pa.Table = pa.table(
            {"rerun_segment_id": list(self.blueprint_rows), "rerun_storage_urls": [[uri] for uri in self.blueprint_rows.values()]}
        )
        return SimpleNamespace(
            schema=lambda: SimpleNamespace(names=["rerun_segment_id", "property:RecordingInfo:name"]),
            select=lambda *columns: SimpleNamespace(to_arrow_table=lambda: rows.select(list(columns))),
        )


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


def test_refresh_blueprints_replaces_the_defaults_and_retires_the_old_entries(tmp_path: Path, catalog: FakeEntry, monkeypatch) -> None:
    """A refresh writes new dated files, makes them the defaults, unregisters every other entry, and deletes
    only the old files in this dataset's blueprint directory, also under the default relative output root."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", "rrd")  # relative, like the default; registered URLs are absolute
    root: Path = tmp_path / "rrd"
    make_rrds(root, paths.BASE_LAYER, ["robocap__a.rrd"])
    blueprint_dir: Path = (root / "blueprints").resolve()
    blueprint_dir.mkdir()
    old: list[Path] = [blueprint_dir / "robocap.rbl", blueprint_dir / "robocap-table.rbl"]
    foreign: Path = tmp_path / "elsewhere.rbl"
    for path in (*old, foreign):
        path.write_bytes(b"rbl")
    catalog.blueprint_rows = {"rec_default": old[0].as_uri(), "rec_table": old[1].as_uri(), "rec_foreign": foreign.as_uri()}
    register.main(Config(dataset=RobocapConfig(), refresh_blueprints=True))
    assert [segment_table for _, segment_table in catalog.blueprints] == [False, True]
    new: list[Path] = [Path(uri.removeprefix("file://")) for uri, _ in catalog.blueprints]
    assert all(path.exists() and path.parent == blueprint_dir and path not in old for path in new)
    assert sorted(catalog.unregistered) == ["rec_default", "rec_foreign", "rec_table"], "the two new entries stay"
    assert not any(path.exists() for path in old) and foreign.exists()


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


def test_registration_skips_duplicates_by_default(tmp_path: Path, catalog: FakeEntry, capsys) -> None:
    """Re-registering a corpus has to stay cheap and idempotent, so SKIP is the default."""
    make_rrds(tmp_path, paths.BASE_LAYER, ["robocap__a.rrd"])

    register.main(Config(dataset=RobocapConfig()))

    assert catalog.duplicates[paths.BASE_LAYER] == OnDuplicateSegmentLayer.SKIP
    assert "skipping duplicates" in capsys.readouterr().out


def test_replace_re_registers_a_regenerated_layer(tmp_path: Path, catalog: FakeEntry, capsys) -> None:
    """A rebuilt layer is a new file at a path the catalog already holds.

    SKIP then leaves the server serving the old registration and says nothing,
    so the rebuilt rrd is simply never seen — which is the failure ``--replace``
    exists for (``rm gt/*.rrd``, a convert, a ``register --replace``).
    """
    make_rrds(tmp_path, paths.BASE_LAYER, ["robocap__a.rrd"])
    make_rrds(tmp_path, paths.GT_LAYER, ["robocap__a.rrd"])

    register.main(Config(dataset=RobocapConfig(), replace=True))

    assert catalog.duplicates[paths.BASE_LAYER] == OnDuplicateSegmentLayer.REPLACE
    assert catalog.duplicates[paths.GT_LAYER] == OnDuplicateSegmentLayer.REPLACE
    assert "replacing duplicates" in capsys.readouterr().out


@pytest.mark.parametrize("show3d", [False, True])
def test_registers_only_dataset_layers(tmp_path: Path, catalog: FakeEntry, show3d: bool) -> None:
    name: str = "show3d" if show3d else "robocap"
    for layer in ("base", "gt", "sensor_metadata", "hand_pose", "captions"):
        make_rrds(tmp_path, layer, [f"{name}__a.rrd"])
    register.main(Config(dataset=Show3dConfig() if show3d else RobocapConfig()))
    assert list(catalog.registered) == (["base", "hand_pose", "captions"] if show3d else ["base", "gt", "sensor_metadata"])
